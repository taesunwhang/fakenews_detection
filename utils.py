import gc
import numpy as np
import math

import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
from tensorflow.python.client import device_lib


def sequence_feature(outputs, lengths):
  batch_size = tf.shape(outputs)[0]

  features_fw_bw_outputs = tf.split(outputs, 2, axis=-1)

  batch_nums = tf.range(0, batch_size, dtype=tf.int32)

  first_ids = tf.stack((batch_nums, tf.zeros(tf.shape(batch_nums), tf.int32)), axis=1)
  targets = lengths - 1

  last_ids = tf.stack((batch_nums, targets), axis=1)
  # We need the last output of each sentence (based on length) from Forward RNN outputs.
  features_fw = tf.gather_nd(features_fw_bw_outputs[0], last_ids)
  # We need the first output of Backward RNN outputs.
  features_bw = tf.gather_nd(features_fw_bw_outputs[1], first_ids)

  return features_fw, features_bw

def stack_bidirectional_rnn(
    cell: str,
    num_layers: int,
    num_units: int,
    inputs: tf.Tensor,
    sequence_length: tf.Tensor,
    input_dropout_keep_prob: tf.Tensor = None,
    output_dropout_keep_prob: tf.Tensor = None,
    residual: bool = False,
    time_major: bool = False,
    dtype=tf.float32,
    state_merge: str = "sum"
):
  # Whether to use concatenation or summation to merge the outputs of
  # forward and backward RNNs.
  if state_merge not in ["sum", "concat"]:
    raise ValueError("Unknown state merge method: %s" % state_merge)

  with tf.variable_scope("stack-bi-rnn"):
    # Use residual connection only when there are more than 1 layers of RNN.
    cell_type = cell.upper()
    layer_out = inputs
    cudnn = "CUDNN" in cell_type

    if state_merge == "concat":
      rnn_dim = round(num_units / 2)
    else:
      rnn_dim = num_units

    if cudnn:
      kernel_stddev = math.sqrt(1.0 / rnn_dim)
      rnn_dim = tf.constant(rnn_dim, dtype=tf.int32)

    if not time_major:
      # Use time-major internally.
      # CudnnRNN is time-major, but everything else is batch-major.
      # Need to transpose the tensor.
      layer_out = tf.transpose(layer_out, [1, 0, 2])
    # shape = [time, batch, input_dim]
    time_axis = 0
    batch_axis = 1

    for i in range(num_layers):
      with tf.variable_scope("RNN_layer%02d" % (i + 1)):
        original_input = layer_out

        # Apply dropout to the input of each RNN layer.
        if input_dropout_keep_prob is not None:
          # Apply variational recurrent dropout on the input of each RNN layer.
          input_shape = tf.shape(layer_out)
          noise_shape = [1, input_shape[1], input_shape[2]]
          layer_out = tf.nn.dropout(
            x=layer_out,
            keep_prob=input_dropout_keep_prob,
            noise_shape=noise_shape
          )

        # Reverse input for backward RNN.
        inputs_rev = tf.reverse_sequence(
          input=layer_out,
          seq_lengths=sequence_length,
          seq_axis=time_axis,
          batch_axis=batch_axis,
          name="inputs_reversed"
        )

        if cudnn:
          if cell_type == "CUDNNLSTM":
            rnn_builder = tf.contrib.cudnn_rnn.CudnnLSTM
          elif cell_type == "CUDNNGRU":
            rnn_builder = tf.contrib.cudnn_rnn.CudnnGRU
          else:
            raise ValueError("Unknown RNN cell type: %s" % cell)

          # !!! Bidirectional RNN with "direction" parameter gives NaN loss or low accuracy.
          rnn_layer_forward = rnn_builder(
            num_layers=1,
            num_units=rnn_dim,
            # input_mode=cudnn_rnn.CUDNN_INPUT_AUTO_MODE,
            direction=cudnn_rnn.CUDNN_RNN_UNIDIRECTION,
            dropout=0,
            name="CudnnRNN_FW",
            # kernel_initializer=tf.initializers.variance_scaling(
            #     scale=1.0, mode="fan_avg", distribution="normal")
            kernel_initializer=tf.truncated_normal_initializer(stddev=kernel_stddev),
            bias_initializer=tf.zeros_initializer()
          )
          rnn_layer_backward = rnn_builder(
            num_layers=1,
            num_units=rnn_dim,
            # input_mode=cudnn_rnn.CUDNN_INPUT_AUTO_MODE,
            # !!! Other values than default for the "input_mode" parameter throuws error.
            direction=cudnn_rnn.CUDNN_RNN_UNIDIRECTION,
            dropout=0,
            name="CudnnRNN_BW",
            # kernel_initializer=tf.initializers.variance_scaling(
            #     scale=1.0, mode="fan_avg", distribution="normal")
            # tf.initializers.variance_scaling throws error.
            kernel_initializer=tf.truncated_normal_initializer(stddev=kernel_stddev),
            bias_initializer=tf.zeros_initializer()
            # Not specifying "bias_initializer" results in an error.
          )
          layer_out_forward, _ = rnn_layer_forward(layer_out)
          layer_out_backward_rev, _ = rnn_layer_backward(inputs_rev)

        else:
          # BiRNN layers.
          if cell_type == "LSTM":
            builder = tf.contrib.rnn.LSTMBlockCell
          elif cell_type == "GRU":
            # builder = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell
            # builder = tf.contrib.rnn.GRUCell
            builder = tf.contrib.rnn.GRUBlockCellV2
          else:
            raise ValueError("Unknown RNN cell type: %s" % cell)

          with tf.variable_scope("RNN_FW"):
            cell_fw = builder(num_units=rnn_dim, name="cell_fw_%d" % (i + 1))
            layer_out_forward, _ = tf.nn.dynamic_rnn(
              cell=cell_fw,
              inputs=layer_out,
              dtype=dtype,
              time_major=True
            )
          with tf.variable_scope("RNN_BW"):
            cell_bw = builder(num_units=rnn_dim, name="cell_bw_%d" % (i + 1))
            layer_out_backward_rev, _ = tf.nn.dynamic_rnn(
              cell=cell_bw,
              inputs=inputs_rev,
              dtype=dtype,
              time_major=True
            )

        # Reverse the output back.
        layer_out_backward = tf.reverse_sequence(
          input=layer_out_backward_rev,
          seq_lengths=sequence_length,
          seq_axis=time_axis,
          batch_axis=batch_axis,
          name="backward_rnn_output"
        )

        if state_merge == "concat":
          # Concatenate forward and backward outputs.
          layer_out = tf.concat([layer_out_forward, layer_out_backward], axis=2)

        else:
          # Add forward and backward outputs.
          layer_out = tf.add(layer_out_forward, layer_out_backward)

        # Apply dropout to the output of each RNN layer.
        if output_dropout_keep_prob is not None:
          # Apply variational recurrent dropout on the input of each RNN layer.
          input_shape = tf.shape(layer_out)
          noise_shape = [1, input_shape[1], input_shape[2]]
          layer_out = tf.nn.dropout(
            x=layer_out,
            keep_prob=output_dropout_keep_prob,
            noise_shape=noise_shape
          )

        if residual:
          in_dim = original_input.get_shape().as_list()[-1]
          out_dim = layer_out.get_shape().as_list()[-1]
          if in_dim != out_dim:
            original_input = tf.layers.dense(
              original_input,
              units=out_dim,
              use_bias=False,
              name="residual_projection"
            )
          layer_out = tf.add(layer_out, original_input)

    # Mask out paddings.
    mask = tf.sequence_mask(sequence_length, dtype=dtype, maxlen=tf.shape(inputs)[1])
    mask = tf.transpose(mask)
    mask = tf.expand_dims(mask, axis=2)
    layer_out = tf.multiply(layer_out, mask)

    if not time_major:
      # Transpose the tensor back to batch-major.
      layer_out = tf.transpose(layer_out, [1, 0, 2])

  return layer_out

class TensorBoardSummaryWriter:
  def __init__(self, root_dir:str, sess:tf.Session = None, graph:tf.Graph = None):
    self._constant_phs = {}
    self._constant_summaries = {}
    self._destroy_session = (sess is None)

    if sess is None:
      sess_config = tf.ConfigProto(
          device_count={'CPU': 1, 'GPU': 0},
          allow_soft_placement=True,
          log_device_placement=False
      )
      sess_config.gpu_options.visible_device_list = ""
      sess_config.gpu_options.per_process_gpu_memory_fraction = 0.0000001
      sess_config.gpu_options.allow_growth = True

      with tf.device("/cpu:0"):
        self._session = tf.Session(config=sess_config)
    else:
      self._session = sess

    self._summary_writer = tf.summary.FileWriter(root_dir, graph)

  def add_tensor_summary(self, value, step:int):
      self._summary_writer.add_summary(value, step)
      self._summary_writer.flush()

  def add_summary(self, name:str, value, step:int, dtype=tf.float32):
    summary_str = self._constant_to_summary_str(name, value, dtype)
    self._summary_writer.add_summary(summary_str, step)
    self._summary_writer.flush()

  def _constant_to_summary_str(self, name: str, value, dtype) -> str:
    """
    Make a python variable to a summary string.
    """
    with tf.device("/cpu:0"):
      target_ph = None
      if name in self._constant_phs:
        target_ph = self._constant_phs[name]
      else:
        target_ph = tf.placeholder(dtype=dtype, name=name, shape=[])
        self._constant_phs[name] = target_ph
      if self._session is None:
        # self._logger.error("[E] Session is none.")
        return ""

      target_summary = None
      if name in self._constant_summaries:
        target_summary = self._constant_summaries[name]
      else:
        target_summary = tf.summary.scalar(name, target_ph)
        self._constant_summaries[name] = target_summary
      feed_dict = {
        target_ph: value
      }
      summary_str = self._session.run(target_summary, feed_dict=feed_dict)

    return summary_str

  def close(self):
    self._summary_writer.flush()
    self._summary_writer.close()
    if self._destroy_session:
      self._session.close()
      tf.reset_default_graph()
    gc.collect()


def check_available_gpus():
  local_devices = device_lib.list_local_devices()
  gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
  gpu_num = len(gpu_names)

  print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))

  return gpu_num

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, var in grad_and_vars:
      if g is None:
        print(g,var)

      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads