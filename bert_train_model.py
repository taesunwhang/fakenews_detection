import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import logging
from datetime import datetime
import time
from tqdm import tqdm

from utils import *

from bert_data_process import *
from bert_model import tokenization, optimization, modeling_base
from evaluate import get_f1_score


class BERTModel(object):
  def __init__(self, hparams):
    self.hparams = hparams
    self._logger = logging.getLogger(__name__)
    self.train_setup_vars = dict()
    self.train_setup_vars["on_training"] = False
    self.train_setup_vars["do_evaluate"] = False
    self.train_setup_vars["is_train_continue"] = False
    self.bert_config = modeling_base.BertConfig.from_json_file(self.hparams.bert_config_dir)

    self._make_data_processor()

  def _make_data_processor(self):
    processors = {
      "fnc": FNCProcessor,
    }
    self.tokenizer = tokenization.FullTokenizer(self.hparams.vocab_dir, self.hparams.do_lower_case)

    data_dir = self.hparams.data_dir
    self.processor = processors[self.hparams.task_name](self.hparams, self.tokenizer)
    self.train_examples, self.label_dict = self.processor.get_train_examples(data_dir)
    self.test_examples = self.processor.get_test_examples(data_dir)
    self.label_list = self.processor.get_labels()

    self.num_train_steps = int(
      len(self.train_examples) / self.hparams.train_batch_size * self.hparams.num_epochs)
    self.warmup_proportion = 0.1
    self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion)

  def _make_placeholders(self):
    self.input_ids_ph = tf.placeholder(tf.int32, shape=[None, self.hparams.max_seq_length], name="input_ids_ph")
    self.input_mask_ph = tf.placeholder(tf.int32, shape=[None, self.hparams.max_seq_length], name="input_mask_ph")
    self.segment_ids_ph = tf.placeholder(tf.int32, shape=[None, self.hparams.max_seq_length], name="segment_ids_ph")

    self.label_ids_ph = tf.placeholder(tf.int32, shape=[None], name="label_ids_ph")

    self.dropout_prob_ph = tf.placeholder(tf.float32, shape=[], name="dropout_prob_ph")

  def _select_train_variables(self):
    # not keep training(loop) | training from pretrained_file(checkpoint)
    if not self.train_setup_vars["on_training"]:
      self.need_to_initialize_vars = []
      vars_in_checkpoint = tf.train.list_variables(self.hparams.init_checkpoint)
      checkpoint_vars = []
      for var_name, _ in vars_in_checkpoint:
        checkpoint_vars.append(var_name)

      var_dict = dict()
      for var in tf.trainable_variables():
        if var.name[:-2] not in checkpoint_vars:
          print("not included variable : ", var.name)
          self.need_to_initialize_vars.append(var)
          continue
        # print(var)
        var_dict[var.name[:-2]] = var

      if not self.train_setup_vars["is_train_continue"] and not self.train_setup_vars["do_evaluate"]:
        saver = tf.train.Saver(var_dict)
        saver.restore(self.sess, self.hparams.init_checkpoint)
        self._logger.info("Restoring Session from checkpoint complete!")

    # all bert pretrained var names
    self.pretrained_all_var_names = []
    # half of the transformer layers which will not be trained during the fine-tuning training
    self.pretrained_not_train_var_names = []
    if len(self.hparams.train_transformer_layer) == 0:
      print("not train transformer_layer", "-"*50)
      for var in tf.trainable_variables():
        if var not in self.need_to_initialize_vars:
          self.pretrained_not_train_var_names.append(var)
    else:
      for var in tf.trainable_variables():
        self.pretrained_all_var_names.append(var)
        print(var)
        var_name_split = var.name.split("/")
        if len(var_name_split) > 1:
          if var_name_split[1] == "encoder":
            layer_num = int(var_name_split[2].split("_")[-1])
            if layer_num not in self.hparams.train_transformer_layer \
                and var not in self.need_to_initialize_vars:
              self.pretrained_not_train_var_names.append(var)

  def _build_train_graph(self):
    gpu_num = len(self.hparams.gpu_num)
    if gpu_num > 1:print("-" * 10, "Using %d Multi-GPU" % gpu_num, "-" * 10)
    else:print("-" * 10, "Using Single-GPU", "-" * 10)

    input_ids_ph = tf.split(self.input_ids_ph, gpu_num, 0)
    input_mask_ph = tf.split(self.input_mask_ph, gpu_num, 0)
    segment_ids_ph = tf.split(self.segment_ids_ph, gpu_num, 0)

    label_ids_ph = tf.split(self.label_ids_ph, gpu_num, 0)

    tower_grads = []
    tot_losses = []
    tot_logits = []
    tot_labels = []

    tvars = []
    for i, gpu_id in enumerate(self.hparams.gpu_num):
      with tf.device('/gpu:%d' % gpu_id):
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
          print("bert_graph_multi_gpu :", gpu_id)


          input_phs = (input_ids_ph[i], input_mask_ph[i], segment_ids_ph[i])
          label_ids_phs = label_ids_ph[i]
          created_model = self.hparams.graph.Model(self.hparams, self.bert_config, input_phs, label_ids_phs,
                                                   self.train_setup_vars, self.label_dict, self.dropout_prob_ph)
          logits, loss_op = created_model.build_graph()

          tot_losses.append(loss_op)
          tot_logits.append(logits)
          tot_labels.append(label_ids_ph[i])

          if i == 0:
            self._select_train_variables()
            if self.hparams.do_adam_weight_optimizer:
              self.optimizer, self.global_step = optimization.create_optimizer(
                loss_op, self.hparams.learning_rate, self.num_train_steps, self.num_warmup_steps, use_tpu=False)
            else:
              self.optimizer = tf.train.AdamOptimizer(self.hparams.learning_rate)
              self.global_step = tf.Variable(0, name="global_step", trainable=False)

          if not self.hparams.do_train_bert:
            if i == 0:
              for var in tf.trainable_variables():
                if var not in self.pretrained_not_train_var_names:
                  tvars.append(var)
          else:
            tvars = tf.trainable_variables()

          if self.hparams.do_adam_weight_optimizer:
            # This is how the model was pre-trained.
            grads = tf.gradients(loss_op, tvars)
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
            tower_grads.append(zip(grads, tvars))
          else:
            grads = self.optimizer.compute_gradients(loss_op, var_list=tvars)
            tower_grads.append(grads)
          tf.get_variable_scope().reuse_variables()

    avg_grads = average_gradients(tower_grads)
    self.loss_op = tf.divide(tf.add_n(tot_losses), gpu_num)
    self.logits = tf.concat(tot_logits, axis=0)
    tot_labels = tf.concat(tot_labels, axis=0)

    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
      self.train_op = self.optimizer.apply_gradients(avg_grads, self.global_step)
      # new_global_step = self.global_step + 1
      # self.train_op = tf.group(self.train_op, [self.global_step.assign(new_global_step)])

    if self.hparams.loss_type == "sigmoid":
      correct_pred = tf.equal(tf.round(tf.nn.sigmoid(self.logits)), tf.cast(self.label_ids_ph, tf.float32))
      self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
      self.confidence = tf.round(tf.nn.sigmoid(self.logits))

    else:
      eval = tf.nn.in_top_k(self.logits, self.label_ids_ph, 1)
      correct_count = tf.reduce_sum(tf.cast(eval, tf.int32))
      self.accuracy = tf.divide(correct_count, tf.shape(self.label_ids_ph)[0])
      self.confidence = tf.nn.softmax(self.logits, axis=-1)

    if not self.train_setup_vars["do_evaluate"] and not self.train_setup_vars["on_training"] \
        and not self.train_setup_vars["is_train_continue"]:
      self._initialize_uninitialized_variables()

  def _initialize_uninitialized_variables(self):
    uninitialized_vars = []
    self._logger.info("Initializing Updated Variables...")
    for var in tf.global_variables():
      if var in self.pretrained_all_var_names and var not in self.need_to_initialize_vars:
        print("Pretrained Initialized Variables / ", var)
        continue
      uninitialized_vars.append(var)
      print("Update Initialization / ", var)
    init_new_vars_op = tf.variables_initializer(uninitialized_vars)
    self.sess.run(init_new_vars_op)

  def _make_feed_dict(self, batch_data, dropout_prob):
    feed_dict = {}
    input_ids, input_mask, segment_ids, label_ids = batch_data

    feed_dict[self.input_ids_ph] = input_ids
    feed_dict[self.input_mask_ph] = input_mask
    feed_dict[self.segment_ids_ph] = segment_ids

    feed_dict[self.label_ids_ph] = label_ids
    feed_dict[self.dropout_prob_ph] = dropout_prob

    return feed_dict

  def train(self, pretrained_file=None):
    config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
    )
    config.gpu_options.allow_growth = True

    if pretrained_file:
      # self.train_setup_vars["on_training"] = True
      self.train_setup_vars["is_train_continue"] = True

    self.sess = tf.Session(config=config)
    self._make_placeholders()
    self._build_train_graph()

    # Tensorboard
    saver = tf.train.Saver(max_to_keep=10)
    if pretrained_file:
      saver.restore(self.sess, pretrained_file)
      self._logger.info("Restoring Session from checkpoint complete!")

    self.tensorboard_summary = TensorBoardSummaryWriter(self.hparams.root_dir, self.sess, self.sess.graph)
    step_loss_mean, step_accuracy_mean = 0, 0
    total_data_len = int(math.ceil(len(self.train_examples) / self.hparams.train_batch_size))
    self._logger.info("Batch iteration per epoch is %d" % total_data_len)

    start_time = datetime.now().strftime('%H:%M:%S')
    self._logger.info("Start train model at %s" % start_time)
    for epoch_completed in range(self.hparams.num_epochs):
      start_time = time.time()
      if epoch_completed > 0 and self.hparams.training_shuffle_num > 0:
        self.train_examples = self.processor.get_train_examples(self.hparams.data_dir)

      for i in range(total_data_len):
        batch_data = self.processor.get_bert_batch_data(i, self.hparams.train_batch_size, "train")

        accuracy_val, loss_val, global_step_val, _ = self.sess.run(
          [self.accuracy,
           self.loss_op,
           self.global_step,
           self.train_op],
          feed_dict=self._make_feed_dict(batch_data, self.hparams.dropout_prob)
        )
        step_loss_mean += loss_val
        step_accuracy_mean += accuracy_val

        if global_step_val % self.hparams.tensorboard_step == 0:
          step_loss_mean /= self.hparams.tensorboard_step
          step_accuracy_mean /= self.hparams.tensorboard_step

          self.tensorboard_summary.add_summary("train/cross_entropy", step_loss_mean, global_step_val)
          self.tensorboard_summary.add_summary("train/accuracy", step_accuracy_mean, global_step_val)

          self._logger.info("[Step %d][%d th] loss: %.4f, accuracy: %.2f%%  (%.2f seconds)" % (
            global_step_val,
            i + 1,
            step_loss_mean,
            step_accuracy_mean * 100,
            time.time() - start_time))

          step_loss_mean, step_accuracy_mean = 0, 0
          start_time = time.time()

        if global_step_val % self.hparams.evaluate_step == 0:
          saver.save(self.sess, os.path.join(self.hparams.root_dir, "model.ckpt"), global_step=global_step_val)
          micro_f1 = self._run_evaluate("test")
          self.tensorboard_summary.add_summary("test/f1-micro", micro_f1, global_step_val)

        if global_step_val % self.hparams.save_step == 0:
          self._logger.info("Saving Model...[Step %d]" % global_step_val)
          self.model_save(saver, global_step_val)

      self._logger.info("End of epoch %d." % (epoch_completed + 1))
    self.tensorboard_summary.close()

    if self.sess is not None:
      self.sess.close()

  def model_save(self, saver, global_step_val):
    save_path = saver.save(self.sess, os.path.join(self.hparams.root_dir, "model.ckpt"), global_step=global_step_val)
    self._logger.info("Model saved at : %s" % save_path)

  def _run_evaluate(self, data_type="test"):
    total_data_len = int(math.ceil(len(self.test_examples) / self.hparams.eval_batch_size)) - 1
    self._logger.info("Evaluation batch iteration per epoch is %d" % total_data_len)

    if self.train_setup_vars["do_evaluate"]:
      print("Evaluation batch iteration per epoch is %d" % total_data_len)

    total_ground_truth = []
    total_pred = []
    for i in tqdm(range(total_data_len), mininterval=1):
      batch_data = self.processor.get_bert_batch_data(i, self.hparams.eval_batch_size, data_type)

      if self.hparams.loss_type == "sigmoid":
        logits_val, confidence_val = self.sess.run([self.logits, self.confidence],
                                                   feed_dict=self._make_feed_dict(batch_data, 0.0))
      else:
        confidence_val, = self.sess.run([self.confidence], feed_dict=self._make_feed_dict(batch_data, 0.0))
        pred_index = np.argmax(confidence_val, axis=-1)

      ground_truth = batch_data[3]
      total_ground_truth.extend(ground_truth)
      total_pred.extend(pred_index)

    assert len(total_ground_truth) == len(total_pred)

    results = get_f1_score(total_ground_truth, total_pred)
    print("f1-macro : ", results['f1-macro'])
    # print("f1-micro : ", results['f1-micro'])
    print("classification report", results['classification_report'])

    self._logger.info(results['classification_report'])

    return results['f1-macro']
