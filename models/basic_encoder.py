import tensorflow as tf
from utils import stack_bidirectional_rnn

class BasicEncoder(object):

    def __init__(self, hparams, dropout_keep_prob_ph):

        self.hparams = hparams
        self.dropout_keep_prob = dropout_keep_prob_ph

    def recurrent_encoder(self, embedded, embedded_len, name, rnn_hidden_dim=None):
        if rnn_hidden_dim is None:
            rnn_hidden_dim = self.hparams.rnn_hidden_dim
        with tf.variable_scope("rnn-encoder-%s" % name, reuse=tf.AUTO_REUSE):
            rnn_outputs = stack_bidirectional_rnn(
                cell="CUDNNGRU",
                num_layers=self.hparams.rnn_depth,
                num_units=rnn_hidden_dim * 2,
                inputs=embedded,
                sequence_length=embedded_len,
                state_merge="concat",
                output_dropout_keep_prob=self.dropout_keep_prob,
                residual=self.hparams.rnn_depth > 1
            )

        return rnn_outputs

    def cnn_encoder(self, embedded, embedded_len):
        # transformer sequence output : [batch, 280/40, 768]
        embedded_feature_map = tf.expand_dims(embedded, axis=-1)
        # Convolution & Maxpool
        features = []
        for size in self.hparams.cnn_filter_size:
            with tf.variable_scope("CNN_filter_%d" % size, reuse=tf.AUTO_REUSE):
                # Add padding to mark the beginning and end of words.
                pad_height = size - 1
                pad_shape = [[0, 0], [pad_height, pad_height], [0, 0], [0, 0]]
                embedded_feature_map = tf.pad(embedded_feature_map, pad_shape)
                feature = tf.layers.conv2d(
                    inputs=embedded_feature_map,
                    filters=self.hparams.num_filters,
                    kernel_size=[size, 768],
                    use_bias=False
                )
                # max_pooling : shape = [batch, time, 1, out_channels]
                feature = tf.reduce_max(feature, axis=1)
                feature = tf.squeeze(feature)
                feature = tf.reshape(feature, [tf.shape(embedded)[0], self.hparams.num_filters])
                # shape = [batch, out_channels]
                features.append(feature)

        # shape = [batch, out_channels * len(self.hparams.filter_size)] // e.g.) [?,120(30*4)]
        cnn_layer_output = tf.concat(features, axis=1)

        return cnn_layer_output