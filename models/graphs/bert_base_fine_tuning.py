from bert_model import modeling_base

import tensorflow as tf

class Model(object):
	def __init__(self, hparams, bert_config=None, input_phs=None, label_ids_phs=None,
	             train_setup_vars=None, label_dict=None, dropout_prob_ph=0.1):

		self.hparams = hparams
		self.dropout_prob_ph = dropout_prob_ph
		self.bert_config = bert_config
		self.input_phs = input_phs
		self.label_id_phs = label_ids_phs
		self.train_setup_vars = train_setup_vars
		self._label_unbalanced_ratio(label_dict)

	def _label_unbalanced_ratio(self, label_dict):
		total_instances = 0
		label_list = []
		for i in label_dict.keys():
			label_list.append(label_dict[i])
			total_instances += label_dict[i]
		modified_label_list = []
		print(label_list)
		for num_label in label_list:
			modified_label_list.append(total_instances / num_label)
		self.label_weights = tf.constant(modified_label_list)
		print(self.label_weights)

	def build_graph(self):
		if not self.train_setup_vars["do_evaluate"]:
			is_training = True
		else:
			is_training = False
		use_one_hot_embeddings = False
		input_ids, input_mask, segment_ids = self.input_phs

		# load bert model
		bert_model = modeling_base.BertModel(
			config=self.bert_config,
			dropout_prob=self.dropout_prob_ph,
			input_ids=input_ids,
			input_mask=input_mask,
			token_type_ids=segment_ids,
			use_one_hot_embeddings=use_one_hot_embeddings,
			scope='bert',
			hparams=self.hparams
		)
		pooled_output = bert_model.get_pooled_output()

		return self._final_output_layer(pooled_output, self.label_id_phs)

	def _final_output_layer(self, final_input_layer, label_ids):
		output_layer = final_input_layer

		if self.hparams.loss_type == "sigmoid":
			logits_units = 1
		else:
			logits_units = 4

		if not self.train_setup_vars["do_evaluate"]:
			print("during training : Dropout!")
			output_layer = tf.nn.dropout(final_input_layer, keep_prob=0.9)
		else:
			print("during evaluation : No Dropout!")

		logits = tf.layers.dense(
			inputs=output_layer,
			units=logits_units,
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
			name="logits"
		)

		print("logits", logits)
		if self.hparams.loss_type == "sigmoid":
			logits = tf.squeeze(logits, axis=-1)
			loss_op = tf.nn.sigmoid_cross_entropy_with_logits(
				logits=logits, labels=tf.cast(label_ids, tf.float32), name="binary_cross_entropy")
		else:
			pred_arg_max = tf.argmax(logits, axis=-1) # logits : [batch, 4] -> [batch]
			print(pred_arg_max)
			label_one_hot = tf.one_hot(label_ids, depth=4)
			label_weights = tf.reduce_sum(self.label_weights * label_one_hot, axis=1)
			unweighted_losses_op = tf.nn.softmax_cross_entropy_with_logits_v2(
				logits=logits, labels=label_one_hot, name="cross_entropy")
			loss_op = unweighted_losses_op * label_weights

		loss_op = tf.reduce_mean(loss_op, name="cross_entropy_mean")


		return logits, loss_op