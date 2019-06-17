from bert_model import tokenization

import os
import csv
import random
import tensorflow as tf
import joblib
from collections import OrderedDict

class PaddingInputExample(object):
	"""Fake example so the num input examples is a multiple of the batch size.

	When running eval/predict on the TPU, we need to pad the number of examples
	to be a multiple of the batch size, because the TPU requires a fixed batch
	size. The alternative is to drop the last batch, which is bad because it means
	the entire output data won't be generated.

	We use this class instead of `None` because treating `None` as padding
	battches could cause silent errors.
	"""

class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, label=None):
		"""Constructs a InputExample.

		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
				sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
				Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
				specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label

class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self,
							 input_ids,
							 input_mask,
							 segment_ids,
							 label_id,
							 is_real_example=True):

		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id
		self.is_real_example = is_real_example

class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	# def get_dev_examples(self, data_dir):
	#   """Gets a collection of `InputExample`s for the dev set."""
	#   raise NotImplementedError()

	def get_test_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for prediction."""
		raise NotImplementedError()

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with tf.gfile.Open(input_file, "r") as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				lines.append(line)
			return lines

class FNCProcessor(DataProcessor):
	def __init__(self, hparams, tokenizer):
		self.hparams = hparams
		self.tokenizer = tokenizer

	def get_train_examples(self, data_dir):
		train_data = self._read_pickle(os.path.join(data_dir, "bert.%s" % "train"), shuffle=True)
		train_example, label_dict = self._create_examples(train_data, "train")
		print(label_dict)
		self.train_example = train_example

		return train_example, label_dict

	def get_test_examples(self, data_dir):
		test_data = self._read_pickle(os.path.join(data_dir, "bert.%s" % "test"), shuffle=False)
		test_example, label_dict = self._create_examples(test_data, "test")
		self.test_example = test_example

		return test_data

	def get_labels(self):
		"""See base class."""
		self.label_list = ["0", "1", "2", "3"]
		return self.label_list

	def _read_pickle(self, data_dir, shuffle=False):
		print("[Reading %s]" % data_dir)
		with open(data_dir, "rb") as frb_handle:
			total_data = joblib.load(frb_handle)
			print("total data length : ", len(total_data))

			if shuffle and self.hparams.training_shuffle_num > 1:
				total_data = self._data_shuffling(total_data, self.hparams.training_shuffle_num)

			return total_data

	def _data_shuffling(self, inputs, shuffle_num):
		for i in range(shuffle_num):
			print(i + 1, "th shuffling has finished!")
			random.shuffle(inputs)
		print("Shuffling Process is done! Total dialog context : %d" % len(inputs))

		return inputs

	def _create_examples(self, inputs, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []

		label_dict = None
		if set_type == "train": label_dict = OrderedDict({0:0,1:0,2:0,3:0})
		for (i, input_data) in enumerate(inputs):
			guid = "%s-%d" % (set_type, i + 1)
			text_a = tokenization.convert_to_unicode(input_data[0])
			text_b = tokenization.convert_to_unicode(input_data[1])
			label = tokenization.convert_to_unicode(input_data[2])
			if label_dict is not None:
				try:
					label_dict[int(label)] += 1
				except KeyError:
					label_dict[int(label)] = 1

			examples.append(
				InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
		print("%s data creation is finished! %d" % (set_type, len(examples)))
		print(label_dict)
		return examples, label_dict

	def get_bert_batch_data(self, curr_index, batch_size, set_type="train"):
		input_ids = []
		input_mask = []
		segment_ids = []
		label_ids = []

		examples = {
			"train": self.train_example,
			"test": self.test_example
		}
		example = examples[set_type]

		for index, each_example in enumerate(example[curr_index * batch_size:batch_size * (curr_index + 1)]):
			feature = convert_single_example(curr_index * batch_size + index, each_example,
																			 self.label_list, self.hparams.max_seq_length, self.tokenizer)

			input_ids.append(feature.input_ids)
			input_mask.append(feature.input_mask)
			segment_ids.append(feature.segment_ids)
			label_ids.append(feature.label_id)

		return [input_ids, input_mask, segment_ids, label_ids]

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()

def convert_single_example(ex_index, example, label_list, max_seq_length,
													 tokenizer):
	"""Converts a single `InputExample` into a single `InputFeatures`."""

	if isinstance(example, PaddingInputExample):
		return InputFeatures(
			input_ids=[0] * max_seq_length,
			input_mask=[0] * max_seq_length,
			segment_ids=[0] * max_seq_length,
			label_id=0,
			is_real_example=False)

	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i

	tokens_a = tokenizer.tokenize(example.text_a)
	tokens_b = None
	if example.text_b:
		tokens_b = tokenizer.tokenize(example.text_b)

	if tokens_b:
		# Modifies `tokens_a` and `tokens_b` in place so that the total
		# length is less than the specified length.
		# Account for [CLS], [SEP], [SEP] with "- 3"
		_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
	else:
		# Account for [CLS] and [SEP] with "- 2"
		if len(tokens_a) > max_seq_length - 2:
			tokens_a = tokens_a[0:(max_seq_length - 2)]

	# The convention in BERT is:
	# (a) For sequence pairs:
	#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
	#  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
	# (b) For single sequences:
	#  tokens:   [CLS] the dog is hairy . [SEP]
	#  type_ids: 0     0   0   0  0     0 0
	#
	# Where "type_ids" are used to indicate whether this is the first
	# sequence or the second sequence. The embedding vectors for `type=0` and
	# `type=1` were learned during pre-training and are added to the wordpiece
	# embedding vector (and position vector). This is not *strictly* necessary
	# since the [SEP] token unambiguously separates the sequences, but it makes
	# it easier for the model to learn the concept of sequences.
	#
	# For classification tasks, the first vector (corresponding to [CLS]) is
	# used as the "sentence vector". Note that this only makes sense because
	# the entire model is fine-tuned.
	tokens = []
	segment_ids = []
	tokens.append("[CLS]")
	segment_ids.append(0)
	for token in tokens_a:
		tokens.append(token)
		segment_ids.append(0)
	tokens.append("[SEP]")
	segment_ids.append(0)

	if tokens_b:
		for token in tokens_b:
			tokens.append(token)
			segment_ids.append(1)
		tokens.append("[SEP]")
		segment_ids.append(1)

	input_ids = tokenizer.convert_tokens_to_ids(tokens)

	# The mask has 1 for real tokens and 0 for padding tokens. Only real
	# tokens are attended to.
	input_mask = [1] * len(input_ids)

	# Zero-pad up to the sequence length.
	while len(input_ids) < max_seq_length:
		input_ids.append(0)
		input_mask.append(0)
		segment_ids.append(0)

	assert len(input_ids) == max_seq_length
	assert len(input_mask) == max_seq_length
	assert len(segment_ids) == max_seq_length

	label_id = label_map[example.label]

	# if ex_index < 5:
	# 	tf.logging.info("*** Example ***")
	# 	tf.logging.info("guid: %s" % (example.guid))
	# 	tf.logging.info("tokens: %s" % " ".join(
	# 		[tokenization.printable_text(x) for x in tokens]))
	# 	tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
	# 	tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
	# 	tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
	# 	tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

	feature = InputFeatures(
		input_ids=input_ids,
		input_mask=input_mask,
		segment_ids=segment_ids,
		label_id=label_id,
		is_real_example=True)
	return feature
