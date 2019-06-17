import os
import numpy as np
from csv import DictReader
import joblib
from nltk import word_tokenize

class DataSet():
	def __init__(self, name="train"):

		print("Reading dataset")
		bodies = name + "_bodies.csv"
		stances = name + "_stances.csv"

		stances = self._read_data(stances)
		articles = self._read_data(bodies)

		self.stances, self.articles = self._filtered_news(stances, articles)

		print("Total stances: " + str(len(self.stances)))
		print("Total bodies: " + str(len(self.articles)))

	def _read_data(self, filename):
		rows = []
		with open(filename, "r", encoding='utf-8') as table:
			r = DictReader(table)

			for line in r:
				rows.append(line)
		return rows

	def _filtered_news(self, stances, articles):
		articles_dict = dict()
		# make the body ID an integer value
		for s in stances:
			s['Body ID'] = int(s['Body ID'])

		# copy all bodies into a dictionary
		for article in articles:
			articles_dict[int(article['Body ID'])] = article['articleBody']

		return stances, articles_dict

	def change_labels(self, stances):
		labels = ["agree", "disagree", "discuss"]
		for s in stances:
			if s["Stance"] in labels:
				s["Stance"] = 'related'

		return stances

class ModelDataset(object):
	def make_model_pickle(self ,dataset:DataSet):
		return NotImplementedError

class BertDataset(ModelDataset):
	def make_model_pickle(self, dataset, path=None):
		print("BERT")
		headlines = dataset.stances
		articles = dataset.articles

		labels = ['unrelated', "agree", "disagree", "discuss"]

		total_data = []
		for h in headlines:
			headline = h['Headline']
			headline = do_lower_case(headline)
			body = articles[h['Body ID']]
			body = [text for text in body.strip().split('\n') if len(text.strip()) > 0]

			body_context = ""
			for text in body:
				if text == body[-1]:
					body_context += text
					break
				body_context += text + ' eop '
			body_context.strip()
			body_context = do_lower_case(body_context)

			label = str(labels.index(h['Stance']))
			total_data.append([headline, body_context, label])
		print("pickle_data", len(total_data))
		with open(path, "wb") as fwb_handle:
			joblib.dump(total_data, fwb_handle)

class AHDEDataset(ModelDataset):
	def make_model_pickle(self, dataset, path=None):
		headlines = dataset.stances
		articles = dataset.articles

		labels = ['unrelated', 'related']

		total_data = []
		for h in headlines:
			headline = h['Headline']
			headline = do_lower_case(headline)
			body = articles[h['Body ID']]
			body = [text for text in body.strip().split('\n') if len(text.strip()) > 0]

			for b_idx, paragraph in enumerate(body):
				body[b_idx] = do_lower_case(paragraph)

			label = str(labels.index(h['Stance']))
			total_data.append([headline, body, label])
		print("pickle_data", len(total_data))
		with open(path, "wb") as fwb_handle:
			joblib.dump(total_data, fwb_handle)

		return total_data

	def make_glove_vocab(self, total_data):
		vocab = set()
		for headline, body, _ in total_data:
			#headline
			vocab.update(word_tokenize(headline))
			for paragraph in body:
				vocab.update(word_tokenize(paragraph))

		print("total vocab size is %d" % len(vocab))

class GLoVEProcessor(object):
	def __init__(self, train_data, test_data):
		# self.vocab = list(self._get_vocabs(train_data, test_data))
		self.vocab_path = "fnc-1/fnc-1_vocab.txt"
		self.glove_path = "/mnt/raid5/shared/word_embeddings/glove.6B.300d.txt"
		trimmed_path = "fnc-1/fnc-1_glove_300_dim_trimmed.npz"

		# self.write_vocab_file(self.vocab, self.vocab_path)
		total_vocab, word2id = self.load_vocab(self.vocab_path)
		glove_vocab = self.load_glove_vocab(self.glove_path)
		print(total_vocab[0])
		print(total_vocab[-1])
		exit()
		vocab_intersection = set(total_vocab) & glove_vocab
		print("%s vocab is in Glove" % len(vocab_intersection))

		self.export_trimmed_glove_vectors(word2id, self.glove_path, trimmed_path, 300)

	def _get_vocabs(self, train_data, test_data):
		data_type = [(train_data, "train"), (test_data, "test")]
		vocab = set()
		for t in data_type:
			for headline, body, _ in t[0]:
				# headline
				vocab.update(word_tokenize(headline))
				for paragraph in body:
					vocab.update(word_tokenize(paragraph))

			print("%s vocab size is %d" % (t[1], len(vocab)))

		return vocab

	def write_vocab_file(self, vocab, path):
		with open(path, "w", encoding="utf-8") as f_handle:
			for i, word in enumerate(vocab):
				if i == len(vocab) - 1:
					f_handle.write(word)
				else:
					f_handle.write("%s\n" % word)

		print("Write Vocabulary is done. %d tokens" % len(vocab))

	def load_vocab(self, path):
		with open(path) as f_handle:
			vocab = f_handle.read().splitlines()

		word2id = dict()
		vocab.insert(0, "[PAD]")
		vocab.append("[UNK]")

		for idx, word in enumerate(vocab):
			word2id[word] = idx

		return [vocab, word2id]

	def load_glove_vocab(self, path):
		"""
		Args:
				filename: path to the glove vectors hparams.glove_path
		"""
		print("Building glove vocab...")
		glove_vocab = set()
		with open(path, "r", encoding='utf-8') as f_handle:
			for line in f_handle:
				word = line.strip().split(' ')[0]
				glove_vocab.add(word)

		print("Getting Glove Vocabulary is done. %d tokens" % len(glove_vocab))

		return glove_vocab

	def export_trimmed_glove_vectors(self, word2id, glove_path, trimmed_path, dim):
		"""
		Saves glove vectors in numpy array
		Args:
				vocab: dictionary vocab[word] = index
				glove_filename: a path to a glove file
				trimmed_filename: a path where to store a matrix in npy
				dim: (int) dimension of embeddings
		"""
		embeddings = np.random.uniform(low=-1, high=1, size=(len(word2id), dim))
		print(embeddings.shape)

		with open(glove_path, encoding='utf-8') as f:
			for line in f:
				line = line.strip().split(' ')
				word = line[0]
				embedding = [float(x) for x in line[1:]]

				if len(embedding) < 2:
					continue

				if word in word2id:
					word_idx = word2id[word]
					embeddings[word_idx] = np.asarray(embedding)

		np.savez_compressed(trimmed_path, embeddings=embeddings)

def do_lower_case(inputs):
	split_inputs = inputs.strip().split(" ")
	for idx, sentence in enumerate(split_inputs):
		split_inputs[idx] = sentence.lower()
	lower_outputs = " ".join(split_inputs)

	return lower_outputs

def make_bert_file():
	train_dataset = DataSet("train")
	test_dataset = DataSet("competition_test")

	bert_data = BertDataset()
	bert_data.make_model_pickle(train_dataset, "./bert.train")
	bert_data.make_model_pickle(test_dataset, "./bert.test")


if __name__ == '__main__':
	make_bert_file()