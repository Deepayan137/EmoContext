import os
from config import *
import pdb
import json
import pickle
from tqdm import *
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
from .embeddings import *
from .emoji_handler import *

class TweetData(Dataset):
	def __init__(self, data_dir, split, lmap, **kwargs):
		super().__init__()
		self.data_dir = data_dir
		self.split = split
		self.raw_data_path = os.path.join(data_dir, split+'.txt')
		self.data_file = split+'.pkl'
		self.vocab_file = 'vocab.pkl'
		self.lmap = lmap
		if not os.path.exists(os.path.join(self.data_dir, self.data_file)):
			print('creating for %s'%split)
			self.glove_model = self.load_gensim()
			self._create_data()
			

		else:
			print('data loading for %s'%split)
			self._load_data()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		idx = str(idx)
		if self.split == 'test':
			return {
				'input': self.data[idx]['input'],
				'feature': self.data[idx]['target']
				}
		return {
			'input': self.data[idx]['input'],
			'target': self.data[idx]['target'],
			'feature': self.data[idx]['feature']
			}

	def load_gensim(self):
		if os.path.isfile('data/gensim_glove_vectors.txt'):
			print('loading...')
			glove_model = KeyedVectors.load_word2vec_format("data/gensim_glove_vectors.txt", 
															binary=False)
		else:
			print('converting to word2vec format')
			glove2word2vec(glove_input_file="data/glove.twitter.27B.50d.txt", 
						word2vec_output_file="data/gensim_glove_vectors.txt")
			print('loading...')
			glove_model = KeyedVectors.load_word2vec_format("data/gensim_glove_vectors.txt", binary=False)
		return glove_model

	def _load_vocab(self):
		with open(os.path.join(self.data_dir, self.vocab_file), 'rb') as vocab_file:
			# vocab = json.load(vocab_file)
			vocab = pickle.load(vocab_file)

		self.vocab = vocab

	def _load_data(self, vocab=True):
		
		with open(os.path.join(self.data_dir, self.data_file), 'rb') as file:
			# self.data = json.load(file)
			self.data = pickle.load(file)
		if vocab:
			self._load_vocab()

	def text2feature(self, text):

		return get_embeds(self.glove_model, text)

	def _create_data(self):

		if self.split == 'train':
			self._create_vocab()
		else:
			self._load_vocab()

		tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)

		data = defaultdict(dict)

		with open(self.raw_data_path, 'r') as file:
			lines = file.readlines()
			for i, line in enumerate(tqdm(lines[1:])):
				units = line.split('\t')
				index = int(units[0])
				text = tokenizer.tokenize(demojify_v3(' '.join(units[1:4])))
				data[index]['input'] = text
				if self.split == 'train':
					data[index]['target'] = self.lmap[units[4].strip()]
				data[index]['feature'] = self.text2feature(text)


		with open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
			# data = json.dump(data, data_file, ensure_ascii=False)
			pickle.dump(data, data_file)
		self._load_data(vocab=False)

	def _create_vocab(self):

		assert self.split == 'train', "Vocablurary can only be created for training file."

		tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)

		vocab = defaultdict(int)

		def update_vocab(vocab, text):
			words = tokenizer.tokenize(demojify_v3(text))
			for word in words:
				vocab[word] += 1

		with open(self.raw_data_path, 'r') as file:
			lines = file.readlines()
			for i, line in enumerate(lines[1:]):
				units = line.split('\t')
				text = ' '.join(units[1:4])
				update_vocab(vocab, text)
		
		with open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
			# data = json.dump(vocab, vocab_file, ensure_ascii=False)
			pickle.dump(vocab, vocab_file)
		self._load_vocab()

# datasets = {}
# datasets['train'] = TweetData(
# data_dir='data',
# split='train'
# )
