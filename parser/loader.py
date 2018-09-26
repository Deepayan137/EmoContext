import os
from config import *
import pdb
import json
import pickle
from tqdm import *
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer

class TweetData(Dataset):
	def __init__(self, data_dir, split, **kwargs):
		super().__init__()
		self.data_dir = data_dir
		self.split = split
		self.raw_data_path = os.path.join(data_dir, split+'.txt')
		self.data_file = split+'.pkl'
		self.vocab_file = 'vocab.pkl'
		
		if not os.path.exists(os.path.join(self.data_dir, self.data_file)):

		    self._create_data()

		else:
		    self._load_data()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		idx = str(idx)
		return {
			'input': self.data[idx]['input'],
			'target': self.data[idx]['target']
			}

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
				text = ' '.join(units[1:4])
				data[index]['text'] = tokenizer.tokenize(text)
				data[index]['label'] = units[4].strip()

		with open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
			# data = json.dump(data, data_file, ensure_ascii=False)
			pickle.dump(data, data_file)
		self._load_data(vocab=False)

	def _create_vocab(self):

		assert self.split == 'train', "Vocablurary can only be created for training file."

		tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)

		vocab = defaultdict(int)

		def update_vocab(vocab, text):
			words = tokenizer.tokenize(text)
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

datasets = {}
datasets['train'] = TweetData(
data_dir='data',
split='train'
)
pdb.set_trace()