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
import torch
class TweetData(Dataset):
	def __init__(self, data_dir, split, lmap, **kwargs):
		super().__init__()
		self.data_dir = data_dir
		self.split = split
		self.vector_size = kwargs['vector_size']
		self.raw_data_path = os.path.join(data_dir, split+'.txt')
		self.data_file = split+ self.vector_size +'.pkl'
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
		# idx = str(idx)

		if self.split == 'test':
			return {
				'input': self.data[idx]['input'],
				'feature': self.data[idx]['feature'],
				'length': self.data[idx]['length']
				}
		return {
			'input': self.data[idx]['input'],
			'target': self.data[idx]['target'],
			'feature': self.data[idx]['feature'],
			'length':  self.data[idx]['length']
			}

	def load_gensim(self):

		if os.path.isfile('data/gensim_glove_%s_vectors.txt'%self.vector_size):
			print('loading...')
			glove_model = KeyedVectors.load_word2vec_format('data/gensim_glove_%s_vectors.txt'%self.vector_size, 
															binary=False)
		else:
			print('converting to word2vec format')
			glove2word2vec(glove_input_file="data/glove.twitter.27B.%s.txt"%self.vector_size, 
						word2vec_output_file="data/gensim_glove_%s_vectors.txt"%self.vector_size)
			print('loading...')
			glove_model = KeyedVectors.load_word2vec_format("data/gensim_glove_%s_vectors.txt"%self.vector_size, binary=False)
		return glove_model

	def _load_vocab(self):
		with open(os.path.join(self.data_dir, self.vocab_file), 'rb') as vocab_file:
			# vocab = json.load(vocab_file)
			vocab = pickle.load(vocab_file)

		self.vocab = vocab

	def seq_max_len(self):
		lengths = [self.data[i]['length'] for i in range(len(self.data))]
		return max(lengths)

	def pad_seq(self):
		# max_length = self.seq_max_len()
		max_length = 156
		for idx in self.data:
			length = self.data[idx]['length']
			difference = max_length - length
			vector_size = self.data[idx]['feature'].shape[1]
			padding = np.zeros((difference, vector_size), dtype=np.float32)
			self.data[idx]['feature'] = np.concatenate((self.data[idx]['feature'], padding),axis=0)
		

	def _load_data(self, vocab=True, padding=True):
		
		with open(os.path.join(self.data_dir, self.data_file), 'rb') as file:
			# self.data = json.load(file)
			self.data = pickle.load(file)
		if padding:
			self.pad_seq()
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
				data[index]['length'] = data[index]['feature'].shape[0]

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


from .preproc import Preprocess

class TweetData_V02(Dataset):
	def __init__(self, path, label_map):
		self.path = path
		self.lmap = label_map
		self.glove_model = KeyedVectors.load_word2vec_format('data/gensim_glove_25d_vectors.txt', 
														binary=False)
		with open(self.path, 'r') as f:
			self.lines = f.readlines()
		self.data = defaultdict(dict)
	def __len__(self):
		return len(self.lines)

	def __getitem__(self, idx):
		if idx == len(self.lines)-1:
			line = self.lines[idx]
		else:
			line = self.lines[idx+1]
		units = line.split('\t')
		# pdb.set_trace()
		input_ = units[1:4]
		if len(units) < 5:
			units.append('')
		# target_ = self.lmap[units[4].strip()]
		# pdb.set_trace()
		target_ = self.lmap.get(units[-1].strip(), 0)
		return {
				'input': self.preprocess(input_),
				'target': target_
		}

	def preprocess(self, x):
		pre = Preprocess(self.glove_model)
		feature = list(map(pre, x))
		return feature

lmap = {'happy': 0,
	    'sad': 1,
	    'angry': 2,
	    'others': 3
          }
# from model.model import Turnip

# turnip = Turnip(25, 256).cuda()

# x = [torch.Tensor(loader[0]['input'][i]).unsqueeze(1).cuda() for i in range(len(loader[0]['input']))]
# # x = torch.Tensor(loader[0]['input']).unsqueeze(1).cuda()
# turnip(x)
# # pdb.set_trace()