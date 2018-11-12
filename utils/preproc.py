import os
import pdb
import numpy as np
import re, string
# from nltk.corpus import stopwords
from tokenizer.tokenizer import TweetTokenizer
from .emoji_handler import *

class Preprocess:
	def __init__(self,  glove_model, dim):
		'''takes tokenized text'''
		self.model = glove_model
		self.tok = self.build_tokenizer()
		self.dim = dim
		

	def __call__(self, x):
		string_ = self.rem_punct(x)
		words = self.tok(string_)
		words = self.to_lower(words)
		# words = self.rem_sw(words)
		return self.embeds(words)
		
	def build_tokenizer(self):
	    T = TweetTokenizer(preserve_handles=False, 
	                        preserve_hashes=False, 
	                        preserve_case=False, 
	                        preserve_url=False,
	                        regularize=True)
	    def _inner(text):
	        return T.tokenize(text)
	    return _inner

	def to_lower(self, words):
		return list(map(lambda x:x.lower(), words))

	def rem_sw(self, words):
		stopWords = set(stopwords.words('english'))
		return list(filter(lambda x:x not in stopWords, words))

	def rem_punct(self, x):
		x = demojify_v4(x)
		regex = re.compile('[%s]' % re.escape(string.punctuation))
		return regex.sub('', x)

	def pad_seq(self, x):
		max_length = 30
		length = len(x)
		difference = max_length - length
		if length < max_length:
			vector_size = x.shape[1]
			padding = np.zeros((difference, vector_size), dtype=np.float32)
			return np.concatenate((x, padding),axis=0)
		else:
			return x[:max_length,:]
			
		

	def embeds(self, words):
		embed_mat = []
		if not len(words):
			words = ['pad']
		for word in words:
			if word in self.model.vocab:
				embed_mat.append(self.model.get_vector(word))
				
			else:
				embed_mat.append(np.zeros(self.dim, dtype=np.float32))
		return self.pad_seq(np.asarray(embed_mat))

	
# from gensim.models.keyedvectors import KeyedVectors
# glove_model = KeyedVectors.load_word2vec_format('../data/gensim_glove_25d_vectors.txt', binary=False)
# pr = Preprocess(glove_model)
# embeds = pr('Yo ssup!!!')

# # embeds = pr.embeds(glove_model)
# print(embeds.shape)