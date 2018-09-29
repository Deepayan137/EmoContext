from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

import re
import string
import numpy as np
import os.path

# Download twitter embeddings from http://nlp.stanford.edu/data/glove.twitter.27B.zip

# If glove embeds is not in word2vec form then first convert it then load it
if os.path.isfile('pretrained_embeds/gensim_glove_vectors.txt'):
    glove_model = KeyedVectors.load_word2vec_format("pretrained_embeds/gensim_glove_vectors.txt", binary=False)
else:
    glove2word2vec(glove_input_file="pretrained_embeds/glove.twitter.27B.50d.txt", word2vec_output_file="pretrained_embeds/gensim_glove_vectors.txt")
    glove_model = KeyedVectors.load_word2vec_format("pretrained_embeds/gensim_glove_vectors.txt", binary=False)

translator = str.maketrans('', '', string.punctuation)

def get_embeds(word_list):
    
    embed_mat = []
    
    for word in word_list:
        # Case folding
        word = word.lower()
        # Not include word containing numbers
        if not bool(re.search(r'\d', word)):
            # Remove punctuations
            word = word.translate(translator)
            # if word is not empty string
            if word:
                try:
                    embed_mat.append(glove_model.get_vector(word))
                except:
                    embed_mat.append(glove_model.get_vector('unk'))
    return np.asarray(embed_mat)



