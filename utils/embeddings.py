from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

import re
import string
import numpy as np
import os.path

# Download twitter embeddings from http://nlp.stanford.edu/data/glove.twitter.27B.zip

# If glove embeds is not in word2vec form then first convert it then load it


def get_embeds(glove_model, word_list):
    translator = str.maketrans('', '', string.punctuation)

    embed_mat = []
    
    for word in word_list:
        # Case folding
        word = word.lower()
        # Not include word containing numbers
        if not bool(re.search(r'\d', word)):
            # Remove punctuations
            word = word.translate(translator)
            # if word is not empty string
            if word in glove_model.vocab:
                try:
                    embed_mat.append(glove_model.get_vector(word))
                except Exception as e:
                    print(e)
                    
    return np.asarray(embed_mat)



