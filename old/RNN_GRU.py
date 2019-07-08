'''
    Implementation the Recurrent Neural Network model with Gated Recurrent Unit (GRU) using CuDNNGRU
'''
#Import Libraries
import tensorflow as tf
from keras.layers import Embedding, CuDNNGRU, GRU
from keras.models import Sequential
from keras.initializers import Constant
from gensim.models import Word2Vec
import functools
import numpy as np
import sys
import os
import pprint
from keras.preprocessing.text import Tokenizer
pp = pprint.PrettyPrinter(indent=4)
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from Keras_baseline import KERAS_baseline
# ***************
# Constant Declaration
# ***************




# ***************
# Model Definitions
# ***************

# Baseline: the Recurrent Neural Network model with Gated Recurrent Unit (GRU)
class RNN_GRU_baseline(KERAS_baseline):
    '''
    Class for the Recurrent Neural Network model with Gated Recurrent Unit (GRU) using CuDNNGRU
    CuDNNGRU: no activation function
    '''
    def __init__(self, type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH=21,type_of_loss_func = 'mean_squared_error', type_of_optimizer = 'adam'):
        super().__init__('RNN_GRU', type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH=21, type_of_loss_func=type_of_loss_func, type_of_optimizer=type_of_optimizer)

        # Model Definition of Simple RNN network
        self.model =  Sequential() # Define Sequential Model
        embedding_layer = Embedding(self.vocab_size,
                                self.embedding_dim,
                                weights=[embedding_matrix],
                                input_length=self.MAX_SEQUENCE_LENGTH,
                                trainable=False)
        # embedding_layer.build(input_shape=(self.MAX_SEQUENCE_LENGTH,))
        # embedding.set_weights(embedding_matrix)
        self.model.add(embedding_layer) # Add the Embedding layers to the model
        self.model.add(GRU(self.embedding_dim, return_sequences=False))
        # Print Model Summary to see the architecture of model
        print(self.model.summary())
