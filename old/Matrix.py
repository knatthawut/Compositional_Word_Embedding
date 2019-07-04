#Import Libraries
import tensorflow as tf
from keras.layers import SimpleRNN, Embedding, Dense, Flatten
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

# Baseline: Matrix baseline
class Matrix_baseline(KERAS_baseline):
    '''
    Class for Matrix baseline
    Use Matrix baseline (Socher et al. 2011)
    p = g(W[u;v]+b)
    '''
    def __init__(self, type_of_wordvec, vocab_size, embedding_dim,
                 embedding_matrix, MAX_SEQUENCE_LENGTH=2,type_of_loss_func ='mean_squared_error', type_of_optimizer = 'adam',
                 activation_func='tanh'):
        super().__init__('Matrix', type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH, type_of_loss_func=type_of_loss_func, type_of_optimizer=type_of_optimizer)
        self.activation_func = activation_func
        # self.print_information()
        self.model =  Sequential() # Define Sequential Model
        embedding_layer = Embedding(self.vocab_size,
                                self.embedding_dim,
                                weights=[embedding_matrix],
                                input_length=self.MAX_SEQUENCE_LENGTH,
                                trainable=False)
        self.model.add(embedding_layer) # Add the Embedding layers to the model
        self.model.add(Flatten())
        self.model.add(Dense(self.embedding_dim,activation=activation_func))
        # Print Model Summary to see the architecture of model
        print(self.model.summary())


    def print_information(self):
        super().print_information()
        # print('Attention Activation: ',self.attention_activation)
        print('Activation Function: ',self.activation_func)
        print(self.model.summary())
