'''
    Implementation the Recurrent Neural Network model with Gated Recurrent Unit (GRU) using CuDNNGRU
'''
#Import Libraries
import tensorflow as tf
from tensorflow.keras.layers import Embedding, CuDNNGRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
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
    def __init__(self, type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH,type_of_loss_func = 'mean_squared_error', type_of_optimizer = 'adam'):
        super().__init__('RNN_GRU', type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH, type_of_loss_func=type_of_loss_func, type_of_optimizer=type_of_optimizer)

        # Model Definition of Simple RNN network
        self.model =  Sequential() # Define Sequential Model
        embedding_layer = Embedding(self.vocab_size,
                                self.embedding_dim,
                                embeddings_initializer=Constant(self.embedding_matrix),
                                input_length=self.MAX_SEQUENCE_LENGTH,
                                trainable=False)
        self.model.add(embedding_layer) # Add the Embedding layers to 
        self.model.add(CuDNNGRU(self.embedding_dim, return_sequences=False))
        # Print Model Summary to see the architecture of model
        print(self.model.summary())
        # Compile the model to use
        self.model.compile(loss= self.type_of_loss_func
                ,optimizer= self.type_of_optimizer
                ,metrics=['acc'])