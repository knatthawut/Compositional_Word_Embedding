'''
    Implementation the 1D Convolution neural network
'''
#Import Libraries
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Dense
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
class Conv1D_baseline(KERAS_baseline):
    '''
    Class for the 1D Convolution neural network
    '''
    def __init__(self, filters, kernel_size, type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH,type_of_loss_func = 'mean_squared_error', type_of_optimizer = 'adam',activation_func = 'relu'):
        # Init attributes
        super().__init__('Conv1D', type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH, type_of_loss_func=type_of_loss_func, type_of_optimizer=type_of_optimizer)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation_func = activation_func

        # Model Definition of Simple RNN network
        self.model =  Sequential() # Define Sequential Model
        embedding_layer = Embedding(self.vocab_size,
                                self.embedding_dim,
                                embeddings_initializer=Constant(self.embedding_matrix),
                                input_length=self.MAX_SEQUENCE_LENGTH,
                                trainable=False)
        self.model.add(embedding_layer) # Add the Embedding layers to the model
        # CNN part
        self.model.add(Conv1D(self.filters, self.kernel_size,activation=activation_func))
        self.model.add(GlobalMaxPool1D())
        self.model.add(Dense(embedding_dim))                        # Add Dense layer with embedding_dim hidden units to return the vector.
        # Print Model Summary to see the architecture of model
        print(self.model.summary())
        # Compile the model to use
        self.model.compile(loss= self.type_of_loss_func
                ,optimizer= self.type_of_optimizer
                ,metrics=['acc'])