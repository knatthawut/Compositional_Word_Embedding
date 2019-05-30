'''
Implementation for Simple Bidirectional RNN
'''
#Import Libraries
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Embedding, Bidirectional
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

# Baseline: Simple Bidirectional RNN network without attention
class Simple_Bidirectional_RNN_baseline(KERAS_baseline):
    '''
    Class for Bidirectional RNN baseline
    '''
    def __init__(self, type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH,type_of_loss_func = 'mean_squared_error', type_of_optimizer = 'adam', activation_func = 'relu'):
        super().__init__('Simple_Bidirectional_RNN', type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH, type_of_loss_func=type_of_loss_func, type_of_optimizer=type_of_optimizer)
        self.activation_func = activation_func
        # Model Definition of Simple RNN network
        self.model =  Sequential() # Define Sequential Model
        embedding_layer = Embedding(self.vocab_size,
                                self.embedding_dim,
                                embeddings_initializer=Constant(self.embedding_matrix),
                                input_length=self.MAX_SEQUENCE_LENGTH,
                                trainable=False)
        self.model.add(embedding_layer) # Add the Embedding layers to the model
        self.model.add(Bidirectional(SimpleRNN(self.embedding_dim, activation=self.activation_func, return_sequences=False))) # Add Bidirectional RNN
        # Print Model Summary to see the architecture of model
        print(self.model.summary())
        # Compile the model to use
        self.model.compile(loss= self.type_of_loss_func
                ,optimizer= self.type_of_optimizer
                ,metrics=['acc'])