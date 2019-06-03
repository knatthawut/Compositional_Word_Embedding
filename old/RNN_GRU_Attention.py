'''
    Implementation the Recurrent Neural Network model with Gated Recurrent Unit (GRU) using CuDNNGRU with Self-Attention Mechanism
'''
#Import Libraries
import tensorflow as tf
from keras.layers import Embedding, CuDNNGRU
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
from keras_self_attention import SeqSelfAttention
from Keras_baseline import KERAS_baseline 
# ***************
# Constant Declaration
# ***************




# ***************
# Model Definitions
# ***************

# Baseline: the Recurrent Neural Network model with Gated Recurrent Unit (GRU) with Self-Attention Mechanism
class RNN_GRU_Attention_baseline(KERAS_baseline):
    '''
    Class for the Recurrent Neural Network model with Gated Recurrent Unit (GRU) using CuDNNGRU with Self-Attention Mechanism
    CuDNNGRU: no activation function
    '''
    def __init__(self, attention_activation,  type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH,type_of_loss_func = 'mean_squared_error', type_of_optimizer = 'adam'):
        # Init model attributes
        super().__init__('RNN_GRU_Attention', type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH, type_of_loss_func=type_of_loss_func, type_of_optimizer=type_of_optimizer)
        self.attention_activation = attention_activation

        # Model Definition of Simple RNN network
        self.model =  Sequential() # Define Sequential Model
        embedding_layer = Embedding(self.vocab_size,
                                self.embedding_dim,
                                embeddings_initializer=Constant(self.embedding_matrix),
                                input_length=self.MAX_SEQUENCE_LENGTH,
                                trainable=False)
        self.model.add(embedding_layer) # Add the Embedding layers to the model
        self.model.add(CuDNNGRU(self.embedding_dim, return_sequences=False))
        self.model.add(SeqSelfAttention(attention_activation=self.attention_activation))
        # Print Model Summary to see the architecture of model
        print(self.model.summary())
        # Compile the model to use
        self.model.compile(loss= self.type_of_loss_func
                ,optimizer= self.type_of_optimizer
                ,metrics=['acc'])
