'''
    Implementation the Recurrent Neural Network model with Long Short-term Memory (LSTM) using CuDNNLSTM with Self-Attention Mechanism
'''
#Import Libraries
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM
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
from keras_self_attention import SeqSelfAttention
from Keras_baseline import KERAS_baseline
from Attention_layer import Attention_layer
# ***************
# Constant Declaration
# ***************




# ***************
# Model Definitions
# ***************

# Baseline: the Recurrent Neural Network model with Long Short-term Memory (LSTM) with Self-Attention Mechanism
class RNN_LSTM_Attention_baseline(KERAS_baseline):
    '''
    Class for the Recurrent Neural Network model with Long Short-term Memory (LSTM) using CuDNNLSTM with Self-Attention Mechanism
    CuDNNLSTM: no activation function
    '''
    def __init__(self, attention_activation,  type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH=21,type_of_loss_func = 'mean_squared_error', type_of_optimizer = 'adam'):
        # Init model attributes
        super().__init__('RNN_LSTM_Attention', type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH, type_of_loss_func=type_of_loss_func, type_of_optimizer=type_of_optimizer)
        self.attention_activation = attention_activation
        # self.print_information()
        # Model Definition of Simple RNN network
        self.model =  Sequential() # Define Sequential Model
        embedding_layer = Embedding(self.vocab_size,
                                self.embedding_dim,
                                weights=[self.embedding_matrix],
                                input_length=self.MAX_SEQUENCE_LENGTH,
                                trainable=False)
        self.model.add(embedding_layer) # Add the Embedding layers to the model
        self.model.add(LSTM(self.embedding_dim, return_sequences=True))
        self.model.add(Attention_layer(self.MAX_SEQUENCE_LENGTH))
        # Print Model Summary to see the architecture of model
        print(self.model.summary())


    def print_information(self):
        super().print_information()
        print('Attention Activation: ',self.attention_activation)
        # print('Activation Function: ',self.activation_func)
        print(self.model.summary())
