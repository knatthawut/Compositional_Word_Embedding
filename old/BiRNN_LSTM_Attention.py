'''
    Implementation the Bidirectional Recurrent Neural Network model with Long Short-term Memory (LSTM) using CuDNNLSTM with Self-Attention Mechanism
'''
#Import Libraries
import tensorflow as tf
from tensorflow.keras.layers import Embedding, CuDNNLSTM, Bidirectional, Dense
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
# ***************
# Constant Declaration
# ***************




# ***************
# Model Definitions
# ***************

# Baseline: the Bidirectional Recurrent Neural Network model with Long Short-term Memory (LSTM) with Self-Attention Mechanism
class Bidirectional_RNN_LSTM_Attention_baseline(KERAS_baseline):
    '''
    Class for the Bidirectional Recurrent Neural Network model with Long Short-term Memory (LSTM) using CuDNNLSTM with Self-Attention Mechanism
    CuDNNLSTM: no activation function
    '''
    def __init__(self, attention_activation,  type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH,type_of_loss_func = 'mean_squared_error', type_of_optimizer = 'adam', activation_func = 'tanh'):
        # Init model attributes
        super().__init__('Bidirectional_RNN_LSTM_Attention', type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH, type_of_loss_func=type_of_loss_func, type_of_optimizer=type_of_optimizer)
        self.attention_activation = attention_activation
        self.activation_func = activation_func
        self.print_information()
        # Model Definition of Simple RNN network
        self.model =  Sequential() # Define Sequential Model
        embedding_layer = Embedding(self.vocab_size,
                                self.embedding_dim,
                                embeddings_initializer=Constant(self.embedding_matrix),
                                input_length=self.MAX_SEQUENCE_LENGTH,
                                trainable=False)
        self.model.add(embedding_layer) # Add the Embedding layers to the model
        self.model.add(Bidirectional(CuDNNLSTM(self.embedding_dim, return_sequences=False)))
        self.model.add(SeqSelfAttention(attention_activation=self.attention_activation))
        self.model.add(Dense(embedding_dim,activation=self.activation_func))                        # Add Dense layer with embedding_dim hidden units to return the vector.

        # Print Model Summary to see the architecture of model
        print(self.model.summary())
        # Compile the model to use
        self.model.compile(loss= self.type_of_loss_func
                ,optimizer= self.type_of_optimizer
                ,metrics=['acc'])
                
    def print_information(self):
        print('Attention Activation: ',self.attention_activation)
        print('Activation Function: ',self.activation_func)