'''
    Implementation the Transformer model
'''
#Import Libraries
import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, Bidirectional, Dense
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
from keras_transformer import get_encoder_component
# ***************
# Constant Declaration
# ***************




# ***************
# Model Definitions
# ***************

# Baseline: the Transformer
class Transformer_baseline(KERAS_baseline):
    '''
    Transformer baseline
    '''
    def __init__(self, attention_activation,  type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH = 21,type_of_loss_func = 'mean_squared_error', type_of_optimizer = 'adam',activation_func = 'tanh',head_num=1,hidden_dim=100,dropout_rate=0.0):
        # Init model attributes
        super().__init__('Transformer', type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH, type_of_loss_func=type_of_loss_func, type_of_optimizer=type_of_optimizer)
        # self.print_information()
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate


        self.model =  Sequential() # Define Sequential Model
        embedding_layer = Embedding(self.vocab_size,
                                self.embedding_dim,
                                weights=[self.embedding_matrix],
                                mask_zero = True,
                                input_length=self.MAX_SEQUENCE_LENGTH,
                                trainable=False)
        transformer_encoder = get_encoder_component(
            name='Encoder',
            input_layer=embedding_layer,
            head_num=self.head_num,
            hidden_dim=self.hidden_dim,
            # attention_activation=attention_activation,
            # feed_forward_activation=feed_forward_activation,
            dropout_rate=self.dropout_rate,
            trainable=True,
        )
        self.model.add(transformer_encoder) # Add the Embedding layers to the model
#        self.model.add(Dense(embedding_dim,activation=self.activation_func))                        # Add Dense layer with embedding_dim hidden units to return the vector.

        # Print Model Summary to see the architecture of model
        print(self.model.summary())
        # Compile the model to use


    def print_information(self):
        super().print_information()
        print(self.model.summary())
