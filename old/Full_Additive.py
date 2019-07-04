#Import Libraries
import tensorflow as tf
from keras.layers import *
from keras.models import Sequential, Model
from keras.initializers import Constant
from keras.backend import slice,squeeze
import keras.backend as K
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

# Baseline: Full Additive
class Full_Additive_baseline(KERAS_baseline):
    '''
    Class for Full Additive baseline
    '''
    def __init__(self, type_of_wordvec, vocab_size, embedding_dim,
                 embedding_matrix, MAX_SEQUENCE_LENGTH=2,type_of_loss_func ='mean_squared_error', type_of_optimizer = 'adam',
                 activation_func='tanh'):
        super().__init__('Full_Additive', type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH, type_of_loss_func=type_of_loss_func, type_of_optimizer=type_of_optimizer)
        self.activation_func = activation_func
        # self.print_information()
        # Model Definition of Full Additive network
        input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
        embedding_layer = Embedding(self.vocab_size,
                                self.embedding_dim,
                                weights=[embedding_matrix],
                                input_length=self.MAX_SEQUENCE_LENGTH,
                                trainable=False)(input_layer)
        print(embedding_layer.shape)
        x1 = Lambda( lambda x: slice(x, (0, 0, 0), (1, 1, -1)))(embedding_layer)
        x2 = Lambda( lambda x: slice(x, (0, 1, 0), (1, 1, -1)))(embedding_layer)
        # print(x1.shape)
        x1 = Flatten()(x1)
        x2 = Flatten()(x2)

        #x = K.variable(value=embedding_layer)
        # x1 = K.eval(x[0,:,:])
        # x2 = K.eval(x[1,:,:])
        w1 = Dense(self.embedding_dim,activation=self.activation_func)(x1)
        w2 = Dense(self.embedding_dim,activation=self.activation_func)(x2)

        output = Add()([w1,w2])
        self.model = Model(input_layer,output)
        # Print Model Summary to see the architecture of model
        print(self.model.summary())


    def print_information(self):
        super().print_information()
        # print('Attention Activation: ',self.attention_activation)
        print('Activation Function: ',self.activation_func)
        print(self.model.summary())
