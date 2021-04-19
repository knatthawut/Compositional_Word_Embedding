'''
Implementation for all Keras model
'''
#Import Libraries
import tensorflow as tf
import keras
from keras.layers import SimpleRNN, Embedding
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
from baseline import Baseline
# ***************
# Constant Declaration
# ***************




# ***************
# Model Definitions
# ***************

# Baseline: Simple RNN network without attention
class KERAS_baseline(Baseline):
    '''
    Class for KERAS baseline
    '''
    def __init__(self, baseline_name, type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH = 21,type_of_loss_func = 'mean_squared_error', type_of_optimizer = 'adam'):
        super().__init__('KERAS_'+baseline_name,type_of_wordvec)
        # Save model parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.type_of_loss_func = type_of_loss_func
        self.type_of_optimizer = type_of_optimizer
        self.model = None
        # self.print_information()

    def print_information(self):
        super().print_information()
        print('Vocab_size: ',self.vocab_size)
        print('Embedding Dimension: ',self.embedding_dim)
        print('Loss function: ',self.type_of_loss_func)
        print('Optimizer function: ',self.type_of_optimizer)

    def train(self, x_train, y_train, num_of_epochs,batch_size,learning_rate=1e-3):
        # Compile the model to use
        optimizer = keras.optimizers.Adam(lr = learning_rate)
        self.model.compile(loss= self.type_of_loss_func
                ,optimizer= optimizer)
        self.history = self.model.fit(x_train , y_train, epochs=num_of_epochs , batch_size=batch_size)
        print('Training Done!')

    def save_model(self, save_path):
        '''

        '''
        self.save_path = save_path
        name ='{}_vocab_size_{}_embedding_dim_{}_loss_function_{}_optimizer_{}'.format(self.baseline_name,self.vocab_size,self.embedding_dim,self.type_of_loss_func,self.type_of_optimizer)
        fname = os.path.join(self.save_path, name)
        self.model.save(fname)
        print('Saved model to: ', fname)

    def predict(self,x_test,wordvec):
        return self.model.predict(x_test)
