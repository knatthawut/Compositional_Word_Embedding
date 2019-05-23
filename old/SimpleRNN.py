#Import Libraries
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Embedding
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
import baseline import Baseline
# ***************
# Constant Declaration
# ***************




# ***************
# Model Definitions
# ***************

# Baseline: Simple RNN network without attention
class Simple_RNN_baseline(Baseline):
    '''
    Class for Simple RNN baseline
    '''
    def __init__(self, type_of_wordvec, vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH,type_of_loss_func = 'mean_squared_error', type_of_optimizer = 'adam', activation_func = 'relu'):
        super().__init__('Simple RNN',type_of_wordvec)
        # Save model parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.type_of_loss_func = type_of_loss_func
        self.type_of_optimizer = type_of_optimizer
        # Model Definition of Simple RNN network
        self.model =  Sequential() # Define Sequential Model
        embedding_layer = Embedding(vocab_size,
                                embedding_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
        self.model.add(embedding_layer) # Add the Embedding layers to 
        self.model.add(SimpleRNN(embedding_dim, activation=activation_func, return_sequences=False))
        # Print Model Summary to see the architecture of model
        print(self.model.summary())
        # Compile the model to use
        self.model.compile(loss= type_of_loss_func
                ,optimizer= type_of_optimizer
                ,metrics=['acc'])

    def train(self, x_train, y_train, num_of_epochs,batch_size, validation_split):
        self.history = self.model.fit(x_train , y_train, epochs=num_of_epochs , batch_size=batch_size, validation_split = validation_split)
        print('Training Done!')

    def save_model(self, save_path):
        '''

        '''
        self.save_path = save_path
        name = f'{self.baseline_name}_vocab_size_{self.vocab_size}_embedding_dim_{self.embedding_dim}_loss_function_{self.type_of_loss_func}_optimizer_{self.type_of_optimizer}'
        fname = os.path.join(self.save_path, name)
        self.model.save(fname)
        print('Saved model to: ', fname)
    
    def predict(self,x_test):
        return self.model.predict(x_test)
