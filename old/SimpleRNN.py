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

# ***************
# Constant Declaration
# ***************




# ***************
# Model Definitions
# ***************

# Baseline: Simple RNN network without attention



def get_rnn_model(vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH ):
    # Model Definition of Simple RNN network
    model =  Sequential() # Define Sequential Model
    embedding_layer = Embedding(vocab_size,
                            embedding_dim,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
    model.add(embedding_layer) # Add the Embedding layers to 
    model.add(SimpleRNN(embedding_dim, return_sequences = False))
    # Print Model Summary to see the architecture of model
    print(model.summary())
    # Compile the model to use
    model.compile(loss='mean_squared_error'
              ,optimizer='rmsprop'
              ,metrics=['acc'])
    return model