'''
Utilities functions
'''
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
# Function Implementation
# ***************

def load_data_from_text_file(input_file_name,wordvec,MAX_SEQUENCE_LENGTH):
    '''
    Create training data for the network from the input_file_name
    Input:  input_file_name: text file for the data
            wordvec: Gensim Word2Vec model
            MAX_SEQUENCE_LENGTH: max length of the compound word 
    Output: x_train: numpy array of feature with shape(number_of_compound_word_in_data , MAX_SEQUENCE_LENGTH) 
            each row consists of word_index of the compound word 's elements
            y_train: the vector representation of the compound word.
    '''
    #Read data
    fin = open(input_file_name,'r', encoding = 'utf-8').read().split('\n')

    # Initiate the return values
    y_train = []
    x_train = []

    # Load data
    with open(input_file_name,'r', encoding = 'utf-8') as fin:
        for line in fin:
            tmp = line.split('\t')
            y_string = tmp[0]
            x_string = tmp[1].lower().strip('\n').split(' ')
            if len(x_string) < 2:
                continue
            if y_string in wordvec.wv:
                y_train.append(wordvec.wv[y_string])
            else:
                continue
                # y_train.append(wordvec.wv['UNKNOWN'])
            # change Text into Integer
            x_train_line = []
            for sample in x_string:
                if sample in wordvec.wv:
                    x_train_line.append(wordvec.wv.vocab[sample].index)
                else:
                    x_train_line.append(wordvec.wv.vocab['unknown'].index)
            x_train.append(x_train_line)

    # Padding
    x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
    y_train = np.array(y_train)

    return x_train , y_train

def load_data_from_numpy(feature_file, label_file):
    '''
    Load data from numpy array for faster speed
    Input:
            feature_file: file name of numpy file the features (X)
            label_file: name of the numpy file of the label (Y): compound word vector
    '''
    feature = np.load(feature_file)
    label = np.load(label_file)

    return feature , label

def save_data_to_numpy(feature, feature_file, label, label_file):
    '''
    Save data to numpy array for faster speed
    Input:
            feature: feature numpy array (X)
            feature_file: file name of numpy file the features (X)
            label: label numpy array (Y)
            label_file: name of the numpy file of the label (Y): compound word vector
    '''
    np.save(feature,feature) # Save feature array into feature file
    np.save(label,label) # Save label array into label file

def Word2VecTOEmbeddingMatrix(wordvec, embedding_dim):
    '''
    Convert Gensim Word2Vec model into Embedding Matrix to fit into Keras
    '''
    model = wordvec
    embedding_matrix = np.zeros((len(model.wv.vocab), embedding_dim))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

