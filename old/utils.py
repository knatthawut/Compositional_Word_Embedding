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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
# ***************
# Function Implementation
# ***************

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
def get_word_index(sample, wordvec):
    if sample in wordvec.wv:
        return wordvec.wv.vocab[sample].index
    else:
        return wordvec.wv.vocab['unknown'].index

def load_data_from_text_file(input_file_name,wordvec,MAX_SEQUENCE_LENGTH=21):
    '''
    Create training data for the network from the input_file_name
    Input:  input_file_name: text file for the data
            wordvec: Gensim Word2Vec model
            MAX_SEQUENCE_LENGTH: max length of the compound word
    Output: x_train: numpy array of feature with shape(number_of_compound_word_in_data , MAX_SEQUENCE_LENGTH)
            each row consists of word_index of the compound word 's elements
            y_train: the vector representation of the compound word.
    '''
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


def load_data_from_text_file_exclude(input_file_name,exclude_data,wordvec,MAX_SEQUENCE_LENGTH=21):
    '''
    Create training data for the network from the input_file_name
    Input:  input_file_name: text file for the data
            wordvec: Gensim Word2Vec model
            MAX_SEQUENCE_LENGTH: max length of the compound word
    Output: x_train: numpy array of feature with shape(number_of_compound_word_in_data , MAX_SEQUENCE_LENGTH)
            each row consists of word_index of the compound word 's elements
            y_train: the vector representation of the compound word.
    '''

    # Initiate the return values
    y_train = []
    x_train = []
    exclude_data = set(exclude_data)
    # Load data
    with open(input_file_name,'r', encoding = 'utf-8') as fin:
        for line in fin:
            tmp = line.split('\t')
            y_string = tmp[0]
            x_string = tmp[1].lower().strip('\n').split(' ')
            if len(x_string) < 2:
                continue
            if y_string in exclude_data:
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


def load_label_data_from_text_file(input_file_name,wordvec,MAX_SEQUENCE_LENGTH=21):
    '''
    Load the compound word label from the input_file_name file
    Input:
            input_file_name: text file for the data
            wordvec: Gensim Word2Vec model
            MAX_SEQUENCE_LENGTH: max length of the compound word

    Output:
            label: list of compound word
    '''

    # Initiate the return values
    label = []

    # Load data
    with open(input_file_name,'r', encoding = 'utf-8') as fin:
        for line in fin:
            tmp = line.strip().split('\t')
            # print(tmp)
            y_string = tmp[0]
            x_string = tmp[1].lower().strip('\n').split(' ')
            if len(x_string) < 2:
                continue
            if y_string in wordvec.wv:
                label.append(y_string)
                # y_train.append(wordvec.wv['UNKNOWN'])


    return label


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

