import tensorflow as tf
import utils
from sklearn import preprocessing
from gensim.models import Word2Vec
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import numpy as np
# from pycm import *
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
import argparse
# Import baseline
from Actual_baseline import Actual_baseline
from SimpleRNN import Simple_RNN_baseline
from Average_baseline import AVG_baseline
# from BiRNN_GRU_Attention import Bidirectional_RNN_GRU_Attention_baseline
from BiRNN_GRU import Bidirectional_RNN_GRU_baseline
# from BiRNN_LSTM_Attention import Bidirectional_RNN_LSTM_Attention_baseline
from BiRNN_LSTM import Bidirectional_RNN_LSTM_baseline
from Conv1D import Conv1D_baseline
from RNN_GRU_Attention import RNN_GRU_Attention_baseline
from RNN_GRU import RNN_GRU_baseline
# from RNN_LSTM_Attention import RNN_LSTM_Attention_baseline
from RNN_LSTM import RNN_LSTM_baseline
from BiSimpleRNN import Simple_Bidirectional_RNN_baseline
from SimpleRNN import Simple_RNN_baseline
from BiSimpleRNN_withoutDense import Simple_Bidirectional_RNN_without_Dense_baseline
# from RNN_GRU_Attention_Multi import RNN_GRU_Attention_Multi_baseline
from Concate_baseline import Concatenate_baseline

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
# ***************
# Constant Declaration
# ***************

# Files Paths
type_of_Word2Vec_model = 'CBOW'
data_name = 'encow14_wiki_'
# vector_file_name = type_of_Word2Vec_model + '_size300_window10_min8'
vector_file_path = ''
baseline_train_file_name = 'train_data'
baseline_train_file_path = './../dataset/train_data/' + baseline_train_file_name
Tratz_data_path = '../dataset/Tratz_data/tratz2011_fine_grained_random/'
class_file = Tratz_data_path + 'classes.txt'
train_data_file = Tratz_data_path + 'train.tsv'
test_data_file = Tratz_data_path + 'test.tsv'
# Word2Vec_SG_file_name_path = vector_file_name_path
# Word2Vec_CBOW_file_name_path = vector_file_name_path
# Word2Vec_Pretrained_file_name_path = './../model/' + 'encow-sample-compounds.bin'
result_path = '../results/'
# Integer Constant
num_of_epoch = 5000
num_of_epoch_composition = 2500
batch_size = 128
batch_size_composition = 1024*16
embedding_dim = 200
num_classes = 37
MAX_SEQUENCE_LENGTH=21

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def loadWordVecModel(vector_file_path,embedding_dim):
    res = None
    vocab_size = 0
    # is Word2Vec model
    # if type_of_Word2Vec_model == 'SG':
    #Word2Vec_file_name_path =  Word2Vec_SG_file_name_path
    res = Word2Vec.load(vector_file_path) # Load the model from the vector_file_name
    #elif type_of_Word2Vec_model == 'CBOW':
    #    Word2Vec_file_name_path = Word2Vec_CBOW_file_name_path
    #    res = Word2Vec.load(Word2Vec_file_name_path) # Load the model from the vector_file_name
    #elif type_of_Word2Vec_model == 'PRETRAINED':
        # Load the Pretrained Word2Vec from bin file
    #    res = KeyedVectors.load_word2vec_format(Word2Vec_Pretrained_file_name_path,binary=True)
    res.wv.init_sims(replace=True)
    vocab_size = len(res.wv.vocab)
    print('Vocab_size: ',vocab_size)
    # is GloVe Model


    # is FastText model

    return res, vocab_size

def readClassLabel(class_file):
    '''
    Function:
        read the label in class_file and return a dictionary
    Input:
        class_file(str): input file consists of the label list
    Output:
        res(dict):
        Key: label
        value: class_id(int)
    '''
    res = {}
    reverse_res = {}

    with open(class_file,'r',encoding='utf-8') as fin:
        for i , line in enumerate(fin):
            res[line.strip()] = i
            reverse_res[str(i)] = line.strip()

    return res, reverse_res

def readData(data_file,target_dict,word_vector):
    '''
    Function:
        read the data_file and get the data
    Input:
        data_file(str): data input file name
        target_dict: dictionary of the label dict[label] = class_id(int)
    Output:
        X_word: compound_word
        y: list of class_id
    '''
    # Init return value
    X_word_idx = []
    y = []
    X_word = []

    # read the train_data_file
    df = pd.read_csv(data_file,sep='\t',encoding='utf-8')
    df.columns = ['word_1','word_2','label']

    # extract information
    for index, row in df.iterrows():
        line = []
        compound = 'COMPOUND_ID/' + row['word_1'] + '_' + row['word_2']
        X_word.append(compound)
        line.append(utils.get_word_index(row['word_1'],word_vector))
        line.append(utils.get_word_index(row['word_2'],word_vector))
        X_word_idx.append(line)
        label = target_dict[row['label']]
        y.append(label)

    y_one_hot = [y]

    y_one_hot = indices_to_one_hot(y, num_classes)
    X_word_idx = np.array(X_word_idx)
    return X_word, X_word_idx, y,y_one_hot


def main():
    vector_file_path = '../model/encow14_wiki_CBOW_size200_window5_min50'
    word_vector, vocab_size = loadWordVecModel(vector_file_path,embedding_dim)

    target_dict, reverse_target_dict = readClassLabel(class_file)

    X_train_word,X_train_word_idx, y_train_label , y_train = readData(train_data_file,target_dict,word_vector)

    inVocab = np.zeros(37)
    outVocab = np.zeros(37)
    for i,label in enumerate(y_train_label):
        if X_train_word[i] in word_vector.wv.vocab:
            inVocab[label] = inVocab[label] + 1
        else:
            outVocab[label] = outVocab[label] + 1
    
    for i in range(37):
        print('Class: ',i)
        print('InVocab: ',inVocab[i])
        print('outVocab: ',outVocab[i])
    print('Total: ')
    print('InVocab: ',inVocab.sum())
    print('outVocab: ',outVocab.sum())

if __name__ == '__main__':
    main()
