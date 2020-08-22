'''
Main file to run the experiment 1:
Compare between 2 models: Direction and Location Accuracy
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
from sklearn.model_selection import StratifiedKFold, KFold
import argparse
# Import modules
import utils
import evaluation
# Import Baselines
from SimpleRNN import Simple_RNN_baseline
from Average_baseline import AVG_baseline
from BiRNN_GRU_Attention import Bidirectional_RNN_GRU_Attention_baseline
from BiRNN_GRU import Bidirectional_RNN_GRU_baseline
from BiRNN_LSTM_Attention import Bidirectional_RNN_LSTM_Attention_baseline
from BiRNN_LSTM import Bidirectional_RNN_LSTM_baseline
from Conv1D import Conv1D_baseline
from RNN_GRU_Attention import RNN_GRU_Attention_baseline
from RNN_GRU import RNN_GRU_baseline
from RNN_LSTM_Attention import RNN_LSTM_Attention_baseline
from RNN_LSTM import RNN_LSTM_baseline
from BiSimpleRNN import Simple_Bidirectional_RNN_baseline
from BiSimpleRNN_withoutDense import Simple_Bidirectional_RNN_without_Dense_baseline
from RNN_GRU_Attention_Multi import RNN_GRU_Attention_Multi_baseline

# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras

# ***************
# Constant Declaration
# ***************



# Parse the arguments
parser = argparse.ArgumentParser(description='Run Exp1 for each baseline')
parser.add_argument('--main_baseline',type=str, metavar='', required=True, help='Name of the main baseline')
parser.add_argument('--compare_baseline',type=str, metavar='', required=True, help='Name of the compare baseline')
parser.add_argument('--vector_file',type=str, metavar='', required=True, help='Path to the vector file')
parser.add_argument('--word2vec_type',type=str, metavar='', required=True, help='Type of word2vec model (CBOW, SG, FastText)')
parser.add_argument('--train_file',type=str, metavar='', required=True, help='Path to the train file')
parser.add_argument('--num_of_epochs',type=int, metavar='', required=True, help='Number of epochs to train the model')
parser.add_argument('--dim',type=int, metavar='', required=True, help='Number of dimension of the model')
parser.add_argument('--max_length',type=int, metavar='', required=True, help='Max sequence length of the input')
parser.add_argument('--batch_size',type=int, metavar='', required=True, help='Batch size to train the model')
args = parser.parse_args()

# Files Paths
type_of_Word2Vec_model = args.word2vec_type
vector_file_name = 'wiki-db_more50_200'
# vector_file_name_path = './../model/' + type_of_Word2Vec_model + '/' + vector_file_name
vector_file_name_path = args.vector_file
# train_file_name = 'uni_pair_combine'
train_file_path = args.train_file

save_model_path = './../model/'
x_file = save_model_path + 'Evaluation/' + type_of_Word2Vec_model + '_X_feature.npy'
y_file = save_model_path + 'Evaluation/' + type_of_Word2Vec_model + '_Y_label.npy'

# Integer Constant
MAX_SEQUENCE_LENGTH = args.max_length
num_of_epochs = args.num_of_epochs
batch_size = args.batch_size
#validation_split = 0.01

# Hyperparameters Setup
embedding_dim = args.dim
num_hidden = 128



def getBaseline(baseline_name,embedding_matrix):
    if baseline_name == 'AVG':
        return AVG_baseline(type_of_Word2Vec_model)
    if baseline_name == 'SimpleRNN':
        return Simple_RNN_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix)
    if baseline_name == 'BiRNN':
            return Simple_Bidirectional_RNN_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix)
    if baseline_name == 'BiRNN_withoutDense':
        return Simple_Bidirectional_RNN_without_Dense_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix)
    if baseline_name == 'GRU':
        return RNN_GRU_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix)
    if baseline_name == 'BiGRU':
        return Bidirectional_RNN_GRU_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix)
    if baseline_name == 'LSTM':
        return RNN_LSTM_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix)
    if baseline_name == 'BiLSTM':
        return Bidirectional_RNN_LSTM_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix)
    if baseline_name == 'GRU_Attention':
        return RNN_GRU_Attention_baseline('tanh',type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix)
    if baseline_name == 'GRU_Attention_Multi':
        return RNN_GRU_Attention_Multi_baseline('tanh',type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix)
    if baseline_name == 'BiGRU_Attention':
        return Bidirectional_RNN_GRU_Attention_baseline('tanh',type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix)
    if baseline_name == 'LSTM_Attention':
        return RNN_LSTM_Attention_baseline('tanh',type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix)
    if baseline_name == 'BiLSTM_Attention':
       return Bidirectional_RNN_LSTM_Attention_baseline('tanh',type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix)
    if baseline_name == 'Conv1D':
        return Conv1D_baseline(32,7,type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix)


def train_evaluate_compare(wordvec,main_baseline, comparison_baseline, x_train_cv, y_train_cv , x_test_cv, y_test_cv):
    '''
    Function to train two baselines: main_baseline and comparison_baseline and evaluation two baselines in Cross-validation scenario for Experiment 1
    Input:
            main_baseline: the main baseline that need to be compare with comparison_baseline
            comparison_baseline: the baseline to compare with main_baseline
            x_train_cv: feature matrix (X) for training, shape(90% number_of_data, MAX_SEQUENCE_LENGTH) of word_idx
            y_train_cv: label matrix (Y) for training, shape(90% number_of_data, embedding_dim) word vector of compount word
            x_test_cv: x_train_cv: feature matrix (X) for testing, shape(10% number_of_data, MAX_SEQUENCE_LENGTH) of word_idx
            y_test_cv: label matrix (Y) for testing, shape(10% number_of_data, embedding_dim) word vector of compount word

    Output:
            DIR_acc: Direction Accuracy of main_baseline comparing to comparison_baseline
            LOC_acc: Location Accuracy of main_baseline comparing to comparison_baseline
    '''
    ## Training Phase
    # Train the main_baseline
    main_baseline.train(x_train_cv,y_train_cv,num_of_epochs,batch_size)
    # Train the comparison_baseline
    comparison_baseline.train(x_train_cv,y_train_cv,num_of_epochs,batch_size)

    ## Inference Phase
    # Predict result of the main_baseline
    main_baseline_y_predict = main_baseline.predict(x_test_cv,wordvec)

    # Predict result of the comparison_baseline
    comparison_baseline_y_predict = comparison_baseline.predict(x_test_cv,wordvec)

    ## Testing
    DIR_acc = evaluation.calculateAccuracy('DIR', y_test_cv, main_baseline_y_predict,comparison_baseline_y_predict) # Get Direction Accuracy of main_baseline comparing to comparison_baseline
    LOC_acc = evaluation.calculateAccuracy('LOC', y_test_cv, main_baseline_y_predict,comparison_baseline_y_predict) # Get Location Accuracy of main_baseline comparing to comparison_baseline

    # print('DIR: ',DIR_acc)
    # print('LOC: ',LOC_acc)
    return DIR_acc, LOC_acc

if __name__ == '__main__':
    # Main function

    # Load the Pretrained Word Vector from Gensim
    wordvec = Word2Vec.load(vector_file_name_path) # Load the model from the vector_file_name
    wordvec.wv.init_sims(replace=True)
    print('Loaded Word2Vec model')
    # Get Vocabulary Size
    vocab_size = len(wordvec.wv.vocab)
    print('Vocab size: ', vocab_size)

    # Prepare Train_data
    fname = train_file_path
    X , Y = utils.load_data_from_text_file(fname,wordvec,MAX_SEQUENCE_LENGTH) # Preprocess the input data for the model
    # X, Y = utils.load_data_from_numpy(x_file, y_file)            # Load input data from numpy file

    # Convert Word2Vec Gensim Model to Embedding Matrix to input into RNN
    embedding_matrix = utils.Word2VecTOEmbeddingMatrix(wordvec,embedding_dim)

    # Do Cross Validation
    kFold = KFold(n_splits = 10)
    #Init the Accuracy dictionary = {}
    accuracy = {}
    accuracy['DIR'] = np.zeros(10)
    accuracy['LOC'] = np.zeros(10)
    idx = 0 # Index of accuracy
    main_baseline = getBaseline(args.main_baseline,embedding_matrix)
    comparison_baseline = getBaseline(args.compare_baseline,embedding_matrix)
    for train_idx, test_idx in kFold.split(X,Y):
        # Define train and test data

        x_train_cv = X[train_idx]
        x_test_cv  = X[test_idx]

        y_train_cv = Y[train_idx]
        y_test_cv  = Y[test_idx]

        # Compare two baseline
        # Define two baseline
        # main_baseline = Conv1D_baseline(32,7,type_of_Word2Vec_model,vocab_size,embedding_dim, embedding_matrix,MAX_SEQUENCE_LENGTH)
        # main_baseline = Bidirectional_RNN_LSTM_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH)



        accuracy['DIR'][idx],accuracy['LOC'][idx] = train_evaluate_compare(wordvec,main_baseline, comparison_baseline , x_train_cv, y_train_cv , x_test_cv, y_test_cv)
        print('========= Fold {} ============='.format(idx))
        print('{} vs {}'.format(main_baseline.baseline_name,comparison_baseline.baseline_name))
        print('DIR accuracy: {}'.format(accuracy['DIR'][idx]))
        print('LOC: {}'.format(accuracy['LOC'][idx]))
        idx += 1
        # break
    print('================ Final {}  vs  {} ==============='.format(main_baseline.baseline_name,comparison_baseline.baseline_name))
    print('DIR accuracy: {}'.format(np.mean(accuracy['DIR'])))
    print('LOC: {}'.format(np.mean(accuracy['LOC'])))
