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

# Import modules
import utils
import evaluation
# Import Baselines
from SimpleRNN import Simple_RNN_baseline
from Average_baseline import AVG_baseline

# ***************
# Constant Declaration
# ***************

# Files Paths
type_of_Word2Vec_model = 'CBOW'
vector_file_name = 'wiki-db_more50_200'
vector_file_name_path = './../model/' + type_of_Word2Vec_model + '/' + vector_file_name
train_file_name = 'uni_pair_combine_less100'
train_file_path = './../dataset/train_data/'

save_model_path = './../model/'
x_file = save_model_path + 'Evaluation/' + type_of_Word2Vec_model + '_X_feature.npy'
y_file = save_model_path + 'Evaluation/' + type_of_Word2Vec_model + '_Y_label.npy'

# Integer Constant
MAX_SEQUENCE_LENGTH = 21
num_of_epochs = 10
batch_size = 1024 
validation_split = 0.01
# Hyperparameters Setup
embedding_dim = 200
num_hidden = 128

def train_evaluate_compare(main_baseline, comparison_baseline, x_train_cv, y_train_cv , x_test_cv, y_test_cv):
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
    main_baseline.train(x_train_cv,y_train_cv)
    # Train the comparison_baseline
    comparison_baseline.train(x_train_cv,y_train_cv)

    ## Inference Phase
    # Predict result of the main_baseline
    main_baseline_y_predict = main_baseline.predict(x_test_cv)

    # Predict result of the comparison_baseline
    comparison_baseline_y_predict = comparison_baseline.predict(x_test_cv)
    
    ## Testing 
    DIR_acc = evaluation.calculateAccuracy('DIR', y_test_cv, main_baseline_y_predict,comparison_baseline_y_predict) # Get Direction Accuracy of main_baseline comparing to comparison_baseline
    LOC_acc = evaluation.calculateAccuracy('LOC', y_test_cv, main_baseline_y_predict,comparison_baseline_y_predict) # Get Location Accuracy of main_baseline comparing to comparison_baseline
    
    
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
    fname = os.path.join(train_file_path,train_file_name)
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
    for train_idx, test_idx in kFold.split(X,Y):
        # Define train and test data
        
        x_train_cv = X[train_idx]
        x_test_cv  = X[test_idx]
        
        y_train_cv = Y[train_idx]
        y_test_cv  = Y[test_idx]

        # Compare two baseline 
        # Define two baseline
        main_baseline = Simple_RNN_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH) # Init main baseline: SimpleRNN
        comparison_baseline = AVG_baseline(type_of_Word2Vec_model) # Init comparison baseline: Average Baseline
        
        accuracy['DIR'][idx],accuracy['LOC'][idx] = train_evaluate_compare(main_baseline, comparison_baseline , x_train_cv, y_train_cv , x_test_cv, y_test_cv)
