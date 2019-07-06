'''
Main file to run the experiment 2:
Compare MRR and HIT
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
from SimpleRNN import Simple_RNN_baseline
from BiSimpleRNN_withoutDense import Simple_Bidirectional_RNN_without_Dense_baseline
from RNN_GRU_Attention_Multi import RNN_GRU_Attention_Multi_baseline

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.gpu_options.allocator_type ='BFC'
#config.gpu_options.per_process_gpu_memory_fraction = 0.90
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# ***************
# Constant Declaration
# ***************

# Files Paths
type_of_Word2Vec_model = 'CBOW'
vector_file_name = 'wiki-db_more50_200'
vector_file_name_path = './../model/' + type_of_Word2Vec_model + '/' + vector_file_name
train_file_name = 'uni_pair_combine'
train_file_path = './../dataset/train_data/'
test_file_path = './../dataset/test_data/'
test_file_name = '1000_SampleWords.test'

save_model_path = './../model/'
x_file = save_model_path + 'Evaluation/' + type_of_Word2Vec_model + '_X_feature.npy'
y_file = save_model_path + 'Evaluation/' + type_of_Word2Vec_model + '_Y_label.npy'

# Integer Constant
MAX_SEQUENCE_LENGTH = 21
num_of_epochs = 5
batch_size = 1024*16
validation_split = 0.01
# Hyperparameters Setup
embedding_dim = 200
num_hidden = 128

# Parse the arguments
parser = argparse.ArgumentParser(description='Run Exp2 for each baseline')
parser.add_argument('--baseline',type=str, metavar='', required=True, help='Name of the baseline')
args = parser.parse_args()

def train_evaluate(wordvec, main_baseline, x_train_cv, y_train_cv , x_test_cv, y_label_cv):
    '''
    Function to train main_baseline evaluation in Cross-validation scenario for Experiment 2
    Input:
            main_baseline: the main baseline that need to be compare with comparison_baseline
            x_train_cv: feature matrix (X) for training, shape(90% number_of_data, MAX_SEQUENCE_LENGTH) of word_idx
            y_train_cv: label matrix (Y) for training, shape(90% number_of_data, embedding_dim) word vector of compount word
            x_test_cv: x_train_cv: feature matrix (X) for testing, shape(10% number_of_data, MAX_SEQUENCE_LENGTH) of word_idx
            y_test_cv: label matrix (Y) for testing, shape(10% number_of_data, embedding_dim) word vector of compount word

    Output:
            MRR: Mean reciprocal rank of the main_baseline
            HIT_1: HIT@1 of the main_baseline
            HIT_10: HIT@10 of the main_baseline
    '''
    ## Training Phase
    MRR = 0.0
    HIT_1 = 0.0
    HIT_10 = 0.0
    # Train the main_baseline
    main_baseline.train(x_train_cv,y_train_cv,num_of_epochs,batch_size)

    ## Inference Phase
    # Predict result of the main_baseline
    main_baseline_y_predict = main_baseline.predict(x_test_cv,wordvec)


    ## Testing
    MRR, HIT_1, HIT_10 = evaluation.calculateMRR_HIT(wordvec,y_label_cv[:10],main_baseline_y_predict[:10])


    return MRR , HIT_1, HIT_10

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





if __name__ == '__main__':
    # Main function

    # Load the Pretrained Word Vector from Gensim
    wordvec = Word2Vec.load(vector_file_name_path) # Load the model from the vector_file_name
    wordvec.wv.init_sims(replace=True)
    print('Loaded Word2Vec model')
    # Get Vocabulary Size
    vocab_size = len(wordvec.wv.vocab)
    print('Vocab size: ', vocab_size)

    # Prepare Train_data, test data
    fname_train = os.path.join(train_file_path,train_file_name)
    fname_test = os.path.join(test_file_path,test_file_name)
    # label = utils.load_label_data_from_text_file(fname_train,wordvec,MAX_SEQUENCE_LENGTH) # Preprocess the input data for the model
    # X, Y = utils.load_data_from_numpy(x_file, y_file)            # Load input data from numpy file
    # X , Y  = utils.load_data_from_text_file(fname,wordvec,MAX_SEQUENCE_LENGTH)
    # Split Train, Test
    y_label = utils.load_label_data_from_text_file(fname_test,wordvec)
    x_train , y_train = utils.load_data_from_text_file_exclude(fname_train,y_label,wordvec)
    x_test , y_test = utils.load_data_from_text_file(fname_test,wordvec)
    # Convert Word2Vec Gensim Model to Embedding Matrix to input into RNN
    embedding_matrix = utils.Word2VecTOEmbeddingMatrix(wordvec,embedding_dim)

    # Init all baseline
    # Init baseline list
    # baseline_list = []
    # baseline_list.append(AVG_baseline(type_of_Word2Vec_model))
    # baseline_list.append(Simple_RNN_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix))
    # baseline_list.append(Simple_Bidirectional_RNN_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix))
    #baseline_list.append(RNN_GRU_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix))
    #baseline_list.append(RNN_LSTM_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix))
    #baseline_list.append(Bidirectional_RNN_GRU_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix))
    #baseline_list.append(Bidirectional_RNN_LSTM_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix))
    #baseline_list.append(Conv1D_baseline(32,7,type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix))

    main_baseline = getBaseline(args.baseline,embedding_matrix)

    # Do Cross Validation
    #Init the Accuracy dictionary = {}
    accuracy = {}
    accuracy['MRR'] = 0.0
    accuracy['HIT_1'] = 0.0
    accuracy['HIT_10'] = 0.0


    accuracy['MRR'],accuracy['HIT_1'],accuracy['HIT_10'] = train_evaluate(wordvec,main_baseline, x_train, y_train , x_test,y_label)
    main_baseline.print_information()
    print('MRR: {}'.format(accuracy['MRR']))
    print('HIT@1: {}'.format(accuracy['HIT_1']))
    print('HIT@10: {}'.format(accuracy['HIT_10']))
    print('===============================')

