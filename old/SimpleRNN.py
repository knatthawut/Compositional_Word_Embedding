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

# Integer Constant
MAX_SEQUENCE_LENGTH = 21
num_of_epochs = 10
batch_size = 1024 
validation_split = 0.1
# Hyperparameters Setup
embedding_dim = 200
num_hidden = 128

# ***************
# Function Implementation
# ***************
def load_data(input_file_name,wordvec):
    '''
    Create training data for the network.
    Input:
    Output: x_train , y_train
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
            if y_string.lower() in wordvec.wv:
                y_train.append(wordvec.wv[y_string])
            else:
                y_train.append(wordvec.wv['UNKNOWN'])
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

# ***************
# Model Definitions
# ***************

# Baseline: Simple RNN network without attention
def init_rnn_model(vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH ):
    model =  Sequential() # Define Sequential Model
    embedding_layer = Embedding(vocab_size,
                            embedding_dim,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
    model.add(embedding_layer) # Add the Embedding layers to 
    model.add(SimpleRNN(embedding_dim, return_sequences = False))
    print(model.summary())
    model.compile(loss='mean_squared_error'
              ,optimizer='rmsprop'
              ,metrics=['acc'])
    return model

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
    x_train , y_train = load_data(fname,wordvec) # Preprocess the input data for the model

    # Convert Word2Vec Gensim Model to Embedding Matrix to input into RNN
    embedding_matrix = Word2VecTOEmbeddingMatrix(wordvec,embedding_dim)

    # training the model
    model = init_rnn_model(vocab_size, embedding_dim, embedding_matrix, MAX_SEQUENCE_LENGTH) # Get model architecture
    history = model.fit(x_train , y_train, epochs = num_of_epochs , batch_size = batch_size, validation_split = validation_split)
    print('Training Done!')

    # Save the model
    model_name = 'CBOW_SimpleRNN'
    fname = os.path.join(save_model_path,model_name)
    model.save_weights(fname)
    print('Saved model to: ', fname)