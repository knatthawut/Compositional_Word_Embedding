'''
Implementation for Experiment the Google Phrase Analogy dataset
https://github.com/nicholas-leonard/word2vec

'''
import tensorflow as tf
import utils
from sklearn import preprocessing
from gensim.models import Word2Vec
from keras.layers import SimpleRNN, Embedding, Dense, Activation, Dropout
from keras.models import Sequential
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import numpy as np
# from pycm import *
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
import argparse
from tensorflow.keras.callbacks import TensorBoard
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
import evaluation

# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras
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
word_analogy_data_path = '../dataset/analogy_test/'
test_data_file = word_analogy_data_path + 'questions-phrases-non_category.txt'
# Word2Vec_SG_file_name_path = vector_file_name_path
# Word2Vec_CBOW_file_name_path = vector_file_name_path
# Word2Vec_Pretrained_file_name_path = './../model/' + 'encow-sample-compounds.bin'
result_path = '../results/'

# Integer Constant
num_of_epoch = 1
num_of_epoch_composition = 1
batch_size = 128
batch_size_composition = 1024*16
embedding_dim = 200
num_classes = 37
MAX_SEQUENCE_LENGTH=27
# Hyperparameters Setup

# Parse the arguments
parser = argparse.ArgumentParser(description='Run Word Phrase Analogy Question Experiment for each baseline')
parser.add_argument('--baseline',type=str, metavar='', required=True, help='Name of the baseline')
parser.add_argument('--type_of_Word2Vec_model', type=str, metavar='', required=True, help='Type of Word2Vec model: CBOW or SG')
parser.add_argument('--vector_file_path', type=str, metavar='', required=True, help='Path to Vector file')
# parser.add_argument('--num_of_epoch', type=int, metavar='', required=True, help='Number of Epoch to train the classifier')
parser.add_argument('--num_of_epoch_composition', type=int, metavar='', required=True, help='Number of Epoch to train the compositional model')
# parser.add_argument('--batch_size', type=int, metavar='', required=True, help='Batch size of the classifier')
parser.add_argument('--batch_size_composition', type=int, metavar='', required=True, help='Batch size of the compositional model')
# parser.add_argument('--activation_func', type=str, metavar='', required=True, help='Activation function of the classifer')
parser.add_argument('--lr', type=float, metavar='', required=True, help='Learning rate of the compositional model')
# parser.add_argument('--tensorboard_path', type=str, metavar='', required=True, help='path to TersorBoard logs')
# parser.add_argument('--momentum', type=float, metavar='', required=True, help='Momentum of the compositional model')
# parser.add_argument('--nesterov', type=bool, metavar='', required=True, help='Nesterov of the compositional model')

args = parser.parse_args()

# num_of_epoch = args.num_of_epoch
num_of_epoch_composition = args.num_of_epoch_composition
# batch_size = args.batch_size
batch_size_composition = args.batch_size_composition
# activation_func = args.activation_func
composition_lr = args.lr
# tensorboard_path = args.tensorboard_path
# composition_decay = args.decay
# composition_momentum = args.momentum 
# composition_nesterov = args.nesterov

# if (run_mode == 'dev'):
#    test_data_file = Tratz_data_path + 'val.tsv'

# Define TensorBoard
# tensorboard= TensorBoard(log_dir=tensorboard_path)

def getBaseline(baseline_name,embedding_matrix,vocab_size):
        # Init Baseline
    # baseline    =   Actual_baseline(type_of_Word2Vec_model)
    # baseline    = AVG_baseline(type_of_Word2Vec_model)
    # baseline    = Concatenate_baseline(type_of_Word2Vec_model)
    if baseline_name == 'AVG':
            return AVG_baseline(type_of_Word2Vec_model)
    if baseline_name == 'Actual':
            return Actual_baseline(type_of_Word2Vec_model)
    if baseline_name == 'Concatenate':
        return Concatenate_baseline(type_of_Word2Vec_model)
    if baseline_name == 'SimpleRNN':
        return Simple_RNN_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
    if baseline_name == 'BiRNN':
            return Simple_Bidirectional_RNN_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix)
    if baseline_name == 'BiRNN_withoutDense':
        return Simple_Bidirectional_RNN_without_Dense_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix)

    if baseline_name == 'GRU':
        return RNN_GRU_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
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

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def getClassifierModel(num_of_classes=37,embedding_dim=300,activation_func='tanh',drop_out_rate=0.1):
    model = Sequential()
    # model.add(Dropout(drop_out_rate))
    model.add(Dense(num_of_classes,input_dim = embedding_dim, activation=activation_func))
    model.add(Activation('softmax'))

    print(model.summary())
    #Compile model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    # return model
    return model

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

def wordTovec(X_word,baseline,word_vec_dict,embedding_dim):
    '''
    Function:
        convert the word into vector by load the Word2Vec model
    Input:
        X_word(list): list of compound word
        each compound_word(list): list of element words
        type_of_Word2Vec_model(str): type of WordVector model (CBOW,SG,GLOVE,FASTTEXT)
    Output:
        X(list): list of compound_word
        each compound_word(list): list of element word_vector
    '''
    # Init return value
    X = []
    # print(X_word)
    X_word = pad_sequences(X_word,maxlen=MAX_SEQUENCE_LENGTH)
    X = baseline.predict(X_word,word_vec_dict)
    X = np.array(X)
    return X

def compound_to_id(compound,word_vec):
    '''
    Function:
        Covert compound to a list of word_id
    Input:
        compound(str): Ex: Albuquerque_Journal
    Output:
        compound_id(list of int): list of word_id [1,2]
    '''
    compound_id = []
    compound = compound.split('_')
    for word in compound:
        compound_id.append(utils.get_word_index(word,word_vec))
    return compound_id


def readData_word_analogy(test_data_file,word_vector):
    '''
    Function:
        Read question_phrases.txt and transfer into word_id
    Input:
        test_data_file(str): name of the input file
        word_vector: word_vector model to convert word into word_id
    Output:
        X_test_word: [man,king,woman,queen] or [Albuquerque,Albuquerque_Journal,Baltimore,Baltimore_Sun]
        X_test_word_id: [[[1],[2,3][]]]
    '''

    X_test_word = []
    X_test_word_idx = []
    with open(test_data_file,'r',encoding='utf-8') as fin:
        for line in fin:
            compounds = line.strip().split(' ')
            X_test_word.append(compounds)
            compounds_id = []
            for compound in compounds:
                compounds_id.append(compound_to_id(compound,word_vector))
            X_test_word_idx.append(compounds_id)

    X_test_word_idx = np.array(X_test_word_idx)
    return X_test_word, X_test_word_idx

def wordTovecAnalogy(X_test_word_idx, baseline, word_vector):
    '''
    Function:
        Convert Word_id into Vector 
    Input:
        X_test_word_idx(list of list of list of id): 
        baseline: baseline object
        word_vector: word2vec model
    Output:
        X_test(list of test(list of 4 compounds(vector)))

    '''
    X_test = []
    for test in X_test_word_idx:
        # print('test',test)
        test = pad_sequences(test,maxlen=MAX_SEQUENCE_LENGTH)
        test_rs = baseline.predict(test,word_vector)
        test_rs = np.array(test_rs)
        X_test.append(test_rs)

    X_test = np.array(X_test)
    return X_test

def predict_analogy(X_test):
    '''
    Function:
        Predict Analogy for Phrases: man:king::woman:queen
        argmin(cosine(king - man + woman)) 
    Input:
        X_test (list of test(list of 4-compounds))
    Output:
        X_predict (list of vector): king - man + woman
        X_label (list of vector) : queen
    '''
    X_predict = []
    X_label = []
    for test in X_test:
        # print(test)
        X_label.append(test[3])
        X_predict.append(test[1]-test[0]+test[2])
    
    return X_predict,X_label

def main():
    # Main fucntion
    # # Load the pretrained Word Vector
    word_vector, vocab_size = loadWordVecModel(args.vector_file_path,embedding_dim)
    print('Loaded Word2Vec model!')
    # Load 
    # target_dict, reverse_target_dict = readClassLabel(class_file)
    # print('Target Dict',target_dict)
    # print('Reverse Target Dict',reverse_target_dict)
    # X_train_word,X_train_word_idx, y_train_label , y_train = readData(train_data_file,target_dict,word_vector)
    # X_dev_word, X_dev_word_idx, y_dev_label , y_dev  = readData(dev_data_file,target_dict,word_vector)
    X_test_word , X_test_word_idx = readData_word_analogy(test_data_file,word_vector)
    print('Read the analogy dataset!')
    # print('Word',X_test_word)
    # print('ID',X_test_word_idx)
    # exit()
    # Init Baseline
    # baseline    =   Actual_baseline(type_of_Word2Vec_model)
    # baseline    = AVG_baseline(type_of_Word2Vec_model)
    # baseline    = Concatenate_baseline(type_of_Word2Vec_model)

    # GRU baseline
    # Load Embedding Matrix for RNN_GRU

    embedding_matrix = utils.Word2VecTOEmbeddingMatrix(word_vector,embedding_dim)

    baseline = getBaseline(args.baseline,embedding_matrix,vocab_size)
    # print(X_test_word)
    X_train_baseline, y_train_baseline = utils.load_data_from_text_file_exclude(baseline_train_file_path,[],word_vector,MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
    # Train Baseline
    baseline.train(X_train_baseline,y_train_baseline,num_of_epoch_composition,batch_size_composition,composition_lr)


    # Use the baseline to convert the word into vector representation
    # X_train = wordTovec(X_train_word_idx,baseline,word_vector,embedding_dim)
    # X_dev = wordTovec(X_dev_word_idx,baseline,word_vector,embedding_dim)
    X_test = wordTovecAnalogy(X_test_word_idx,baseline,word_vector)

    # Get Model
    # model = getClassifierModel(activation_func=activation_func,embedding_dim=200)

    #Train Model
    # model.fit(X_train,y_train,epochs=num_of_epoch , batch_size=batch_size,validation_data=(X_dev,y_dev), callbacks=[tensorboard])

    # Predict
    predict, label = predict_analogy(X_test)
    MRR, HIT1, HIT10 = evaluation.calculateMRR_HIT(word_vector,label,predict)
    print('MRR: {}'.format(MRR))
    print('HIT@1: {}'.format(HIT1))
    print('HIT@10: {}'.format(HIT10))
    print('===============================')

if __name__ == '__main__':
    main()
