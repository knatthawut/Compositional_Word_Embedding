'''
Implementation for Experiment the Automatic Noun Compound Interpretion using Deep Neural Networks

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
from pycm import *
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
import argparse
# Import baseline
from Actual_baseline import Actual_baseline
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
from Concate_baseline import Concatenate_baseline
from Transformer import Transformer_baseline

#from keras.backend.tensorflow_backend import set_session
# from tensorflow.compat.v1.keras.backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras
# ***************
# Constant Declaration
# ***************


# Parse the arguments
parser = argparse.ArgumentParser(description='Run Tratz exp for each baseline')
parser.add_argument('--baseline',type=str, metavar='', required=True, help='Name of the baseline')
parser.add_argument('--vector_file',type=str, metavar='', required=True, help='Path to the vector file')
parser.add_argument('--word2vec_type',type=str, metavar='', required=True, help='Type of word2vec model (CBOW, SG, FastText)')
parser.add_argument('--train_file',type=str, metavar='', required=True, help='Path to the train file of classifier')
parser.add_argument('--test_file',type=str, metavar='', required=True, help='Path to the test file of classifier')
parser.add_argument('--class_file',type=str, metavar='', required=True, help='Path to the test file')
parser.add_argument('--baseline_train_file',type=str, metavar='', required=True, help='Path to the train file for baseline')
# parser.add_argument('--baseline_test_file',type=str, metavar='', required=True, help='Path to the test file for baseline')
parser.add_argument('--num_of_epochs',type=int, metavar='', required=True, help='Number of epochs to train the classifier model')
parser.add_argument('--num_of_epochs_compo',type=int, metavar='', required=True, help='Number of epochs to train the compositional model')
parser.add_argument('--dim',type=int, metavar='', required=True, help='Number of dimension of the model')
parser.add_argument('--max_length',type=int, metavar='', required=True, help='Max sequence length of the input')
parser.add_argument('--batch_size',type=int, metavar='', required=True, help='Batch size to train the classifier model')
parser.add_argument('--batch_size_compo',type=int, metavar='', required=True, help='Batch size to train the compositional model')
parser.add_argument('--num_classes',type=int, metavar='', required=True, help='Number of classes in the dataset ')

args = parser.parse_args()

# Files Paths
type_of_Word2Vec_model = args.word2vec_type
# vector_file_name = type_of_Word2Vec_model + '_size300_window10_min8'
vector_file_name_path = args.vector_file
# baseline_train_file_name = 'compound_word.tsv'
baseline_train_file_path = args.baseline_train_file
# Tratz_data_path = '../dataset/Tratz_data/tratz2011_fine_grained_random/'
class_file = args.class_file
train_data_file = args.train_file
test_data_file = args.test_file
# Word2Vec_SG_file_name_path = vector_file_name_path
# Word2Vec_CBOW_file_name_path = vector_file_name_path
# Word2Vec_Pretrained_file_name_path = './../model/' + 'encow-sample-compounds.bin'
# result_path = '../results/'
# Integer Constant
num_of_epoch = args.num_of_epochs
num_of_epoch_composition = args.num_of_epochs_compo
batch_size = args.batch_size
batch_size_composition = args.batch_size_compo
embedding_dim = args.dim
num_classes = args.num_classes
MAX_SEQUENCE_LENGTH=args.max_length
# Hyperparameters Setup



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
       return Simple_RNN_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH)
    if baseline_name == 'BiRNN':
       return Simple_Bidirectional_RNN_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH)
    if baseline_name == 'BiRNN_withoutDense':
       return Simple_Bidirectional_RNN_without_Dense_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH)

    if baseline_name == 'GRU':
       return RNN_GRU_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH)
    if baseline_name == 'BiGRU':
       return Bidirectional_RNN_GRU_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH)

    if baseline_name == 'LSTM':
       return RNN_LSTM_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH)

    if baseline_name == 'BiLSTM':
       return Bidirectional_RNN_LSTM_baseline(type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH)

    if baseline_name == 'GRU_Attention':
       return RNN_GRU_Attention_baseline('tanh',type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH)

    if baseline_name == 'GRU_Attention_Multi':
       return RNN_GRU_Attention_Multi_baseline('tanh',type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH)

    if baseline_name == 'BiGRU_Attention':
       return Bidirectional_RNN_GRU_Attention_baseline('tanh',type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH)

    if baseline_name == 'LSTM_Attention':
       return RNN_LSTM_Attention_baseline('tanh',type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH)

    if baseline_name == 'BiLSTM_Attention':
       return Bidirectional_RNN_LSTM_Attention_baseline('tanh',type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH)

    if baseline_name == 'Conv1D':
       return Conv1D_baseline(32,7,type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH)

    if baseline_name == 'Transformer':
<<<<<<< Updated upstream
        return Transformer_baseline('tanh',type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH)
=======
       return Transformer_baseline('tanh',type_of_Word2Vec_model,vocab_size,embedding_dim,embedding_matrix,MAX_SEQUENCE_LENGTH)
    return None
>>>>>>> Stashed changes

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def getClassifierModel(num_of_classes=37,embedding_dim=300,activation_func='softmax',drop_out_rate=0.1):
    model = Sequential()
    # model.add(Dropout(drop_out_rate))
    model.add(Dense(num_of_classes,input_dim = embedding_dim, activation=activation_func))
    # model.add(Activation('softmax'))

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

def loadWordVecModel(type_of_Word2Vec_model,embedding_dim):
    res = None
    vocab_size = 0
    # is Word2Vec model
    # if type_of_Word2Vec_model == 'SG':
    #     Word2Vec_file_name_path =  Word2Vec_SG_file_name_path
    #     res = Word2Vec.load(Word2Vec_file_name_path) # Load the model from the vector_file_name
    # elif type_of_Word2Vec_model == 'CBOW':
    #     Word2Vec_file_name_path = Word2Vec_CBOW_file_name_path
    #     res = Word2Vec.load(Word2Vec_file_name_path) # Load the model from the vector_file_name
    # elif type_of_Word2Vec_model == 'PRETRAINED':
    #     # Load the Pretrained Word2Vec from bin file
    #     res = KeyedVectors.load_word2vec_format(Word2Vec_Pretrained_file_name_path,binary=True)
    res = Word2Vec.load(vector_file_name_path)
    res.wv.init_sims(replace=True)
    vocab_size = len(res.wv.vocab)
    print('Vocab_size: ',vocab_size)
    # is GloVe Model


    # is FastText model

    return res, vocab_size

def wordTovec(X_word,type_of_Word2Vec_model,baseline,word_vec_dict,embedding_dim):
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

def main():
    # Main fucntion
    # # Load the pretrained Word Vector
    word_vector, vocab_size = loadWordVecModel(type_of_Word2Vec_model,embedding_dim)

    # Load Tratz data
    target_dict, reverse_target_dict = readClassLabel(class_file)
    X_train_word,X_train_word_idx, y_train_label , y_train = readData(train_data_file,target_dict,word_vector)
    X_test_word, X_test_word_idx, y_test_label , y_test  = readData(test_data_file,target_dict,word_vector)

    model = getClassifierModel(num_of_classes=num_classes,activation_func='relu',embedding_dim=embedding_dim)


    # GRU baseline
    # Load Embedding Matrix for RNN_GRU

    embedding_matrix = utils.Word2VecTOEmbeddingMatrix(word_vector,embedding_dim)

    baseline = getBaseline(args.baseline,embedding_matrix,vocab_size)
    # print(X_test_word)
    X_train_baseline, y_train_baseline = utils.load_data_from_text_file_exclude(baseline_train_file_path,X_test_word,word_vector,MAX_SEQUENCE_LENGTH)
    # Train Baseline
    baseline.train(X_train_baseline,y_train_baseline,num_of_epoch_composition,batch_size_composition)

    # Use the baseline to convert the word into vector representation
    X_train = wordTovec(X_train_word_idx,type_of_Word2Vec_model,baseline,word_vector,embedding_dim)
    X_test = wordTovec(X_test_word_idx,type_of_Word2Vec_model,baseline,word_vector,embedding_dim)


    # Get Model

    #Train Model
    model.fit(X_train,y_train,epochs=num_of_epoch, batch_size=batch_size)

    # Predict
    round_predictions = model.predict_classes(X_test)
    #print('Predict: ',type(round_predictions))
    #print('Predict: ',round_predictions)
    #print('Label: ',type(y_test))
    #print('Label: ',y_test)
    y_predict = np.array(round_predictions)
    # Evaluate
    target_names = target_dict.keys()
    report = classification_report(y_test_label,y_predict,digits=4)
    print(report)

    # cm = ConfusionMatrix(actual_vector=y_test_label, predict_vector=y_predict)
    # print(cm.classes)
    # print(reverse_target_dict)
    # cm.classes = list(reverse_target_dict.keys())
    # label_set = set(y_test_label + y_predict)
    # for key in list(reverse_target_dict):
    #    if key not in label_set:
    #        del reverse_target_dict[key]
    # print(reverse_target_dict[0])
    # cm.relabel(mapping=reverse_target_dict)
    # result_file_name = result_path + 'test_result_wtDB{}_{}_ComEpoch{}_Epoch{}'.format(baseline.baseline_name,baseline.type_of_wordvec,num_of_epoch_composition,num_of_epoch)
    # cm.save_html(result_file_name,color=(255,204,255))
if __name__ == '__main__':
    main()
