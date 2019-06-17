'''
Implementation for Experiment the Automatic Noun Compound Interpretion using Deep Neural Networks

'''

from gensim.models import Word2Vec
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
import pandas as pd
# ***************
# Constant Declaration
# ***************



# Files Paths
type_of_Word2Vec_model = 'CBOW'
vector_file_name = 'wiki-db_more50_200'
vector_file_name_path = './../model/' + type_of_Word2Vec_model + '/' + vector_file_name
train_file_name = 'uni_pair_combine'
train_file_path = './../dataset/train_data/'


# Integer Constant
num_of_epoch = 100
batch_size = 100


# Hyperparameters Setup


def getClassifierModel(num_of_classes=37,embedding_dim=300,activation_func='tanh',drop_out_rate=0.1):
    model = Sequential()
    # model.add(Dropout(drop_out_rate))
    model.add(Dense(num_of_classes,input_dim = embedding_dim, activation=activation_func))
    model.add(Activation('softmax'))
    
    #Compile model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    # return model
    return model

def readData(data_file,target_dict):
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
    X_word = []
    y = []

    # read the train_data_file
    df = pd.read_csv(data_file,sep='\t',encoding='utf-8')
    df.columns = ['word_1','word_2','label']

    # extract information
    for index, row in df.iterrows():
        line = row['word_1']+'_'+row['word_2']
        X_word.append(line)
        label = target_dict[row['label']]
        y.append(label)

    return X_word, y

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

    with open(class_file,'r',encoding='utf-8') as fin:
        for i , line in enumerate(fin):
            res[line.strip()] = i 

    return res

def loadWordVecModel(type_of_Word2Vec_model,embedding_dim):
    res = {}
    return res

def wordTovec(X_word,type_of_Word2Vec_model,embedding_dim):
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

    # Load the Word Vec model in Dictionary
    word_vector_dict = loadWordVecModel(type_of_Word2Vec_model,embedding_dim)
    for compound_word in X_word:
        compound_word_id = 'COMPOUND_ID/' + compound_word
        if compound_word_id in word_vector_dict:
            X.append(word_vector_dict[compound_word_id])
        else:
            print('Compound word not in dictionary')
    
    return X
        

def main():
    # Main fucntion
    # Load the pretrained Word Vector from Gensim
    wordvec = Word2Vec.load(vector_file_name_path) # Load the model from the vector_file_name
    wordvec.wv.init_sims(replace=True)
    print('Loaded Word2Vec model')
    # Get Vocabulary Size
    vocab_size = len(wordvec.wv.vocab)
    print('Vocab size: ', vocab_size)

    # Load Tratz data
    target_dict = readClassLabel(class_file)
    X_train_word, y_train = readData(train_data_file,target_dict)
    X_test_word, y_test = readData(test_data_file,target_dict)

    # Use the baseline to convert the word into vector representation
    X_train = wordTovec(X_train_word,type_of_Word2Vec_model)
    X_test = wordTovec(X_test_word,type_of_Word2Vec_model)


    # Get Model
    model = getClassifierModel()

    #Train Model
    model.fit(X_train,y_train,epochs=num_of_epoch , batch_size=batch_size)
    
    # Predict
    round_predictions = model.predict_classes(X_test)

    # Evaluate
    target_names = target_dict.keys()
    report = classification_report(y_test,round_predictions,target_names=target_names)
    print(report)


if __name__ == '__main__':
    main()