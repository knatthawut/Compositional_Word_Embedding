import functools
import scipy.spatial
from gensim.models import Word2Vec
import numpy as np
import sys

#Editing
n_input = int(sys.argv[1]) # Word Vector dimension

n_classes = n_input # classfication type (y/n)
max_length = 21

filename = 'uni_pair_combine_less10_'

type_model = sys.argv[3]
vectorFile = './'+type_model+'/wiki-db_more50_'+str(n_input)

wordvec = Word2Vec.load(vectorFile)
wordvec.init_sims(replace=True)
model_item = wordvec.vocab.items()
print('\n---> Loading W2V from function files <---')

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

def avgSample(sample,model):
    model = wordvec
    vect = []
    count = 0.0
    for word in sample:
        if len(vect) == 0 and word in model:
            vect = model[word.lower()]
            count = count + 1.0
        elif len(vect) == 0:
            vect = model['UNKNOWN']
            count = count + 1.0
        elif word in model:
            vect = vect + model[word.lower()]
            count = count + 1.0
    #        debugWords.append(word)
        else:
            vect = vect + model['UNKNOWN']
            count = count + 1.0
    if count == 0:
        print('Error_Vocab Code 1')
    return vect/count


def getSample(sample,model):
    model = wordvec
    vector  = [] 

    #debugWords = []
    for word in sample:
        if word in model:
            vector.append(model[word.lower()])
    #        debugWords.append(word)
        else:
            vector.append(model['UNKNOWN'])
    #        debugWords.append('UNK')
    for i in range(max_length-len(sample)):
        vector.append(np.zeros(n_input))
        
    #print debugWords
    return vector
       

def loadData(train_data,model,start,bs):
    model = wordvec
    train = []
    label = []
    not_found = 0
    counter = 0
    no_len = 0
    for line in train_data[start:start+bs]:
        data = line[:-1].split('\t')
        counter = counter + 1
        if len(data) == 2:        
            if data[0] in model:
                label.append(model[data[0]])
                train.append(np.array( getSample(data[1].split(),model) , dtype = np.float).astype(np.float32))   
                
            else:
                #print 'Not Found :',data[0]
                not_found = not_found + 1
        else:
            print('Length Error')  
            no_len = no_len + 1
    #print 'Count',str(counter),'Not found',str(not_found),str(no_len)
    return np.array(train, dtype = np.float).astype(np.float32), np.array(label, dtype = np.float).astype(np.float32)    
    

def checkExist(line,model):
    model = wordvec
    data = line[:-1].split('\t')
    if len(data) == 2:        
        if data[0] not in model:
            return False
    else:
        return False
    if len(data[1].split()) == 0:
        return False
    return True
        
    
def loadDataTest(line,model):
    model = wordvec
    train = []
    label = []
    not_found = 0
    counter = 0
    no_len = 0
    data = line[:-1].split('\t')
    counter = counter + 1
    if len(data) == 2:        
        if data[0] in model:
            label.append(model[data[0]])
            train.append(np.array(getSample(data[1].split(),model), dtype = np.float).astype(np.float32))   
        else:
            #print 'Not Found :',data[0]
            not_found = not_found + 1
    else:
        print('Length Error')    
        no_len = no_len + 1
    return np.array(train, dtype = np.float).astype(np.float32),np.array(avgSample(data[1].split(),model), dtype = np.float).astype(np.float32), np.array(label, dtype = np.float).astype(np.float32)    

def checkLine(filename):
    counter = 0
    with open(filename+'.txt','r') as pos:
        for p in pos:
            counter = counter + 1
    return counter


def cos(a,b):
    ab=np.sum(np.multiply(a,b))
    aa=np.sum(np.multiply(a,a))
    bb=np.sum(np.multiply(b,b))
    return ab/(aa*bb)


def ranking(ref_vec,vec,model):
    hit = 1
    model = wordvec
    ranked = 1
    items = model.most_similar(ref_vec,topn = 10)
    w_score = float((1 - scipy.spatial.distance.cosine(ref_vec,vec)))
    for k,v in items:
        if float((  1 - scipy.spatial.distance.cosine(ref_vec,model[k])  )  ) > w_score:
            ranked = ranked + 1
    if ranked == 11:
        hit = 0
        ranked = 1
        items = model.most_similar(ref_vec,topn = 180)
        for k,v in items:
            if float((  1 - scipy.spatial.distance.cosine(ref_vec,model[k])  )  ) > w_score:
                ranked = ranked + 1
    return float(1/ranked) , int(hit)