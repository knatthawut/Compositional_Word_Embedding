'''
The file to implement Actual Baseline for estimating the distributed representation of Compound words
'''
from baseline import Baseline
import numpy as np
import utils
class Actual_baseline(Baseline):
    def __init__(self, type_of_wordvec):
        super().__init__('Actual',type_of_wordvec)
        self.count_not_in_vocab = 0
        self.count_in_vocab = 0

    def get_vector_for_one_compound(self, compound, wordvec):
        # print(compound)
        compound_id = 'COMPOUND_ID/'
        for i, idx in enumerate(compound):
            compound_id += wordvec.wv.index2word[idx]
            if i < (len(compound) - 1):
                compound_id += '_'
        # print(compound_id)
        if compound_id in wordvec.wv:
            return wordvec.wv[compound_id]
        else:
            print('Not in Vocab')
            return wordvec.wv['unknown']


    def train(self, x_train, y_train,num_of_epochs,batch_size):
        pass
        #     raise NotImplementedError

    def save_model(self, save_path):
        pass

    def predict(self,x_word,wordvec):
        '''
        Use Actual Baseline to predict the estimated vector of compound words from vector of element words
                wordvec: Gensim Word2Vec model.
        Output: an result array with N(row), each row is a vector of embedding dim of wordvec model
        '''
        # Init return value
        result = []

        # Calculate the average vector
        for compound in x_word:                                  # Iterating for each compound word
            result.append(self.get_vector_for_one_compound(compound,wordvec.wv))                                  # add into final result

        result = np.array(result)
        return result

    def predict_for_all(self, X , wordvec):
        # Init return value
        result = {}
        # Calculate the average vector
        for compound in X:                                      # Iterating for each compound word
            result[compound] = self.get_vector_for_one_compound(compound,wordvec)                               # add into final result

        return result
