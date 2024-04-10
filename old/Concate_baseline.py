'''
The file to implement Average Baseline for estimating the distributed representation of Compound words
                                                Sum( w1_vec + w2_vec + ...  + wn_vec )
Compound_Word_Vector_estimated (w1w2...wn)  =   ______________________________________
                                                                    N
'''
from baseline import Baseline
import numpy as np

class Concatenate_baseline(Baseline):
    def __init__(self, type_of_wordvec):
        super().__init__('Concatenate',type_of_wordvec)

    def get_vector_for_one_compound_from_id(self, compound, wordvec):
        words = compound[compound !=  0]                         # Eliminate of zeros padding
        res = []
        for word_idx in words:                                   # Iterating each element word in compound
            word = wordvec.wv.index2word[word_idx]               # Get word from word index
            vector = wordvec.wv[word]                            # Get Vector for that word
            res.append(vector)                             # Append into temporary res
        concate_res = np.concatenate(res)                            # Get the mean of each dimension
        return concate_res

    def get_vector_for_one_compound(self, compound, wordvec):
        words = compound[compound !=  0]                         # Eliminate of zeros padding
        res = []
        for word_idx in words:                                   # Iterating each element word in compound
            word = wordvec.wv.index2word[word_idx]               # Get word from word index
            vector = wordvec.wv[word]                            # Get Vector for that word
            res.append(vector)                             # Append into temporary res
        concate_res = np.concatenate(res)                            # Get the mean of each dimension
        return concate_res


    def train(self, x_train, y_train,num_of_epochs,batch_size):
        pass
        #     raise NotImplementedError

    def save_model(self, save_path):
        pass

    def predict(self,x_test,wordvec):
        '''
        Use Concate Baseline to predict the estimated vector of compound words from vector of element words
        Input:  x_test is array of N(row) compound words, each row consists of MAX_SEQUENCE_LENGTH word_index because of padding with 0.
                wordvec: Gensim Word2Vec model.
        Output: an result array with N(row), each row is a vector of embedding dim of wordvec model
        '''
        # Init return value
        result = []

        # Calculate the average vector
        for compound in x_test:                                  # Iterating for each compound word
            result.append(self.get_vector_for_one_compound(compound,wordvec))                                  # add into final result

        result = np.array(result)
        return result

    def predict_for_all(self, X , wordvec):
        # Init return value
        result = {}
        # Calculate the average vector
        for compound in X:                                      # Iterating for each compound word
            result[compound] = self.get_vector_for_one_compound(compound,wordvec)                               # add into final result

        return result
