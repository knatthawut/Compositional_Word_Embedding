from baseline import Baseline
import numpy as np

class AVG_baseline(Baseline):
    def __init__(self, type_of_wordvec):
        super().__init__('AVG',type_of_wordvec)

    def predict(self,x_test,wordvec):
        '''
        Use Average to predict from elements of compound words
        Input:  x_test is array of N(row) compound words, each row consists of MAX_SEQUENCE_LENGTH word_index because of padding with 0.
                wordvec: Gensim Word2Vec model.
        Output: an result array with N(row), each row is a vector of embedding dim of wordvec model
        '''
        # Init return value
        result = []

        # Calculate the average vector
        for sample in x_test:                                        # Iterating for each compound word
            words = sample[sample !=  0]                             # Eliminate of zeros padding
            res = []
            for word_idx in words:                                   # Iterating each element word in compound
                word = wordvec.wv.index2word[word_idx]               # Get word from word index          
                vector = wordvec.wv[word]                            # Get Vector for that word 
                res = res.append(vector)                                # Append into temporary res
            res = np.array(res)                                         # Convert list of numpy array into 2D array
            mean_res = res.mean(axis = 0)                               # Get the mean of each dimension
            result.append(mean_res)                                     # add into final result

        result = np.array(result)
        return result
    