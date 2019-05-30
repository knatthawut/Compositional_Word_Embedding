'''
This file is used to implement all the evaluation of the system
'''
import numpy as np 
import scipy


def direction_distance(vecA,vecB):
    '''
    Calculate the eucliden distance for direction
    '''
    return np.linalg.norm(vecB - vecA) 

def location_distance(vecA,vecB):
    '''
    Calculate the location distance for location
    '''
    return scipy.spatial.distance.cosine(vecB,vecA)

def D_function(type_acc, ref_vec,predictA_vec,predictB_vec):
    '''
    Calculate the distance between ref_vec and predictA_vec or ref_vec and predictB_vec
    type_acc = 'DIR' or 'LOC'
    '''
    if type_acc == 'DIR':
        if direction_distance(ref_vec, predictA_vec) > direction_distance(ref_vec, predictB_vec):
            return 1
        else:
            return 0
    if type_acc == 'LOC':
        if location_distance(ref_vec, predictA_vec) < location_distance(ref_vec, predictB_vec):
            return 1
        else:
            return 0
      

def calculateAccuracy(type_acc, label, predictA, predictB):
    '''
    Calculate the Direction Accuracy of the System
    Input: label_vector(list of vectors), predictA(list of vectors), predictB(list of vectors)
            type_acc = 'DIR' or 'LOC'
    Output: Direction Accuracy based on the angle 
                    Sum( D_direction (ref_vector, predictA_vector, predictB_vector) )
    direction_acc = _________________________________________________________________
                                                        N 
                                                                                    
    D_direction (ref_vector, predictA_vector, predictB_vector) = 1 IF cosine_diff(ref_vector, predictA_vector) < cosine_diff(ref_vector, predictB_vector)
    
    '''     
    # Init return values
    N = len(label)
    Sum = 0.0
    # Calculate the accuracy
    for i in range(len(label)):     # run all sample in label 
        ref_vec = label[i]          # get the reference vector
        predictA_vec = predictA[i]  # get the predictA vector 
        predictB_vec = predictB[i]  # get the predictB vector
        Sum = Sum + D_function(type_acc, ref_vec,predictA_vec,predictB_vec) # Calculate the sum

    acc = Sum / (N*1.0)
    return acc

def getRanking(wordvec, compound_word, vec):
    '''
    Calculate the rank of the vector using the similar_by_vector function from Gensim
    Input:
        wordvec: Gensim word2vec model
        compound_word: the compound word need to be compare
        vec: the estimated vector of that compound word
    Output:
        rank: the rank of that compound_word when calculate the similarity by word
        if rank > 10: return 11
    similar_by_vector(vector, topn=10, restrict_vocab=None)
    Find the top-N most similar words by vector.

    Parameters:	
    vector (numpy.array) – Vector from which similarities are to be computed.
    topn ({int, False}, optional) – Number of top-N similar words to return. If topn is False, similar_by_vector returns the vector of similarity scores.
    restrict_vocab (int, optional) – Optional integer which limits the range of vectors which are searched for most-similar values. For example, restrict_vocab=10000 would only check the first 10000 word vectors in the vocabulary order. (This may be meaningful if you’ve sorted the vocabulary by descending frequency.)
    Returns:	
    Sequence of (word, similarity).

    Return type:	
    list of (str, float)
    '''
    top10 = wordvec.wv.similar_by_vector(vec,topn=10)
    for i, word_tuple in enumerate(top10):
        word = word_tuple[0]
        if word == compound_word:
            return i+1

    return 11




def calculateMRR_HIT(wordvec, label, baseline_predict):
    '''
        Calculate MRR and HIT result of baseline
        Input:  label: list of the actual compound word
                baseline_predict: list of the estimated representation for that word

    '''
    # Init return value
    MRR = 0.0
    HIT_1 = 0
    HIT_10 = 0 

    for i,compound_word in enumerate(label):
        vec = baseline_predict[i]
        rank = getRanking(wordvec, compound_word, vec)
        MRR = MRR + 1.0/rank
        if rank == 1:
            HIT_1 += 1
        elif rank < 11:
            HIT_10 += 1

    return MRR, HIT_1, HIT_10
