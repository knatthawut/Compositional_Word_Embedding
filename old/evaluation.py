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