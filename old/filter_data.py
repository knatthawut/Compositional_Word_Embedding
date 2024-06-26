'''
Filter data without the UNKNOWN WORD
'''
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
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

word2vec_file = ''
input_file_name = ''
output_file_name = ''

def main():
    # Load Word2Vec
    if len(sys.argv) < 4:
        print('Usages word2vec_file input_file output_file')
        sys.exit()

    word2vec_file = str(sys.argv[1])
    input_file_name = str(sys.argv[2])
    output_file_name = str(sys.argv[3])

    word_vec = Word2Vec.load(word2vec_file)
    word_vec.wv.init_sims(replace=True)
    vocab_size = len(word_vec.wv.vocab)

    fout = open(output_file_name,'w',encoding='utf-8')

    # Load data
    with open(input_file_name,'r', encoding = 'utf-8') as fin:
        for line in fin:
            tmp = line.split('\t')
            y_string = tmp[0]
            if y_string in word_vec.wv:
                fout.write(line+'\n')

    fout.close()

if __name__ == '__main__':
    main()
