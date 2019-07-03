'''
Filter data without the UNKNOWN WORD in Tratz data
'''
#Import Libraries
from gensim.models import Word2Vec
import functools
import numpy as np
import sys
import os
import pprint
import pandas as pd

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

    # Load data
    df = pd.read_csv(input_file_name,sep='\t',encoding='utf-8')
    df.columns = ['word_1','word_2','label']
    df['Compound'] = 'COMPOUND_ID/' + df.word_1 + '_' + df.word_2

    print(df.head())
    df['inVocab'] = [x in word_vec.wv for x in df['Compound']]

    # df['inVocab'] = df['Compound'].isin(word_vec.wv)
    print(df.head())
    filtered_df = df[df['inVocab']==True]

    print(filtered_df.head())
    del filtered_df['inVocab']
    del filtered_df['Compound']
    print(filtered_df.head())

    filtered_df.to_csv(output_file_name,sep='\t',encoding='utf-8',header=False,index=False)

if __name__ == '__main__':
    main()
