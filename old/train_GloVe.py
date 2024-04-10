'''
The implementation for training GloVe vector
'''
from gensim.models import word2vec
import sys
from gensim.test.utils import datapath
from glove import Corpus, Glove


def trainGloVe(sentences, size=vec_size, window=wind, no_threads=num_thread,num_of_epochs=num_of_epochs):
    '''
    Train the word Embedding using GloVe algorithm
    Input:
        sentences(list of list word): the training data
        size(int): dimension of the vector
        window(int): the distance between two words algo should consider to find some relationship between them
        min_count: min occurences consider in the vocab
    '''
    corpus = Corpus()

    corpus.fit(sentences,window=window)

    glove = Glove(no_components=size)
    glove.fit(corpus.matrix,epochs=num_of_epochs,no_threads=no_threads)
    glove.add_dictionary(corpus.dictionary)

    return glove
    

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print('Usages filename save_location vector_size window epochs thread')
        sys.exit()


    sentences = word2vec.LineSentence(datapath(sys.argv[1]))
    fname = sys.argv[2]
    vec_size = int(sys.argv[3])
    wind = int(sys.argv[4])
    num_of_epochs = int(sys.argv[5])
    num_thread = int(sys.argv[6])
    model = trainGloVe(sentences, size=vec_size, window=wind, no_threads=num_thread,num_of_epochs=num_of_epochs)

    model.save(fname)
    print("Finished Saved Vector to " + fname)