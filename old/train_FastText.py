'''
The implementation for training FastText in Gensim
'''

from gensim.models.fasttext import FastText
from gensim.models import word2vec
from gensim.test.utils import datapath
import sys

# def trainFastText(corpus_file,size=300,):
#     corpus_file = datapath(corpus_file)
#     # Init
#     model = FastText(size=size)

#     # Build Vocab
#     model.build_vocab(corpus_file=corpus_file)

#     # Train model
#     model.tarin(corpus_file=corpus_file,epochs)

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print('Usages filename save_location vector_size sg window min_count thread')
        sys.exit()


    sentences = word2vec.LineSentence(sys.argv[1])
    fname = sys.argv[2]
    vec_size = int(sys.argv[3])
    sg_mark = int(sys.argv[4])
    wind = int(sys.argv[5])
    min_words = int(sys.argv[6])
    num_thread = int(sys.argv[7])
    model = FastText(sentences, size=vec_size, sg=sg_mark, window=wind, min_count=min_words, workers=num_thread)

    model.save(fname)
    print("Finished Saved Vector to " + fname)