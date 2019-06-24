'''
The implementation for training Word2Vec in Gensim
'''
from gensim.models import word2vec
import sys

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print('Usages filename save_location vector_size sg window min_count thread')
        sys.exit()


    sentences = word2vec.LineSentence(sys.argv[1])
    vec_size = int(sys.argv[3])
    sg_mark = int(sys.argv[4])
    wind = int(sys.argv[5])
    min_words = int(sys.argv[6])
    num_thread = int(sys.argv[7])
    if sg_mark == 1:
        type_model = 'SG'
    else:
        type_model = 'CBOW'

    fname = sys.argv[2] + '{}_size{}_window{}_min{}'.format(type_model,vec_size,wind,min_count)

    model = word2vec.Word2Vec(sentences, size=vec_size, sg=sg_mark, window=wind, min_count=min_words, workers=num_thread)

    model.save(fname)
    print("Finished Saved Vector to " + fname)
