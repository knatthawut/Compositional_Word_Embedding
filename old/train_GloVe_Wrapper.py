import glove_pywrapper
import sys

def trainGloVe(filename,vector_size=300,vocab_min_count=100,max_iter=15,window_size=10):
    CORPUS = "/home/an/an_workspace/Compositional_Word_Embedding/dataset/Wiki_data/" + filename
    model_name = '{}_window{}_mincount{}_iter{}'
    glove = glove_pywrapper.GloveWrapper(CORPUS, model_name,vocab_min_count=vocab_min_count,vector_size=vector_size,window_size=window_size,max_iter=max_iter)
    #prepare vocabulary count
    glove.vocab_count()
    #prepare co-occurrence matrix
    glove.cooccur()
    #reshuffle
    # glove.shuffle()
    #glove train
    glove.glove()


if __name__ == "__main__":
    if len(sys.argv) < 8:
        print('Usages filename vector_size window min_count iter')
        sys.exit()


    fname = sys.argv[1]
    vec_size = int(sys.argv[2])
    wind = int(sys.argv[3])
    min_count = int(sys.argv[4])
    num_of_iter = int(sys.argv[5])
    trainGloVe(fname,vector_size=vec_size,vocab_min_count=min_count,max_iter=num_of_iter,window_size=wind)
