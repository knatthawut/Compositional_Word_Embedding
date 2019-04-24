from gensim.models.keyedvectors import KeyedVectors

def getCompoundList(filename):
    cwList = []
    with open(filename,'r') as fp:
        for line in fp:
            words = line.split()
            for word in words:
                if '_' in word:
                    cwList.append(word)
    return cwList

def getTrainingList(cwList,model):
    listTrainingWord = []
    for cw in cwList:
        iwords = cw.split('_')
        err = False
        for w in iwords:
            if w not in word_vectors:
                err = True
        if not err:
            listTrainingWord.append(iwords)
    #for l in listTrainingWord:
    #    print l
    return listTrainingWord

if __name__ == "__main__":
    word_vectors = KeyedVectors.load_word2vec_format('concept2_100.bin', binary=True)
    #train = getTrainingList(getCompoundList(('medline17n0001.txt')), word_vectors)
    with open('training_concept', 'w') as fp:
        for t in word_vectors.vocab:
            if '_' in t:
                w = t.split('_')
                try:
                    fp.write(u' '.join(w)+'\t'+t+'\n')
                except UnicodeEncodeError :
                    print 'error'
