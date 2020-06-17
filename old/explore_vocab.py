'''
Function
    Explore Vocab in Word2Vec model
Input: 
    Word2Vec model path
    training file path of word2vec
    Training file of compositional models (list of compounds)
'''
import argparse
from gensim.models import Word2Vec

def loadWordVecModel(vector_file_path):
    '''
    Function
        Load Word2Vec model from Gensim
    Input:
        vector_file_path(str): path to the pretrained Gensim model
    Output:
        res(Word2Vec model)
        vocab_size(int): size of the vocabulary of the model
    '''
    res = None
    vocab_size = 0
    res = Word2Vec.load(vector_file_path) # Load the model from the vector_file_name
    res.wv.init_sims(replace=True)
    vocab_size = len(res.wv.vocab)
    print('Vocab_size: ',vocab_size)
    return res, vocab_size

def print_vocab_stats(word_vector, vocab_size, train_Word2Vec_file, list_compound_file):
    '''
    Function
        Print Statistics for the corpus from the Word2Vec model
        Q1: How many compounds is in the vocab?
        Q2: What is top-100 most frequent compounds in the vocab?
    Input:
        word_vector(Gensim Word2Vec model): the pretrained Gensim Word2Vec model
        vocab_size(int): size of vocabulary of the Word2Vec model
        train_Word2Vec_file(str): path to the training file of Word2Vec model
        list_compound_file(str): path to the compound list
    Output: 
        in_vocab(int): How many compounds is in the vocab?
        top_100_compounds(list of str): top-100 most frequent compounds in the corpus
    '''
    





def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Explore Vocab in Word2Vec model')
    parser.add_argument('--Word2Vec_model',type=str, metavar='', required=True, help='Path to the Word2Vec model')
    parser.add_argument('--train_Word2Vec',type=str, metavar='', required=True, help='Path to training file (raw text) of the Word2Vec model')
    parser.add_argument('--list_compound',type=str, metavar='', required=True, help='Path to list of compound file')
    
    args = parser.parse_args()
    Word2Vec_model_file = args.Word2Vec_model
    train_Word2Vec_file = args.train_Word2Vec
    list_compound_file = args.list_compound
    
    # Load word2vec model
    word_vector, vocab_size = loadWordVecModel(Word2Vec_model_file)
    print('Loaded Word2Vec model!')

    print_vocab_stats(word_vector,vocab_size,train_Word2Vec_file,list_compound_file)
    
if __name__ == '__main__':
    main()
