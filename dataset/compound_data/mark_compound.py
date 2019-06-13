'''
Mark the compound word in the wiki data to train the word embedding.
'''

import re
import spacy
from spacy.tokenizer import Tokenizer

prefix_re = re.compile(r'''^DBPEDIA_ID\/\w+''')
# infix_re = re.compile(r'''[.\~\-\_\(\)]''')

def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search)

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = custom_tokenizer(nlp)


def get_possible_surface(element_words):
    '''
    Function:
        Generate all the surfaces of a compound.
        Example: all the surfaces of the 'dresscode' could be
        dress code, dress_code, dress-code,dress codes, dress_codes, dress-codes 
    Input:
        element_words(list of str): list of element words from the compound words
    Output:
        res(list of str): all the surface of that compound
    '''
    res = []

    # Generate '' surface
    surface = ''.join(element_words)
    res.append(surface)

    # # Generate 'space' surface
    # surface = ' '.join(element_words)
    # res.append(surface)
    
    # Generate '_' surface
    surface = '_'.join(element_words)
    res.append(surface)

    # Generate '-' surface
    surface = '-'.join(element_words)
    res.append(surface)
    
    tmp = []
    # Generate Plural surface
    for surface in res:
        tmp.append(surface+'s')

    res = res + tmp
    return res

def create_compound_dict(input_file):
    '''
    Function:
        Create a dictionary of Compound word from input_file
    Input:
        input_file(string): name of input_file
    Output:
        res(dictionary): of compound word
        key: compound_word
        value: COMPOUND_ID/compound_word
    '''
    res = {}

    with open(input_file,'r', encoding = 'utf-8') as fin:
        for line in fin:
            tmp = line.split('\t')
            compound_id = tmp[0]
            element_words = tmp[1].split(' ')
            posible_surface_list = get_possible_surface(element_words)
            for surface in posible_surface_list:
                res[surface] = compound_id
    return res


def mark_on_text(wiki_input_file,output_file,compound_word_dict):
    '''
    Function:
        Mark all the surface of the compound word in the wiki_input_file into COMPOUND_ID/ and write into output_file
    Input:
        wiki_input_file(str): name of the input file
        output_file(str): name of the output file
        compound_word_dict: dictionary of compound word:
        key: surface of compound_word
        value: COMPOUND_ID/compound_word
    Output: NONE 
    '''
    # Open output_file to write
    fout = open(output_file,'w',encoding = 'utf-8')

    # Open Wiki_input_file to write
    with open(wiki_input_file,'r',encoding = 'utf-8') as fin:
        for line in fin:
            new_line = ''
            # Process 4 surface: 'dress code','dress codes','dresscode','dresscodes'
            # Remove _ and - without losing DBPEDIA_ID/compound_word
            doc = nlp(line)
            words = [t.text for t in doc]
            # get all unigram of the line
            for i in range(len(words)):
                print('At step {} process {}'.format(str(i),words[i]))
                # process unigram words[i]
                if words[i] in compound_word_dict:
                    # meet dresscode or dresscodes
                    new_line = new_line + compound_word_dict[words[i]]
                elif (i < (len(words)-1) ):
                    if words[i] + words[i+1] in compound_word_dict:
                        # Meet dress code or dress codes
                        new_line = new_line + compound_word_dict[words[i]+words[i+1]]
                else: new_line = new_line + words[i]
            new_line += '\n'
            fout.write(new_line)




def main():
    compound_list_input = 'compound_word.tsv'
    wiki_input_file = 'test'
    output_file = 'test.out'

    compound_word_dict = create_compound_dict(compound_list_input)
    mark_on_text(wiki_input_file,output_file,compound_word_dict)

if __name__ == '__main__':
    main()
