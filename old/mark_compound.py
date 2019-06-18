'''
Mark the compound word in the wiki data to train the word embedding.
'''
import sys
import re
import spacy
from spacy.tokenizer import Tokenizer

prefix_re = re.compile(r'''^DBPEDIA_ID\/[^\s]+''')
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
            line = line.strip()
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
    count = 0
    # Open Wiki_input_file to write
    with open(wiki_input_file,'r',encoding = 'utf-8') as fin:
        for j, line in enumerate(fin):
            if j % 100 == 0:
                print('Process {} lines'.format(str(j)))
            new_line = ''
            check2word = False
            # Process 4 surface: 'dress code','dress codes','dresscode','dresscodes'
            # Remove _ and - without losing DBPEDIA_ID/compound_word
            doc = nlp(line)
            words = [t.text for t in doc]
            # get all unigram of the line
            for i in range(len(words)):
                if check2word:
                    check2word = False
                    continue
                # process unigram words[i]
                if words[i] in compound_word_dict:
                    # meet dresscode or dresscodes
                    # print('Compound Word: ', compound_word_dict[words[i]])
                    count += 1
                    new_line = new_line + ' ' + compound_word_dict[words[i]]
                elif (i < (len(words)-1) ):
                    if words[i] + words[i+1] in compound_word_dict:
                        # Meet dress code or dress codes
                        # print('Compound Word: ', compound_word_dict[words[i]])
                        count += 1
                        new_line = new_line + ' ' + compound_word_dict[words[i]+words[i+1]]
                        check2word = True
                    else: new_line = new_line + ' ' + words[i]
                else: new_line = new_line + ' ' + words[i]
                # print('New line: ',new_line)
            new_line = new_line + '\n'
            fout.write(new_line)



def main():
    if len(sys.argv) < 4:
        print('Usages compound_file input_file output_file')
        sys.exit()
    compound_list_input = str(sys.argv[1])
    wiki_input_file = str(sys.argv[2])
    output_file = str(sys.argv[3])

    compound_word_dict = create_compound_dict(compound_list_input)
    # print(compound_word_dict)
    mark_on_text(wiki_input_file,output_file,compound_word_dict)

if __name__ == '__main__':
    main()
