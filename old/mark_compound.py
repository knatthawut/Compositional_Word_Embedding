'''
Mark the compound word in the wiki data to train the word embedding.
'''

def get_possible_surface(element_words):
    res = []

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
    '''

def main():
    compound_word_dict = create_compound_dict(input_file)
    mark_on_text(wiki_input_file,output_file,compound_word_dict)

if __name__ == '__main__':
    main()
