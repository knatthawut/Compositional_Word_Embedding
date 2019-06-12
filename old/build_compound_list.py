'''
Implementation for changing the format of the dataset to desired format
#COMPOUND_ID/compound_word  [tab]   w1  [space] w2  ... wn
#COMPOUND_ID/robot_arm  [tab]   robot [space] arm
'''

def format_comma(line):
    '''
    Function:
        Convert one line of format w1,w2,compound_word(w1_w2) to the desired format
    Input:
        line(string): the line needs to be converted
    Output:
        res(string): converted line
    '''
    element = line.strip().split(',')
    compound_word = element[-1]
    element_words = element[:-1]
    res = 'COMPOUND_ID/' + compound_word + '\t'
    for word in element_words:
        res = res + word + ' '

    res = res.strip(' ')

    return res


def convert_format_comma(input_file, output_file):
    '''
        Function:
            convert the compound dataset in input_file to the desired format
            The compound dataset in input_file has format:
            w1,w2,compound_word(w1_w2)
            Example: survey,committee,survey_committee
        Input:
            input_file(string): input file name for the function
            output_file(string): name of output file
        Output: None
    '''
    # Init File writter
    fout = open(output_file,'w',encoding = 'utf-8')
    
    # Open input_file to process
    with open(input_file,'r',encoding = 'utf-8') as fin:
        for line in fin:
            new_line = format_comma(line) + '\n'
            fout.write(new_line)
    
    fout.close()

def format_wiki(line):
    '''
    Function:
        Convert one line of format compoundword [space] w1-w2 [space] w1 [space] + [space] w2 [space] wikilink [space] full_wiki_link to the desired format
    Input:
        line(string): the line needs to be converted
    Output:
        res(string): converted line
    '''
    element = line.strip().split('\t')
    compound_word = element[0]
    element_words = element[1].split('-')
    res = 'COMPOUND_ID/' + compound_word + '\t'
    for word in element_words:
        res = res + word + ' '

    res = res.strip(' ')

    return res

def convert_format_wiki(input_file, output_file):
    '''
        Function:
            convert the compound dataset in input_file to the desired format
            The compound dataset in input_file has format:
            compoundword [space] w1-w2 [space] w1 [space] + [space] w2 [space] wikilink [space] full_wiki_link
            Example: abbeystead	abbey-stead	abbey + stead	/wiki/abbeystead	https://en.wiktionary.org/wiki/Category:English_compound_words
        Input:
            input_file(string): input file name for the function
            output_file(string): name of output file
        Output: None
    '''
    # Init File writter
    fout = open(output_file,'w',encoding = 'utf-8')
    
    # Open input_file to process
    with open(input_file,'r',encoding = 'utf-8') as fin:
        for line in fin:
            new_line = format_wiki(line) + '\n'
            fout.write(new_line)
    
    fout.close()

def main():
    input_file = 'English_downloaded'
    output_file = 'compound_word_2.tsv'
    convert_format_wiki(input_file,output_file)

if __name__ == '__main__':
    main()