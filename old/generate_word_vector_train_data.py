'''
Implement for Generate Word Vector training data for the Wiki2Vec
Convert DBPEDIA_ID/Computer_accessibility_ANCHORTEXTSTARTHERE_Computer_accessibility
into 2 line
Line 1: DBPEDIA_ID/Computer_accessibility
Line 2: Computer accessibility
'''

import re

input_file = ''
output_file = ''

regex_pattern = re.compile(r'((DBPEDIA|COMPOUND)\_ID\/[^\s]+)\_ANCHORTEXTSTARTHERE\_([^\s]+)') 

def generate_first_line(line):
    '''
    Function:
        Generate Line 1: DBPEDIA_ID/Computer_accessibility
    Input:
        line(str): input line
    Output:
        res(str): output line
    '''
    # Init res is empty string
    res = ''
    # Find and replace all DBPEDIA_ID/Computer_accessibility_ANCHORTEXTSTARTHERE_Computer_accessibility with DBPEDIA_ID/Computer_accessibility
    res = re.sub(regex_pattern,r'\1',line)
    return res

def my_replace(match):
    return match.group(3).replace('_',' ')

def generate_second_line(line):
    '''
    Function:
        Generate Line 2: Computer accessibility
    Input:
        line(str): input line
    Output:
        res(str): output line
    '''
    # Init res is empty string
    res = ''
    # Find and replace all DBPEDIA_ID/Computer_accessibility_ANCHORTEXTSTARTHERE_Computer_accessibility with Computer accessibility
    res = ''
    res = re.sub(regex_pattern,my_replace,line)
    return res

def main():
    # Open output file to write
    fout = open(output_file,'w',encoding = 'utf-8')
    # Open input file to read
    with open(input_file,'r',encoding = 'utf-8') as fin:
        for line in fin: #for every line in input_file
            # 
            line = line.strip()
            line_1 = generate_first_line(line) + '\n'
            line_2 = generate_second_line(line) + '\n'
            fout.write(line_1)
            fout.write(line_2)
    fout.close()
if __name__ == '__main__':
    # test()
    main()