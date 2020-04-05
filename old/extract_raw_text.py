'''
    Extract raw text from train file.
    Extract line without pattern COMPOUND_ID/ or DBPEDIA_ID/
'''

import argparse

def process(input_file,output_file):
    fout = open(output_file,'w',encoding='utf-8')

    with open(input_file,'r',encoding='utf-8') as fin:
        for line in fin:
            if not(('COMPOUND_ID' in line) or ('DBPEDIA_ID' in line)):
                fout.write(line)
    fout.close()


if __name__ == '__main__':
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Extract raw text file from training file')
    parser.add_argument('--input',type=str, metavar='', required=True, help='data input file')
    parser.add_argument('--output',type=str, metavar='', required=True, help='data out file')
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    process(input_file,output_file)