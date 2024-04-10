'''
Convert Analogy test into Compounds.

'''
import argparse

def process(input_file,output_file):
    fout = open(output_file,'w',encoding='utf-8')

    with open(input_file,'r',encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            phrases = line.split(' ')
            for phrase in phrases:
                elements = phrase.split('_')
                compound = 'COMPOUND_ID/' + phrase + '\t' + ' '.join(elements) + '\n'
                fout.write(compound)
    fout.close()


if __name__ == '__main__':
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Extract Compound from Phrase Analogy Test')
    parser.add_argument('--input',type=str, metavar='', required=True, help='data input file')
    parser.add_argument('--output',type=str, metavar='', required=True, help='data out file')
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    process(input_file,output_file)
