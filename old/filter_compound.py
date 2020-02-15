'''
Delete all compound with more than two word in the input file
'''
import argparse

parser = argparse.ArgumentParser(description='Filter all compound with more than two words')
parser.add_argument('--input',type=str, metavar='', required=True, help='Path of input file')
parser.add_argument('--output', type=str, metavar='', required=True, help='Path of output file')

args = parser.parse_args()

input_file = args.input
output_file = args.output

def main(input_file,output_file):
    fout = open(output_file,'w',encoding='utf-8')
    with open(input_file,'r',encoding='utf-8') as fin:
        for line in fin:
            num_element = len(line.split())
            if (num_element == 3):
                fout.write(line)
                fout.write('\n')
    fout.close()

if __name__ == '__main__':
    main(input_file,output_file)
