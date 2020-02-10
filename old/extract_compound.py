'''
Extract new compound for training data in the new Encow16 dataset
Input: Encow_16 text file
Output: list of the compound 
COMPOUND_ID/w1_w2 w1 w2
'''
# Import Libs
import argparse
import re


# Parse the input arguments
parser = argparse.ArgumentParser(description='Extract new compound from text file')
parser.add_argument('--input',type=str, metavar='', required=True, help='new text data input file')
parser.add_argument('--output',type=str, metavar='', required=True, help='new text data out file')
args = parser.parse_args()

# Define constants
# Define regex
compound_regex = re.compile(r'COMPOUND\_ID\/[^\s]+')


def getCompoundfromText(text):
'''
    extract compound (COMPOUND_ID/w1_w2) from text and return a list of compounds
'''
    res = re.findall(compound_regex,text)
    return res

def writeCompounds(fout, compounds)
'''
    Write a list of compounds into fout
    compounds example: COMPOUND_ID/robot_arm robot arm
'''
    for compound in compounds:
        fout.write(compound)
        fout.write('\t')
        core_compound = compound.replace('COMPOUND_ID/','')
        core_compound = core_compound.replace('_',' ')
        fout.write(core_compound)
        fout.write('\n')

if __name__ =='__main__':
    input_file = args.input
    output_file = args.output
    fout = open(output_file,'w',encoding='utf-8') 
    with open(input_file,'r',encoding='utf-8') as fin:
        for line in fin:
            compounds = getCompoundfromText(line)
            writeCompounds(fout, compounds)
    fout.close()

    