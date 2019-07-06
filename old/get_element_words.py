'''
    Get element words for the input file

'''
# Import Libraries
import sys

def load_compound_dict(ref_file):
    res = {}

    with open(ref_file,'r',encoding='utf-8') as fin:
        for line in fin:
            tmp = line.split('\t')
            y_string = tmp[0]
            x_string = tmp[1]
            res[y_string] = x_string

    return res

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print('Usage: input_file output_file reference_file')
        sys.exit()
    
    input_file = str(sys.argv[1])
    output_file = str(sys.argv[2])
    ref_file = str(sys.argv[3])

    compound_dict = load_compound_dict(ref_file)
    # create output_file to output
    fout = open(output_file,'w',encoding='utf-8')
    # open input file
    with open(input_file,'r',encoding='utf-8') as fin:
        for line in fin:
            compound = line.strip()
            print('#'+compound+'#')
            if compound in compound_dict:
                element = compound_dict[compound]
            else:
                element = compound
                print('Not in Dict')
            fout.write(compound+'\t'+element+'\n')
    
    fout.close()