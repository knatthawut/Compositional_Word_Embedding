'''
    Extract the raw text from the Encow 16 corpus with XML files

'''
# Import Libraries
import sys


# Import modules


# Function Implementations

def main(input_file,output_file):
    # Open Input file and read each line
    fout = open(output_file,'w',encoding='utf-8')
    with open(input_file,'r',encoding='utf-8') as fin:
        for i, line in enumerate(fin):
            print('Processing {} line: {}'.format(i,line))
            
            if (i>200):
                break



if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print("Usages input_file output_file")
        sys.exit()
    
    input_file = str(sys.argv[1])
    output_file = str(sys.argv[2])
    main(input_file,output_file)
