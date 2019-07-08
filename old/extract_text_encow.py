'''
    Extract the raw text from the Encow 16 corpus with XML files

'''
# Import Libraries
import sys
import re


# Import modules

# Constants
sent_begin_regex = re.compile(r'<s\sxid=.*>') 

# Function Implementations
def is_sent_start(line):
    '''
    Check if that line is the start of sentence
    '''
    match = re.match(sent_begin_regex,line)
    if match:
        return True
    else:
        return False

def is_sent_end(line):
    return (line == '</s>')

def get_text(sent):
    '''
    Convert sent (list of words)
    '''
    res = ''
    for token in sent:
        if is_tag(token):
            if is_nc_start(token):
                
            elif is_nc_end(token):
        else: # Normal word

    return res

def main(input_file,output_file):
    # Open Input file and read each line
    fout = open(output_file,'w',encoding='utf-8')
    sent = []
    in_sent = False
    with open(input_file,'r',encoding='utf-8') as fin:
        for i, tmp in enumerate(fin):
            line = tmp.split('\t')[0]
            # print('Sent: ',sent)
            # print('Processing {} line: {}'.format(i,line))
            if is_sent_start(line):
                # print('End of Sent!!!!!')
                print('Sent ',sent)
                raw_text = get_text(sent[:-1])
                fout.write(raw_text+'\n')
                
                sent = []
            else:
                sent.append(line.strip())
            
            if (i>200):
                break
    fout.close()


if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print("Usages input_file output_file")
        sys.exit()
    
    input_file = str(sys.argv[1])
    output_file = str(sys.argv[2])
    main(input_file,output_file)
