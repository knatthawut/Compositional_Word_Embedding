'''
    Extract the raw text from the Encow 16 corpus with XML files

'''
# Import Libraries
import sys
import re


# Import modules

# Constants
sent_begin_regex = re.compile(r'<s\sxid=.*>')
tag_regex = re.compile(r'<[a-z\/]+>')
nc_start_regex = re.compile(r'<nc>')
nc_end_regex = re.compile(r'</nc>')

# Function Implementations
def is_sent_start(line):
    '''
    Check if that line is the start of sentence
    '''
    return is_tag(line,sent_begin_regex)

def is_sent_end(line):
    return (line == '</s>')

def is_tag(token,tag_regex=tag_regex):
    match = re.match(tag_regex,token)
    if match:
        return True
    else:
        return False

   
def is_nc_start(token):
    return is_tag(token,nc_start_regex) 

def is_nc_end(token):
    return is_tag(token,nc_end_regex)

def get_compound_text(compound):
    '''
    Convert a list of tag compound into the form
    COMPOUND_ID/self-governance_ANCHORTEXTSTARTHERE_self-governed

    '''
    res = 'COMPOUND_ID/'
    compound = compound[1:-1]
    compound = [token for token in compound if not is_tag(token)]
    if len(compound) < 2:
        return compound[0] + ' '
    res = res + '_'.join(compound) + '_ANCHORTEXTSTARTHERE_' + '_'.join(compound) + ' '
    # print('Compound: ',res)
    return res

def get_text(sent):
    '''
    Convert sent (list of words)
    '''
    res = ''
    is_in_compound = False
    for token in sent:
        if is_tag(token,tag_regex):
            if is_nc_start(token):
                compound = []
                is_in_compound = True
                compound.append(token)
            elif is_nc_end(token):
                compound.append(token)
                compound_text = get_compound_text(compound)
                res = res + compound_text
                is_in_compound = False
        else: # Normal word
            if is_in_compound:
                compound.append(token)
            else:
                res = res + token + ' '

    res = res.strip()
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
                print('Processed Sent: ',raw_text)
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
