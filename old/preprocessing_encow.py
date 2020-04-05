'''
    Preprocessing special char in encow dataset.
'''

# import textacy
import string
import re
import sys
from gensim.parsing.preprocessing import split_alphanum, strip_multiple_whitespaces, strip_tags

# Init punctuation string without / _
# punctuation_string = ''
# for c in string.punctuation:
#     if c != {'_','/'}:
#         punctuation_string += c

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
word_regex = re.compile(r'[^A-Za-z\_\/\/\s]+')
tag_regex = re.compile(r'&[a-z]+')

def remove_punct(line):
    clean = str(line)
    for punct in puncts:
        if punct in clean:
            clean = clean.replace(punct,'') 
    return clean

def remove_number(line):
    x = str(line)
    if bool(re.search(r'\d',x)):
        x = re.sub(r'[0-9]+','',x)
    return x

def remove_tag(line):
    x = re.sub(tag_regex,'',line)
    return x 

def cleanLine(line):
    clean = line.encode('ascii','ignore').decode('ascii')
    # clean = split_alphanum(clean)
    # clean = strip_tags(clean)
    # clean = strip_multiple_whitespaces(clean)
    # clean = remove_punct(clean)
    # clean = remove_number(clean)
    clean = remove_tag(clean)
    clean = re.sub(word_regex,'',clean)
    return clean.strip()

def clean_text(text):
    cleaned = cleanLine(text)
    # cleaned = textacy.preprocessing.preprocess_text(text, lowercase=False, no_urls=True, no_emails=True, no_phone_numbers=True, no_numbers=True, no_currency_symbols=True, no_contractions=True, no_accents=True)
    # cleaned = textacy.preprocess.remove_punct(cleaned,marks=punctuation_string)
    # cleaned = textacy.preprocess.normalize_whitespace(cleaned)
    return cleaned

def process(input_file,output_file):
    fout = open(output_file,'w',encoding = 'utf-8')
    with open(input_file,'r',encoding = 'utf-8') as fin:
        for line in fin:
            clean_line = clean_text(line)
            print(clean_line)
            fout.write(clean_line)
            fout.write('\n')
    fout.close()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usages input_file output_file')
        sys.exit()
    input_file = str(sys.argv[1])
    output_file = str(sys.argv[2])
    process(input_file,output_file)
