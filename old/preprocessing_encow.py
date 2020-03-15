'''
    Preprocessing special char in encow dataset.
'''

import textacy
import string
import re
import sys

# Init punctuation string without / _
# punctuation_string = ''
# for c in string.punctuation:
#     if c != {'_','/'}:
#         punctuation_string += c

word_regex = re.compile(r'\W+')

def cleanLine(line):
    clean = ''
    clean = re.sub(word_regex,'',line)
    return clean.strip()

def clean_text(text):
    # cleaned = cleanLine(text)
    cleaned = textacy.preprocess.preprocess_text(text, lowercase=False, no_urls=True, no_emails=True, no_phone_numbers=True, no_numbers=True, no_currency_symbols=True, no_contractions=True, no_accents=True)
    # cleaned = textacy.preprocess.remove_punct(cleaned,marks=punctuation_string)
    cleaned = textacy.preprocess.normalize_whitespace(cleaned)
    return cleaned

def process(input_file,output_file):
    fout = open(output_file,'w',encoding = 'utf-8')
    with open(input_file,'r',encoding = 'utf-8') as fin:
        for line in fin:
            clean_line = clean_text(line)
            fout.write(clean_line)
    fout.close()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usages input_file output_file')
        sys.exit()
    input_file = str(sys.argv[1])
    output_file = str(sys.argv[2])
    process(input_file,output_file)