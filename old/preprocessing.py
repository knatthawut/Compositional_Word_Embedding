'''
Implementation of Preprocessing wikipedia data for training Word2Vec model
'''

import textacy
import string

input_file = ''
output_file = ''

# Init punctuation string without / _
punctuation_string = ''
for c in string.punctuation:
    if c != {'_','/'}:
        punctuation_string += c

def clean_text(text):
    cleaned = textacy.preprocess.preprocess_text(text, fix_unicode=True, lowercase=False, no_urls=True, no_emails=True, no_phone_numbers=True, no_numbers=True, no_currency_symbols=True, no_contractions=True, no_accents=True)
    cleaned = textacy.preprocess.remove_punct(cleaned,marks=punctuation_string)
    cleaned = textacy.preprocess.normalize_whitespace(cleaned)
    return cleaned

def main():
    fout = open(output_file,'w',encoding = 'utf-8')
    with open(input_file,'r',encoding = 'utf-8') as fin:
        for line in fin:
            clean_line = clean_text(line)
            print(clean_line)
            break