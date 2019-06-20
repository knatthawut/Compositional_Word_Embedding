'''
Implementation of Preprocessing wikipedia data for training Word2Vec model
'''

import textacy
import string
import re
import sys

# Init punctuation string without / _
punctuation_string = ''
for c in string.punctuation:
    if c != {'_','/'}:
        punctuation_string += c

def cleanLine(line):
    pat = re.compile(r'[^\w-]')
    replace = ' '    
    clean = ''
    items = line.split(' ')
    if items[0].lower() == 'null':
        items[0] = ''
        if len(items) > 2 and items[2].lower() == 'redirect':
            items[2] = ''
    tmp = ''
    for item in items:
        if "DBPEDIA_ID" in item:
            idx = item.find("DBPEDIA_ID")
            if idx != 0:
                item = item[:idx].lower()+' '+item[idx:]
            tmp = pat.sub(replace,tmp)
            clean = clean +' '+tmp+' '+item
            tmp = ''
        elif item != '':
            if tmp != '':
                tmp = tmp + ' '
            tmp = tmp + item.lower()
    tmp = pat.sub(replace,tmp)
    clean = clean +' '+tmp
    clean = clean.replace('  ',' ')
    return clean.strip()

def clean_text(text):
    cleaned = cleanLine(text)
    cleaned = textacy.preprocess.preprocess_text(cleaned, lowercase=False, no_urls=True, no_emails=True, no_phone_numbers=True, no_numbers=True, no_currency_symbols=True, no_contractions=True, no_accents=True)
    # cleaned = textacy.preprocess.remove_punct(cleaned,marks=punctuation_string)
    cleaned = textacy.preprocess.normalize_whitespace(cleaned)
    return cleaned

def main():
    if len(sys.argv) < 3:
        print('Usages input_file output_file')
        sys.exit()
    input_file = str(sys.argv[1])
    output_file = str(sys.argv[2])


    fout = open(output_file,'w',encoding = 'utf-8')
    with open(input_file,'r',encoding = 'utf-8') as fin:
        for line in fin:
            clean_line = clean_text(line)

if __name__ == '__main__':
    main()