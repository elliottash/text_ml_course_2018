#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Code for Course on Text Analysis

August 2017

@author: Elliott Ash
"""

import pandas as pd
import os

# set this to your working directory
PATH = '/Users/malkaguillot/Documents/GitHub/text_ml_course_2018'

WORKING_DIR = os.path.join(PATH,'data')
os.chdir(WORKING_DIR) # set working directory 
df1 = pd.read_csv('death-penalty-cases.csv')

text = "Prof. Milano hailed from Milano. She got 3 M.A.'s from Bocconi."

###################################
# Splitting into sentences
###################################

from nltk import sent_tokenize
sentences = sent_tokenize(text) # split document into sentences
print(sentences)

import spacy
nlp = spacy.load('en')
doc = nlp(text)
sentences = list(doc.sents)
print(sentences)

#####
# Capitalization
#####

text_lower = text.lower() # go to lower-case

#####
# Punctuation
#####

# recipe for fast punctuation removal
from string import punctuation
import sys
if (sys.version)[0]=='3':
    translator = str.maketrans('','',punctuation)
    text_nopunc = text_lower.translate(translator)
if (sys.version)[0]=='2':
    text_nopunc = text_lower.translate(None, punctuation)
print(text_nopunc) 

#####
# Tokens
#####

tokens = text_nopunc.split() # splits a string on white space
print(tokens)

#####
# Numbers
#####

# remove numbers (keep if not a digit)
no_numbers = [t for t in tokens if not t.isdigit()]
# keep if not a digit, else replace with "#"
norm_numbers = [t if not t.isdigit() else '#' 
                for t in tokens ]
print(no_numbers )
print(norm_numbers)

#####
# Stopwords
#####

from nltk.corpus import stopwords
stoplist = stopwords.words('english') 
# keep if not a stopword
nostop = [t for t in norm_numbers if t not in stoplist]
print(nostop)

#####
# Stemming
#####

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('german') # snowball stemmer, german
print(stemmer.stem("Autobahnen"))
stemmer = SnowballStemmer('english') # snowball stemmer, english
# remake list of tokens, replace with stemmed versions
tokens_stemmed = [stemmer.stem(t) for t in tokens]
print(tokens_stemmed)

# other options:
# from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer

#####
# Corpus statistics
#####
docs = df1['snippet']

print(len(sentences),'sentences in corpus.')
print(len(tokens),'words in corpus.')
words_per_sent = len(tokens) / len(sentences)
print(words_per_sent,'words per sentence.')

#####
# Bag of words representation
#####

from collections import Counter
freqs = Counter(tokens)
freqs.most_common()[:20]



