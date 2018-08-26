#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Code for Course on Text Analysis

August 2017

@author: Elliott Ash
"""
# Setup
# set this to your working directory
WORKING_DIR = '/home/elliott/Dropbox/_Ash_Teaching/2018-09 - Bocconi - Text Data and ML/code'
import os
os.chdir(WORKING_DIR)

import pandas as pd
df1 = pd.read_csv('death-penalty-cases.csv')

###################################
# Screen Scraping
###################################

import urllib # Python's module for accessing web pages
url = 'https://goo.gl/VRF8Xs' # shortened URL for court case
page = urllib.request.urlopen(url) # open the web page

html = page.read() # read web page contents as a string
print(html[:400])  # print first 400 characters
print(html[-400:]) # print last 400 characters
print(len(html))   # print length of string

#############
# Translation
#############

from googletrans import Translator
translator = Translator()
lang = translator.detect('이 문장은 한글로 쓰여졌습니다.').lang
eng = translator.translate('이 문장은 한글로 쓰여졌습니다.',
                           src=lang,
                           dest='en')
eng.text

###################################
# HTML parsing
###################################

# Parse raw HTML
from bs4 import BeautifulSoup # package for parsing HTML
soup = BeautifulSoup(html, 'lxml') # parse html of web page
print(soup.title) # example usage: print title item

# extract text
text = soup.get_text() # get text (remove HTML markup)
lines = text.splitlines() # split string into separate lines
print(len(lines)) # print number of lines

lines = [line for line in lines if line != ''] # drop empty lines
print(len(lines)) # print number of lines
print(lines[:20]) # print first 20 lines

###################################
# Removing unicode characters
###################################

from unidecode import unidecode # package for removing unicode
fixed = unidecode('Visualizations\xa0') # example usage
print(fixed) # print cleaned string

##########
# Exploring a Corpus
##########
df1 = df1[['state','snippet']]
# Number of documents
len(df1['snippet'])
# Number of label categories (e.g. states)
df1['state'].describe()
# Number of samples per class
counts_per_class = df1.groupby('state').count()
counts_per_class.head()
# Words per sample
def get_words_per_sample(txt):
    return len(txt.split())
df1['num_words'] = df1['snippet'].apply(get_words_per_sample)
df1['num_words'].describe()
# Frequency distribution over words
from collections import Counter
freqs = Counter()
for i, row in df1.iterrows():
    freqs.update(row['snippet'].lower().split())
freqs.most_common()[:20]
# (Number of samples) / number of words per sample)
len(df1['snippet']) / df1['num_words'].mean()

