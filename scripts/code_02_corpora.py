#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Code for Course on Text Analysis

August 2017

@author: Elliott Ash
"""
# set this to your working directory
PATH = '/Users/malkaguillot/Downloads/text_ml_course_2018-master'
import os
WORKING_DIR = os.path.join(PATH,'data')
os.chdir(WORKING_DIR) # set working directory 

# Pandas Data-frames

# open dataset
import pandas as pd
df1 = pd.read_csv('death-penalty-cases.csv')
df1.head() # show top few lines of data

df1.info() 

df1['court_id'].value_counts()

df1[['year','citeCount']].hist()

###################################
# Iterating over documents in a dataframe
###################################

WORKING_DIR = os.path.join(PATH,'scripts')
os.chdir(WORKING_DIR) # set working directory 
from utils import process_document

processed = {} # empty python dictionary for processed data
# iterate over rows
for i, row in df1.iterrows():
    docid = row['court_id'] # get document identifier
    text = row['snippet']     # get text snippet
    document = process_document(text) # get sentences/tokens
    processed[docid] = document # add to dictionary    
      
    
###################################
# Iterating over documents in text files
###################################

# select all files in your directory
from glob import glob
fnames = glob('contracts/*txt') # selects files using wildcards

# iterate over files
for fname in fnames:
    docid = fname.split('/')[-1][:-4] # get docid from filename
    text = open(fname).read() # read file as string
    document = process_document(text) # get sentences/tokens
    processed[docid] = document # add to dictionary
    
###################################
# Saving data in python
###################################

# save as python pickle
pd.to_pickle(processed, 'processed_corpus.pkl')

# other options:
# pd.to_csv
# pd.to_excel
# pd.to_stata
