#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 21:43:57 2018

@author: elliott
"""

# set this to your working directory
WORKING_DIR = '/home/elliott/Dropbox/_Ash_Teaching/2018-09 - Bocconi - Text Data and ML/code'
import os
os.chdir(WORKING_DIR)
import pandas as pd
df1 = pd.read_csv('death-penalty-cases.csv')
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(min_df=0.01, # at min 1% of docs
                        max_df=.9,  
                        max_features=10000,
                        stop_words='english',
                        ngram_range=(1,3))
X = vec.fit_transform(df1['snippet'])
pd.to_pickle(X,'X.pkl')
vocabdict = vec.vocabulary_
vocab = [None] * len(vocabdict) 
for word,index in vocabdict.items():
    vocab[index] = word
pd.to_pickle(vocab,'vocab.pkl')


###
# tf-idf weights
####

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=0.01, # at min 1% of docs
                        max_df=0.9,  # at most 90% of docs
                        max_features=1000,
                        stop_words='english',
                        use_idf=True,
                        ngram_range=(1,3))

X_tfidf = tfidf.fit_transform(df1['snippet'])
pd.to_pickle(X_tfidf,'X_tfidf.pkl')

#####
# Our first word cloud
#####

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils import get_docfreqs

f = get_docfreqs(df1['snippet']) # makes python dictionary
wordcloud = WordCloud().generate_from_frequencies(f) 

plt.imshow(wordcloud, interpolation='bilinear') 
plt.axis("off") 
plt.show()

#####
# POS tagging in Python
#####

text = 'Science cannot solve the ultimate mystery of nature. And that is because, in the last analysis, we ourselves are a part of the mystery that we are trying to solve.'

from nltk.tag import perceptron
from nltk import word_tokenize
tagger = perceptron.PerceptronTagger()
tokens = word_tokenize(text)
tagged_sentence = tagger.tag(tokens)
tagged_sentence

#####
# Our first visualization
#####
from collections import Counter
from nltk import word_tokenize

def get_nouns_adj(snippet):
    tags = [x[1] for x in tagger.tag(word_tokenize(snippet))]
    num_nouns = len([t for t in tags if t[0] == 'N'])
    num_adj = len([t for t in tags if t[0] == 'J'])
    return num_nouns, num_adj

dfs = df1.sample(frac=.1)
dfs['nouns'], dfs['adj'] = zip(*dfs['snippet'].map(get_nouns_adj))
dfs.groupby('year')[['nouns','adj']].mean().plot()



##########
# Sentiment Analysis
#####

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
polarity = sid.polarity_scores(text)

def get_sentiment(snippet):
    return sid.polarity_scores(snippet)['compound']
dfs['sentiment'] = dfs['snippet'].apply(get_sentiment)
dfs.groupby('year')[['sentiment']].mean().plot()

##########
# N-grams
#####

from nltk import ngrams
grams = []
for n in range(2,4):
    grams += list(ngrams(tokens,n))
Counter(grams).most_common()[:8]  

###
# Collocations: Point-Wise Mutual Information
###

from operator import mul
from functools import reduce
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer()

def get_gmean(phrase, termfreqs):
    words = phrase.split('_')
    n = len(words)
    p = [termfreqs[w]**(1/n) for w in words]
    numerator = termfreqs[phrase]   
    denominator = reduce(mul, p)
    gmean = numerator / denominator
    return gmean


###
# POS-filtered N-grams
###

# Normalize Penn tags
tagdict = { 'NN':'N',
            'NNS':'N',
                                    
            'JJ':'A',
            'JJR':'A',
            'JJS':'A',
            'VBG':'A', # gerunds/participles treated like adjectives

            'RB':'A', # adverbs treated as adjectives
            'RBR':'A',
            'RBS':'A',
            'PDT':'A', # predeterminer            

            'VB':'V',
            'VBD':'V',
            'VBN':'V',
            'VBP':'V',
            'VBZ':'V',
            'MD': 'V', # modals treated as verbs
            'RP': 'V', # particles treated as verbs
            
            'DT':'D',
                        
            'IN':'P',
            'TO':'P',

            'CC': 'C'}

tagpatterns = {'A','N','V','P','C','D',
           'AN','NN', 'VN', 'VV', 'NV',
            'VP',                                    
            'NNN','AAN','ANN','NAN','NPN',
            'VAN','VNN', 'AVN', 'VVN',
            'VPN','ANV','NVV','VDN', 'VVV', 'NNV',
            'VVP','VAV','VVN',
            'NCN','VCV', 'ACA',  
            'PAN',
            'NCVN','ANNN','NNNN','NPNN', 'AANN' 'ANNN','ANPN','NNPN','NPAN', 
            'ACAN', 'NCNN', 'NNCN', 'ANCN', 'NCAN',
            'PDAN', 'PNPN',
            'VDNN', 'VDAN','VVDN'}

max_phrase_length = 3

termfreqs = Counter()

from nltk import sent_tokenize
sentences = sent_tokenize(text)
for sentence in sentences:    
    # split into words and get POS tags
    tagwords = []
    for (word,tag) in tagger.tag(sentence):
        if tag in tagdict:
            normtag = tagdict[tag]
            stemmed = stemmer.stem(word)
            tagwords.append((stemmed,normtag))
        else:
            tagwords.append(None)
    for n in range(1,max_phrase_length+1):            
        rawgrams = ngrams(tagwords,n)
        for rawgram in rawgrams:
            # skip grams that have rare words
            if None in rawgram:
                continue
            gramtags = ''.join([x[1][0] for x in rawgram])
            if gramtags in tagpatterns:
                 # if tag sequence is allowed, add to counter
                gram = '_'.join([x[0] for x in rawgram])
                termfreqs[gram] += 1
                                
###
# Dependency Parsing
###

import spacy
nlp = spacy.load('en')
doc = nlp(text)
for sent in doc.sents:
    print(sent.root)
    print([(w, w.dep_) for w in sent.root.children])
    print()
    