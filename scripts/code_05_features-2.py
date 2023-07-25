#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 21:43:57 2018

@author: elliott
"""

#setup
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

# generate document counts for each word
f = get_docfreqs(df1['snippet']) 
# generate word cloud of words with highest counts
wordcloud = WordCloud().generate_from_frequencies(f) 

plt.imshow(wordcloud, interpolation='bilinear') 
plt.axis("off") 
plt.show()

#####
# POS tagging 
#####
text = 'Science cannot solve the ultimate mystery of nature. And that is because, in the last analysis, we ourselves are a part of the mystery that we are trying to solve.'

from nltk.tag import perceptron
from nltk import word_tokenize
import nltk
nltk.download('averaged_perceptron_tagger')
tagger = perceptron.PerceptronTagger()
tokens = word_tokenize(text)
tagged_sentence = tagger.tag(tokens)
tagged_sentence

#####
# Our first visualization
# Plot nouns and adjectives over time
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


# Get list of nouns, adjectives, and verbs from WordNet
from nltk import wordnet as wn
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
stop.add('well')

full_vocab = set()

for x in wn.wordnet.all_synsets('a'):
    full_vocab.add(x.lemma_names()[0].lower())
for x in wn.wordnet.all_synsets('n'):
    full_vocab.add(x.lemma_names()[0].lower())
for x in wn.wordnet.all_synsets('v'):
    full_vocab.add(x.lemma_names()[0].lower())

full_vocab = full_vocab - stop

pd.to_pickle(full_vocab,'full_vocab.pkl')


# Sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
polarity = sid.polarity_scores(text)
polarity

# show higest and lowest sentiment snippets
def get_sentiment(snippet):
    return sid.polarity_scores(snippet)['compound']
dfs['sentiment'] = dfs['snippet'].apply(get_sentiment)
dfs.sort_values('sentiment',inplace=True)
list(dfs[:2]['snippet'])

list(dfs[-2:]['snippet'])


##########
# N-grams
#####
from nltk import ngrams
grams = []
for i, row in df1.iterrows():
    for n in range(2,4):
        grams += list(ngrams(row['snippet'].lower().split(),n))
    if i == 10:
        break
Counter(grams).most_common()[:8]  

###
# Collocations: Point-Wise Mutual Information
###
from operator import mul
from functools import reduce
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')

def get_gmean(phrase, termfreqs):
    words = phrase.split('_')
    n = len(words)
    p = [termfreqs[w]**(1/n) for w in words]
    numerator = termfreqs[phrase]   
    denominator = reduce(mul, p)
    if denominator == 0:
        return 0
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
           'AN','NN', 'VN', 'VV', 
            #'NV',
            'VP',                                    
            'NNN','AAN','ANN','NAN','NPN',
            'VAN','VNN', 'AVN', 'VVN',
            'VPN', 'VDN', 
            #'ANV','NVV','VVV', 'NNV',
            'VVP','VAV','VVN',
            'NCN','VCV', 'ACA',  
            'PAN',
            'NCVN','ANNN','NNNN','NPNN', 'AANN' 'ANNN','ANPN','NNPN','NPAN', 
            'ACAN', 'NCNN', 'NNCN', 'ANCN', 'NCAN',
            'PDAN', 'PNPN',
            'VDNN', 'VDAN','VVDN'}

max_phrase_length = 4

termfreqs = Counter()

docs = pd.read_pickle('processed_corpus.pkl')

for i, doc in enumerate(docs.values()):
    if i > 2000:
        break
    for sentence in doc:    
        # split into words and get POS tags
        tagwords = []
        for (word,tag) in tagger.tag(sentence):
            if tag in tagdict:
                normtag = tagdict[tag]
                stemmed = word#stemmer.stem(word)
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

# filter out unigrams
grams = [x for x in termfreqs.most_common() if '_' in x[0]]
# make dataframe of geometric mean associations for each gram
gmeans = pd.DataFrame([(gram[0], get_gmean(gram[0],termfreqs)) for gram in grams],
              columns=['ngram','gmean'])
gmeans
                        
# Dependency Parsing
import spacy
nlp = spacy.load('en')
doc = nlp(text)
for sent in doc.sents:
    print(sent)
    print(sent.root)
    print([(w, w.dep_) for w in sent.root.children])
    print()
    
