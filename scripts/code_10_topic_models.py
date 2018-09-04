#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:12:15 2017

@author: elliott
"""

# set this to your working directory
WORKING_DIR = '/home/elliott/Dropbox/_Ash_Teaching/2016-08  - Max Planck - Machine Learning/code/'
os.chdir(WORKING_DIR)
import pandas as pd
df1 = pd.read_csv('death-penalty-cases.csv')
X = pd.read_pickle('X.pkl')
X_tfidf = pd.read_pickle('X_tfidf.pkl')

###
# Cosine Similarity
###

# compute pair-wise similarities between all documents in corpus"
from sklearn.metrics.pairwise import cosine_similarity

sim = cosine_similarity(X[:100])
sim.shape
sim[:3,:3]

tsim = cosine_similarity(X_tfidf[:100])
tsim[:3,:3]

###
# K-means clustering
###

# create 5 clusters of similar documents
from sklearn.cluster import KMeans
num_clusters = 100
km = KMeans(n_clusters=num_clusters,n_jobs=-1)
km.fit(X_tfidf[:1000])
doc_clusters = km.labels_.tolist()
dfs = df1[:1000]
dfs['cluster'] = doc_clusters
dfs[dfs['cluster']==1]['snippet']

###
# Latent Dirichlet Allocation
###

# clean document
from utils import clean_document
doc_clean = [clean_document(doc) for doc in df1['snippet'][:1000]]
# note: removed 'death' and 'penalty'

# randomize document order
from random import shuffle
shuffle(doc_clean)

# creating the term dictionary
from gensim import corpora
dictionary = corpora.Dictionary(doc_clean)

# creating the document-term matrix
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# config gensim logging feature
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# train LDA with 10 topics and print
from gensim.models.ldamodel import LdaModel
lda = LdaModel(doc_term_matrix, num_topics=10, 
               id2word = dictionary, passes=3)
lda.show_topics(formatted=False)

###
# Get topic proportions for a document
###

# to get the topic proportions for a document, use
# the corresponding row from the document-term matrix.
lda[doc_term_matrix[0]]

# or, for all documents
[lda[d] for d in doc_term_matrix]


###
# LDA Word Clouds
###

from numpy.random import randint
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# make directory if not exists
from os import mkdir
try:
    mkdir('lda')
except:
    pass

# make word clouds for the topics
for i,weights in lda.show_topics(num_topics=-1,
                                 num_words=100,
                                 formatted=False):
    
    #logweights = [w[0], np.log(w[1]) for w in weights]
    maincol = randint(0,360)
    def colorfunc(word=None, font_size=None, 
                  position=None, orientation=None, 
                  font_path=None, random_state=None):   
        color = randint(maincol-10, maincol+10)
        if color < 0:
            color = 360 + color
        return "hsl(%d, %d%%, %d%%)" % (color,randint(65, 75)+font_size / 7, randint(35, 45)-font_size / 10)   

    
    wordcloud = WordCloud(background_color="white", 
                          ranks_only=False, 
                          max_font_size=120,
                          color_func=colorfunc,
                          height=600,width=800).generate_from_frequencies(dict(weights))

    plt.clf()
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis("off")
    plt.show()
















