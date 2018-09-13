#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:13:32 2017

@author: elliott
"""

import os
from txt_utils import WORK_DIR
os.chdir(WORK_DIR)
import numpy as np
import pandas as pd
df1 = pd.read_csv('death-penalty-cases.csv')

# word2vec requires sentences as input
from txt_utils import get_sentences
sentences = []
for doc in df1['snippet']:
    sentences += get_sentences(doc)
from random import shuffle
shuffle(sentences) # stream in sentences in random order

###
# Make document vectors from word embeddings
##

# Continuous bag-of-words representation
from gensim.models import Word2Vec
w2v = Word2Vec.load('w2v-vectors.pkl')

sentvecs = []
for sentence in sentences:
    vecs = [w2v.wv[w] for w in sentence if w in w2v.wv]
    if len(vecs)== 0:
        sentvecs.append(np.nan)
        continue
    sentvec = np.mean(vecs,axis=0)
    sentvecs.append(sentvec.reshape(1,-1))
sentvecs[0]

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(sentvecs[0],
                  sentvecs[1])[0][0]

# embedding matrix:
W = w2v.wv.syn0
W.shape

# get first principal component
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(W)
pc = pca.components_

# project onto the component and remove
Wproj = W.dot(pc.transpose()) * pc
Wnorm = W - Wproj

###
# Doc2Vec
###

from nltk import word_tokenize
docs = []

for i, row in df1.iterrows():
    docs += [word_tokenize(row['snippet'])]
shuffle(docs)

# train the model
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
doc_iterator = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
d2v = Doc2Vec(doc_iterator,
                min_count=50, 
                window=10, 
                vector_size=200,
                sample=1e-4, 
                negative=5, 
                workers=4, 
                max_vocab_size=10e4,
                #dbow_words = 1 # uncomment to get word vectors too
                )

d2v.save('d2v-vectors.pkl')

# matrix of all document vectors:
D = d2v.docvecs.vectors_docs
D.shape

D

# infer vectors for new documents
d2v.infer_vector('the judge on the court')

# get all pair-wise document similarities
pairwise_sims = cosine_similarity(D)
pairwise_sims.shape

pairwise_sims[:3,:3]

# Document clusters
from sklearn.cluster import KMeans

# create 50 clusters of similar documents
num_clusters = 50
kmw = KMeans(n_clusters=num_clusters)
kmw.fit(D)
doc_clusters = kmw.labels_.tolist()

# Documents from an example cluster
for i, doc in enumerate(docs):
    if kmw.labels_[i] == 25:
        print(doc[:5])
    if i == 1000:
        break
    
# t-SNE for visualization
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
d2v_tsne = tsne.fit_transform(D)

import ggplot as gg
vdf = pd.DataFrame(d2v_tsne,
                  columns=['x', 'y'])
vdf['cluster'] = kmw.labels_

chart = gg.ggplot( vdf, gg.aes(x='x', y='y', color='cluster') ) \
                  + gg.geom_point(size=10, alpha=.8, label='cluster') 
chart.show()



