#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:13:32 2017

@author: elliott
"""

# set this to your working directory
WORKING_DIR = '/home/elliott/Dropbox/_Ash_Teaching/2018-09 - Bocconi - Text Data and ML/code'
import os
os.chdir(WORKING_DIR)
import pandas as pd
df1 = pd.read_csv('death-penalty-cases.csv')
df1 = df1[pd.notnull(df1['author_id'])] # drop cases without an author
import numpy as np
vocab = pd.read_pickle('vocab.pkl')
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import re
from string import punctuation
translator = str.maketrans('','',punctuation) 
def fix_snippet(txt):
    a = txt.encode("ascii", errors="ignore").decode()
    a = re.sub('\W\w\W', ' ', a).lower()
    a = re.sub('\W\w\w\W', ' ', a)
    a = a.replace('&quot;', ' ').replace ('\n', ' ')
    a = a.translate(translator)
    a = a.replace('deathpenalty',' ')
    a = ' '.join(a.split())
    return a
df1['snippet'] = df1['snippet'].apply(fix_snippet)
df1['snippet'] = df1['snippet'].apply(fix_snippet)
df1.to_csv('cases-processed.csv')
df1['author_id'] = df1['author_id'].astype(str)

###
# Entity Embeddings
###

# make judge dummy variables
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
J = encoder.fit_transform(df1['author_id'])
num_judges = max(J)+1
Y = df1['citeCount'] > 0
Y2 = np.log(1+df1['citeCount'])

# set up DNN
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

model = Sequential()
model.add(Embedding(num_judges, # number of categories
                    2, # dimensions of embedding
                    input_length=1)) 
model.add(Flatten()) # needed after Embedding
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

dot = model_to_dot(model,
                   show_shapes=True,
                   show_layer_names=False)
SVG(dot.create(prog='dot', format='svg'))

model.summary()

# Visualize the Judge Vectors
import ggplot as gg
judge_cites = dict(Y.groupby(J).mean())
df2 = pd.DataFrame(J,columns=['judge']).drop_duplicates().sort_values('judge')
df2['cites'] = df2['judge'].apply(lambda x: judge_cites[x])

for i in range(5):
    if i > 0:
        model.fit(J,Y,epochs=1, validation_split=.2)
    
    judge_vectors = model.layers[0].get_weights()[0]
    df2['x'] = judge_vectors[:,0]
    df2['y'] = judge_vectors[:,1]    
    chart = gg.ggplot( df2, gg.aes(x='x', y='y', color='cites') ) \
                      + gg.geom_point(size=10, alpha=.8) 
    chart.show()


# Word Embeddings

# convert documents to sequences of word indexes
from keras.preprocessing.text import Tokenizer
num_words = 200
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(df1['snippet'])
sequences = tokenizer.texts_to_sequences(df1['snippet'])

# represent data as numrows x maxlen matrix
from keras.preprocessing.sequence import pad_sequences
maxlen = max([len(sent) for sent in sequences]) 
maxlen

X = pad_sequences(sequences, maxlen=maxlen)

# Model setup
model = Sequential()
model.add(Embedding(num_words,
                    2,
                    input_length=maxlen)) # sequence length
model.add(Flatten()) # 86*2 = 172 dims
model.add(Dense(2))
model.add(Dense(1))
model.compile(optimizer='adam',loss='binary_crossentropy')
dot = model_to_dot(model, show_shapes=True, show_layer_names=False)
SVG(dot.create(prog='dot', format='svg'))

model.summary()

# show the vectors
df3 = pd.DataFrame(list(tokenizer.word_index.items()),
                  columns=['word', 'word_index']).sort_values('word_index')[:num_words]

for i in range(5):
    if i > 0:
        model.fit(X,Y,epochs=1, validation_split=.2)

    word_vectors = model.layers[0].get_weights()[0]
    df3['x'] = word_vectors[:,0]
    df3['y'] = word_vectors[:,1]
    chart = gg.ggplot( df3, gg.aes(x='x', y='y', label='word') ) \
                      + gg.geom_text(size=10, alpha=.8, label='word') 
    chart.show()

# Word Similarity
from scipy.spatial.distance import cosine

vec_defendants = word_vectors[tokenizer.word_index['defendants']-1]
vec_convicted = word_vectors[tokenizer.word_index['convicted']-1]
vec_against = word_vectors[tokenizer.word_index['against']-1]

print(1-cosine(vec_defendants, vec_convicted))
print(1-cosine(vec_defendants, vec_against))

###
# Word2Vec in gensim
###

# word2vec requires sentences as input
from utils import get_sentences
sentences = []
for doc in df1['snippet']:
    sentences += get_sentences(doc)
from random import shuffle
shuffle(sentences) # stream in sentences in random order

# train the model
from gensim.models import Word2Vec
w2v = Word2Vec(sentences,  # list of tokenized sentences
               workers = 8, # Number of threads to run in parallel
               size=300,  # Word vector dimensionality     
               min_count =  25, # Minimum word count  
               window = 5, # Context window size      
               sample = 1e-3, # Downsample setting for frequent words
               )

# done training, so delete context vectors
w2v.init_sims(replace=True)

w2v.save('w2v-vectors.pkl')

w2v.wv['judg'] # vector for "judge"

w2v.wv.similarity('judg','juri') # similarity 

w2v.wv.most_similar('judg') # most similar words

# analogies: judge is to man as __ is to woman
w2v.wv.most_similar(positive=['judg','man'],
                 negative=['woman'])

# Word2Vec: K-Means Clusters
from sklearn.cluster import KMeans
kmw = KMeans(n_clusters=50)
kmw.fit(w2v.wv.vectors)
judge_clust = kmw.labels_[w2v.wv.vocab['judg'].index]
for i, cluster in enumerate(kmw.labels_):
    if cluster == judge_clust:
        print(w2v.wv.index2word[i])
        
###
# Pre-trained vectors
###

import spacy
en = spacy.load('en_core_web_lg')
apple = en('apple') 
apple.vector # vector for 'apple'

apple.similarity(apple)

orange = en('orange')
apple.similarity(orange)

it = spacy.load('it')
mela = it('mela')
arancia = it('arancia')
mela.similarity(arancia)

# Initializing an embedding layer with pre-trained vectors
embed_dims = len(apple.vector)
embedding_matrix = np.zeros([num_words, embed_dims])
for word, i in tokenizer.word_index.items():
    if i > num_words:
        break
    embedding_vector = en(word).vector
    embedding_matrix[i-1] = embedding_vector    
        
model = Sequential()
model.add(Embedding(num_words,
                    embed_dims,
                    weights=[embedding_matrix],
                    input_length=maxlen,
                    trainable=False)) # frozen layer
model.add(Flatten()) # 86*300 = 25800 dims
model.add(Dense(64,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# show the vectors
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)

df3 = pd.DataFrame(list(tokenizer.word_index.items()),
                  columns=['word', 'word_index']).sort_values('word_index')[:num_words]

for i in range(3):
    if i > 0:
        model.fit(X,Y,epochs=1, validation_split=.2)
    
    word_vectors = model.layers[0].get_weights()[0]
    wv_tsne = tsne.fit_transform(word_vectors)

    df3['x'] = wv_tsne[:,0]
    df3['y'] = wv_tsne[:,1]
    chart = gg.ggplot( df3, gg.aes(x='x', y='y', label='word') ) \
                      + gg.geom_text(size=10, alpha=.8, label='word') 
    chart.show()

###
# Word Mover Distance
###

import spacy
import wmd 
nlp = spacy.load('en', 
                 create_pipeline=wmd.WMD.create_spacy_pipeline)
doc1 = nlp("Politician speaks to the media in Illinois.")
doc2 = nlp("The president greets the press in Chicago.")
print(doc1.similarity(doc2))