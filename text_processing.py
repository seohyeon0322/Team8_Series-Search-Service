import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
from nltk import sent_tokenize

def get_corpus(book):
    path = './Data/novel_wordset/'
    text = []
    series_list = sorted(os.listdir(path+book))
    if (series_list[0] == '.DS_Store'):
        del series_list[0]
    for series in series_list: 
        f = open(path+book+'/'+series, 'r',encoding='latin-1')
        harry = f.read()
        text.append(harry)
    return text

def compute_tfidf(corpus):
    tfidv = TfidfVectorizer()
    tfidf = tfidv.fit_transform(corpus)
    return tfidf, tfidv

def make_csv(corpus, name):
    tfidf, tfidv = compute_tfidf(corpus)
    tfidf_array = tfidf.toarray()
    features = tfidv. get_feature_names_out()
    df = pd.DataFrame(data = tfidf_array,
    columns = features)
    df.to_csv("./Data/results/"+name+"_tfidf_matrix.csv", mode='w')

"""
To make csv file
"""
# book = 'Harry Potter'
# corpus = get_corpus(book)
# make_csv(corpus, book)


