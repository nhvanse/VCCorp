import os
import re 
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle
from pyvi import ViTokenizer
import numpy as np
from numpy.linalg import norm
from numpy import dot

def preprocess(text):
    text = re.sub('[0-9]|[.“:,;”/)()?%\"\'\\+*&-]', '', text)
    text = text.lower()
    return text 

def find(text):
    text = preprocess(text)
    text = ViTokenizer.tokenize(text)
    stopwords = pickle.load(open('RESTful/stopwords', 'rb'))
    vocal = pickle.load(open('RESTful/vocal', 'rb'))
    model = pickle.load(open('RESTful/model', 'rb'))

    vectorizer = TfidfVectorizer(stop_words=stopwords, vocabulary=vocal)
    corpus = [text,]
    
    x = vectorizer.fit_transform(corpus)
    y = model.predict(x)

    file  = open('data/data.txt')
    content = file.read()
    lines = content.split('\n')
    result = {}
    index = 0
    # tìm 5 kết quả
    while index < 5:
        i = np.random.random_integers(0, len(lines)-50)
        line = lines[i]
        origin_line = line
        line = preprocess(line)
        line = ViTokenizer.tokenize(line)

        corpus = [line,]
        x_find = vectorizer.fit_transform(corpus)
        y_find = model.predict(x_find)
        
        if (y_find == y ):
            result[index] = origin_line
            index += 1
        if index > 5:
            break
        
    return result   
print(find('bệnh viện'))

