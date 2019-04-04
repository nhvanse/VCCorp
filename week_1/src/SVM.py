import os
import re
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score

def loadData(dataPath):
    data = []
    for f in os.listdir(dataPath):
        filePath = dataPath + f
        with open(filePath,encoding = 'utf-8') as file:
            content = file.read()
            label = str(f).replace('.txt','')
            data.append([label, content])
    
    return data
def preprocess(data):
    for doc in data:
        content = doc[1]
        content = re.sub('[0-9]|[.“:,;”/)()?%\"\'\\+*&-]', '', content)
        content = content.lower()
        doc[1] = content

# tạo từ điển
def buildDict(data):
    word_dict = {}
    for doc in data:
        content = doc[1]
        lines = content.split('\n')
        for line in lines:
            words = [i for i in line.split(' ') if i != '']
            for word in words:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
    # loại bỏ các stopword theo số từ 
    stopwords = []
    for word in word_dict:
        if (word_dict[word] > 5000  or (word_dict[word] == 1 and '_' not in word)):
            stopwords.append(word)
    
    for word in stopwords:
        word_dict.pop(word)

    return word_dict

def buildVector (data, word_dict):
    x_vector = []
    y_vector = []
    for doc in data:
        content = doc[1]
        lines = content.split('\n')
        
        for line in lines:
            words_in_line = dict.fromkeys(word_dict, 0)
            words = [word for word in line.split(' ') if word != '' and word in word_dict]
            
            for word in words:
                words_in_line[word] += 1
                
            vector = words_in_line.values()
            vector = list(vector)
            
            x_vector.append(vector)
            y_vector.append(int(doc[0]))
            
    return x_vector, y_vector

def test(cfl, testData, word_dict, x_vector, y_vector):
    data = ''
    labels = ''
    
    x_test = []
    y_test = []
    
    for i in testData:
        if i[0] == 'data':
            data = i[1]
        elif i[0] == 'label':
            labels = i[1]
            
    data = re.sub('[0-9]|[.“:,;”/)()?%\"\'\\+*&-]', '', data)
    data = data.lower()
    
    labels = labels.split('\n')
    y_test = [int(label) for i in labels]
    lines = data.split('\n')
    words_in_line = dict.fromkeys(word_dict, 0)
    
    for i in range(len(lines)):
        line = lines[i]
        label = labels[i]
        words = [word for word in line.split(' ') if word in word_dict and word != '']
        words_in_line = dict.fromkeys(word_dict, 0)
        
        for word in words:
            words_in_line[word] += 1
            
        vector = words_in_line.values()
        x_test.append(vector)
    
    cfl.predict(x_test)

trainData = loadData('classify_data/train/')
testData = loadData('classify_data/test/')

preprocess(trainData)
word_dict = buildDict(trainData)

x_train, y_train = buildVector(trainData, word_dict)

cfl = svm.LinearSVC(C=1.0)
cfl.fit(x_train, y_train)