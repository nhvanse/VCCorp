import re
from time import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

t0 = time()

def loadData():
    print('Loading data..')
    data = []
    labels = []
    with open('classify_data/test/data.txt', mode = 'r', encoding = 'utf-8') as data_file:
        content = data_file.read()
        data = content.split('\n')[0:6500]
    
    with open('classify_data/test/label.txt', mode = 'r', encoding = 'utf-8') as label_file:
        content = label_file.read()
        labels = content.split('\n')[0:6500]
    
    return data, labels

def preprocess(data):
    print('Processing...')
    for i in range(len(data)):
        line = data[i]
        line = re.sub('[0-9]|[.“:,;”/)()?%\"\'\\+*&-]', '', line)
        line = line.lower()
        data[i] = line

def makeStopwordsFile(data):
    with open('src/stopwords.txt', mode = 'w', encoding = 'utf-8') as stw_file:
        content = ' '.join([line for line in data])
        words = [word for word in content.split(' ') if word != '']
        word_dict = dict.fromkeys(words, 0)
        for word in words:
            word_dict[word] += 1
        
        stopwords = []
        for word in word_dict:
            if (word_dict[word] > 5000 or word_dict[word] == 1 ):
                stopwords.append(word)

        for word in stopwords:
            stw_file.write(word + '\n')

def computeTfidf(data):
    print('Computing TF-idf...')
    with open('src/stopwords.txt', mode = 'r', encoding = 'utf-8') as stopwords_file:
        stopwords = stopwords_file.read()
        stopwords = [word for word in stopwords.split('\n') if word != '']
    
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    x = vectorizer.fit_transform(data)

    return x

# def vectorizer(data):
#     with open('src/stopwords.txt', mode = 'r', encoding = 'utf-8') as stopwords_file:
#         stopwords = stopwords_file.read()
#         stopwords = [word for word in stopwords.split('\n') if word != '']
    
#     vectorizer = CountVectorizer(stop_words=stopwords)
#     x = vectorizer.fit_transform(data)

#     return x


data, labels = loadData()
preprocess(data)
x = computeTfidf(data)

kmeans = KMeans(n_clusters = 13, init='k-means++')
y_km = kmeans.fit(x)

# print(y_km.labels_[1500:2000])

# print(len(y_km.labels_))

kq = dict.fromkeys(y_km.labels_, 0)
for label in y_km.labels_:
    kq[label] +=1
print(kq)


print('Time: {}'.format(time() - t0))

