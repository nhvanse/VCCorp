import os
import re 
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

vocal = {}
t0 = time()

def loadData(dataPath):
    print('Loading data...')
    data = []
    for f in os.listdir(dataPath):
        filePath = dataPath + f
        with open(filePath,encoding = 'utf-8') as file:
            content = file.read()
            label = str(f).replace('.txt','')
            data.append([label, content])
    
    return data

def preprocess(data):
    print('Preprocessing...')
    for doc in data:
        content = doc[1]
        content = re.sub('[0-9]|[.“:,;”/)()?%\"\'\\+*&-]', '', content)
        content = content.lower()
        doc[1] = content

def computeTfidf(data):
    print('Computing tf-idf...')
    corpus = []
    labels = []
    for doc in data:
        content = doc[1]
        lines = content.split('\n')
        lines = lines[0:2000]
        for line in lines:
            corpus.append(line)
            labels.append(int(doc[0]))

    with open('src/stopwords.txt', mode = 'r', encoding = 'utf-8') as stopwords_file:
        stopwords = stopwords_file.read()
        stopwords = [word for word in stopwords.split('\n') if word != '']
    
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    x = vectorizer.fit_transform(corpus)

    global vocal
    vocal = vectorizer.vocabulary_

    return x, labels

def test(cfl, x_train, y_train):
    print('Testing...')
    # tính tfidf của từng dòng văn bản 
    testData = open('classify_data/test/data.txt', encoding = 'utf-8').read()
    testData = re.sub('[0-9]|[.“:,;”/)()?%\"\'\\+*&-]', '', testData)
    testData = testData.lower()
    lines = testData.split('\n')[0:6500]

    corpus  = []
    for line in lines:
        corpus.append(line)

    with open('src/stopwords.txt', mode = 'r', encoding = 'utf-8') as stopwords_file:
        stopwords = stopwords_file.read()
        stopwords = [word for word in stopwords.split('\n') if word != '']
    
    vectorizer = TfidfVectorizer(stop_words=stopwords, vocabulary=vocal)
    x_test = vectorizer.fit_transform(corpus)
    # predict label cảu các văn bản 
    y_pre = cfl.predict(x_test)

    # tính độ chính xác 
    testLabel = open('classify_data/test/label.txt', encoding = 'utf-8').read()
    testLabel = testLabel.split('\n')
    y_test = [int(i) for i in testLabel[0:6500]]

    score = accuracy_score(y_test, y_pre)
    print('Accuracy score: {}'.format(score))
    
    # tính độ chính xác cho từng label.
    score_detail = dict.fromkeys(y_test, 0)
    for i in range(len(y_test)):
        label = y_test[i]
        if (y_pre[i] == label):
            score_detail[label] += 1
    
    for label in score_detail:
            score_detail[label] /= 500
    
    print('Detail: {}'.format(score_detail))


trainData = loadData('classify_data/train/')
preprocess(trainData)

x_train, y_train = computeTfidf(trainData)

cfl = LinearSVC()
cfl.fit(x_train, y_train)
test(cfl, x_train, y_train)
# 



from sklearn import model_selection
# from sklearn.ensemble import BaggingClassifier
# from sklearn.tree import DecisionTreeClassifier


# print('Testing...')
# # tính tfidf của từng dòng văn bản 
# testData = open('classify_data/test/data.txt', encoding = 'utf-8').read()
# testData = re.sub('[0-9]|[.“:,;”/)()?%\"\'\\+*&-]', '', testData)
# testData = testData.lower()
# lines = testData.split('\n')[0:6500]

# corpus  = []
# for line in lines:
#     corpus.append(line)

# with open('src/stopwords.txt', mode = 'r', encoding = 'utf-8') as stopwords_file:
#     stopwords = stopwords_file.read()
#     stopwords = [word for word in stopwords.split('\n') if word != '']

# vectorizer = TfidfVectorizer(stop_words=stopwords, vocabulary=vocal)
# x_test = vectorizer.fit_transform(corpus)
# testLabel = open('classify_data/test/label.txt', encoding = 'utf-8').read()
# testLabel = testLabel.split('\n')
# y_test = [int(i) for i in testLabel[0:6500]]



# # cfl = DecisionTreeClassifier(max_features=None)
# # cfl = cfl.fit(x_train, y_train)
# # y_pre = cfl.predict(x_test)
# # print(accuracy_score(y_test, y_pre)) #0.75 58s

# # cfl = MultinomialNB(alpha=0.001)
# # cfl = cfl.fit(x_train, y_train)
# # y_pre = cfl.predict(x_test)
# # print(accuracy_score(y_test, y_pre)) #0.75 58s

# cfl = LinearSVC()
# cfl = cfl.fit(x_train, y_train)
# y_pre = cfl.predict(x_test)
# print(accuracy_score(y_test, y_pre)) #0.88 23s
# print('Time: {} s'.format(time() - t0))