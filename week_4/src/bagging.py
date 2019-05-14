import os
import re 
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


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

def processTestData():
    print("Processing test data...")
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
    

    # tính độ chính xác 
    testLabel = open('classify_data/test/label.txt', encoding = 'utf-8').read()
    testLabel = testLabel.split('\n')
    y_test = [int(i) for i in testLabel[0:6500]]
    return x_test, y_test
    
def testBoost():
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import VotingClassifier
    
    global x_test, y_test, x_train, y_train

    ada_boost = AdaBoostClassifier()
    grad_boost = GradientBoostingClassifier()

    # boost_array = [ada_boost, grad_boost]
    eclf = VotingClassifier([ada_boost, grad_boost], voting='hard')
    labels = ['Ada Boost', 'Grad Boost', 'vote']
    for clf, label in zip([ada_boost, grad_boost, eclf], labels):
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        print("score: {0:.3f} [{1}]".format(score,label))


trainData = loadData('classify_data/train/')
preprocess(trainData)
x_train, y_train = computeTfidf(trainData)
x_test, y_test = processTestData()

print("Classifying....")
t1 = time()
# cfl = BaggingClassifier(base_estimator=LinearSVC(), n_estimators=10, max_samples=0.1, n_jobs=-1)
# from sklearn.ensemble import RandomForestClassifier
# # cfl =RandomForestClassifier(n_estimators=50)
# cfl.fit(x_train, y_train)
# score = cfl.score(x_test, y_test)
# print("\tacc: {}".format(score))


testBoost()
print("\tTime: {}".format(time() - t1))

print('\nTotal Time: {} s'.format(time() - t0))

