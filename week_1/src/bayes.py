import os
import re
import math
from time import time

t0 = time()
# dùng để  load data vào mảng
# mảng trainData gồm 13 phần tử  tương ứng 13 class
# mảng testData gồm 2 phần tử tương dứng là data và label
# mỗi phàn tử là mảng một chiều [label, content]
def loadData(dataPath):
    data = []
    for f in os.listdir(dataPath):
        filePath = dataPath + f
        with open(filePath,encoding = 'utf-8') as file:
            content = file.read()
            label = str(f).replace('.txt','')
            data.append([label, content])
    
    return data

# tiền xử lý dữ liệu 
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

def train(data, word_dict):
    vector_dict = {}
    alpha = 0.001
    
    for doc in data:
        label = doc[0]
        content = doc[1]
        lines = content.split('\n')
        words_in_doc = dict.fromkeys(word_dict, 0)
        count_word = 0
        vector = {}

        for line in lines:
            words = [i for i in line.split(' ') if i in word_dict and i != '']
            count_word += len(words)
            for word in words:
                words_in_doc[word] += 1
        
        for word in word_dict:
            # tính xác suất một từ thuộc một label ( có làm mịn )
            p_word_label =  (alpha + words_in_doc[word]) / ( alpha * len(word_dict) + count_word)
            vector[word] = p_word_label

        vector_dict[label] = vector
    return vector_dict

# phân loại tập test    
def test(testData, word_dict, vector_dict):
    data = ''
    labels = ''
    for i in testData:
        if i[0] == 'data':
            data = i[1]
        elif i[0] == 'label':
            labels = i[1]
    
    data = re.sub('[0-9]|[.“:,;”/)()?%\"\'\\+*&-]', '', data)
    data = data.lower()

    labels = labels.split('\n')
    lines = data.split('\n')
    count_true = 0

    for i in range(len(lines)):
        line = lines[i]
        words = [word for word in line.split(' ') if word != '' and word in word_dict]
        vector = dict.fromkeys(word_dict, 0)
        for word in words:
            vector[word] += 1
        
        label_max = '1'
        max_p = 0

        for label in vector_dict:
            vector_dict_label = vector_dict[label]
            p = 1
            for word in vector:
                if vector[word] != 0:
                    # tính đại lượng tỷ lệ với xác suất vào 1 label 
                    # nhân 1000000 vào để p không quá nhỏ và tiến tới không 
                    p = 1000000 * p * (vector_dict_label[word] ** vector[word])
            
            if (p > max_p):
                max_p = p
                label_max = label
            
        
        print("Dự đoán: {} - Thực tế: {}".format(label_max, labels[i]))
        if (label_max == labels[i]):
            count_true += 1

        print("Đúng: {}. Tổng số: {}. Tỷ lệ đúng: {}".format(count_true, i+1, count_true / (i+1)))



trainData = loadData('classify_data/train/')
testData = loadData('classify_data/test/')

preprocess(trainData)
word_dict = buildDict(trainData)
vector_dict = train(trainData, word_dict)

test(testData, word_dict, vector_dict)
print("Thời gian: {}".format(time() - t0))

# ĐỘ CHÍNH XÁC: 0.81 alpha = 1
# THỜI GIAN CHẠY: 15m