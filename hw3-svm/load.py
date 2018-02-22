import os
import numpy as np
import pickle
import random
from collections import Counter
from util import dotProduct, increment
import time
import matplotlib.pyplot as plt


'''
Note:  This code is just a hint for people who are not familiar with text processing in python. There is no obligation to use this code, though you may if you like. 
'''


def folder_list(path,label):
    '''
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    '''
    filelist = os.listdir(path)
    review = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        r.append(label)
        review.append(r)
    return review

def read_data(file):
    '''
    Read each file into a list of strings. 
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on', 
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    '''
    f = open(file)
    lines = f.read().split(' ')
    symbols = '${}()[].,:;+-*/&|<>=~" '
    words = map(lambda Element: Element.translate(str.maketrans("","",symbols)).strip(), lines)
    words = list(filter(None, words))
    return words
	
###############################################
######## YOUR CODE STARTS FROM HERE. ##########
###############################################

def shuffle_data():
    '''
    pos_path is where you save positive review data.
    neg_path is where you save negative review data.
    '''
    print('done')
    pos_path = "data/pos"
    neg_path = "data/neg"
	
    pos_review = folder_list(pos_path,1)
    neg_review = folder_list(neg_path,-1)
	
    review = pos_review + neg_review
    random.shuffle(review)

    #Pickle Dump
    
    with open('data.pickle', 'wb') as f:
        pickle.dump(review, f)




def split(review):
    train = []
    test = []
    for i in range(len(review)):
        if i%4 == 0:
            test.append(review[i])
        else:
            train.append(review[i])
    return train, test


def bag_of_words(list):

    cnt = Counter()
    for word in list:
        cnt[word] += 1

    return cnt

def loss(x,y,l,w):
    loss = (l*dotProduct(w,w))/2
    m = len(x)
    for i in range(m):
        loss = loss + (max(0, 1- y[i]*dotProduct(w,x[i])))/m
    return loss

def pegasos(x, y, l):

    w = dict()
    t = 2
    temp_loss = 0
    for i in range(2):
        for j in range(len(x)):
            t = t + 1
            n = 1/(l*t)
            if y[j]*(dotProduct(w, x[j])) < 1:
                temp = x[j]
                increment(temp, (n*y[j]-1), temp)
                increment(w,-n*l,w)
                increment(w,1,temp)
            else:
                increment(w,-n*l,w)
    return w

def per_loss(x,y,w):
    cnt = 0
    total = len(y)
    for i in range(total):
        if np.sign(dotProduct(w, x[i])) != np.sign(y[i]):
            cnt = cnt + 1
    error = (cnt/total)*100
    return error

def pegasos_fast(x, y, l):

    w = dict()
    t = 2
    s = 1
    temp_loss = 0
    for i in range(2):
        for j in range(len(x)):
            t = t + 1
            n = 1/(l*t)
            s = (1-n*l)*s
            if y[j]*(dotProduct(w, x[j])) < 1:
                temp = x[j]
                increment(temp, (n*y[j]-1), temp)
                increment(w,(1/s), temp)
            #else:
                #increment(w,-n*l,w)
    
    increment(w, s-1, w)
    return w






def main():

    #loading the shuffled data
    with open('data.pickle', 'rb') as f:
        review = pickle.load(f)
    

    #Splitting into training and test sets
    train, test = split(review)

    #Splitting x and y values and getting reasy for training
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for i in train:
        y_train.append(i.pop())
        x_train.append(bag_of_words(i))

    for i in test:
        y_test.append(i.pop())
        x_test.append(bag_of_words(i))

    
 
    l = 0.01
    
    print("Pegasos fast")
    
    start_time = time.time()
    w1 = pegasos_fast(x_train, y_train, l)
    time1 = time.time() - start_time
    print("--- %s seconds ---" % (time1) )
    print(len(w1))
    

    print("Pegasos ")

    start_time = time.time()
    w = pegasos(x_train, y_train, l)
    time2 = time.time() - start_time
    print("--- %s seconds ---" % (time2) )
    print(len(w))

    #comparision
    diff = 0
    for key in w1:
        diff = w1[key] - w[key]

    print(diff)
    

    '''
    #hyperparameter tuning
    x_axis = []
    error = []
    lst = [0.01]
    for l in lst:
        w1 = pegasos_fast(x_train, y_train, l)
        e = per_loss(x_test,y_test,w1)
        print("lambda=",l ,"         ", e)
        x_axis.append(l)
        error.append(e)

    plt.plot(x_axis, error)
    #plt.show()
    '''
    











	

if __name__ == "__main__":
    main()

