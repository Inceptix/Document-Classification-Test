# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 12:30:36 2018

@author: Inceptix
"""

import nltk
import pandas as pd

import numpy as np
import random 
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify.scikitlearn import SklearnClassifier


#Vote between three classifiers to obtain best result
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

#create a two dimensional list from the sample file
data = pd.read_csv('shuffled-full-set-hashed.csv')
array = data.values

#populate all_cat and all_words lists respectively 
#with the categories and words of each document
all_cat = []
all_words =[]
i=0
j=0

#iterate through the original list and populate the new lists with the values
while len(array) > i:
    all_cat.append(array[i][0])
    i +=1
    
while len(array) > j:
    all_words.append(array[j][1])
    j +=1
    

#create lists docs and tokenized_words 
#tokenized_words list creates a list with lists of each document while separating the words
#essentailly the list docs treats each line in the csv file as a separate document
#which makes the manipulation of the data easier

docs = []
tokenized_words = []

#create long string of all the words
all_words_str = ''.join(map(str,all_words))
i = 0

#populate lists through iteration
while len(all_words) > i:
    tokenized_words.append(nltk.word_tokenize(all_words[i]))
    i += 1
    

#populate lists through iteration
i = 0
while len(tokenized_words) > i:
    docs.append((tokenized_words[i],all_cat[i]))
    i += 1
    

#shuffle the lines while preserving order
random.shuffle(docs)

#tokenize the string of all words to use with the Frequency Distribution method
#FreqDist help find the most important words in the documents to use for classification
tokenized_str = nltk.word_tokenize(all_words_str)
tokenized_dist = nltk.FreqDist(tokenized_str)

#make list of the most important features
word_features = list(tokenized_dist.keys())[:5000]

#find features in each "document" included in csv file
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        
    return features



featuresets = [(find_features(doc), category) for (doc, category) in docs]


#train with the first 3250 features found
training_set = featuresets[:3250]
#test against the reamining 1750 features
testing_set = featuresets[3250:]


#use the defined training set and use the Naive Bayes algorithm to train a classifier
#to later use to identify other documents.
classifier = nltk.NaiveBayesClassifier.train(training_set)
#test the classifier against the testing set of known documents to find out the accuracy of the classifier
#uncomment the next line to see the accuracy of the algorithm
#takes abour 30-45 to go through all the data on my computer
#print("Naive Bayes accuracy%:",(nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(30)

#train other algorithms as well and observe any significant positive or negative changes.
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
#print("Multinomial Naive Bayes Algo accuracy %:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)


BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
#print("BernoulliNB Naive Bayes Algo accuracy %:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

#use the voting system between the different classifiers
voted_classifier = VoteClassifier(classifier,MNB_classifier,BernoulliNB_classifier)

#use this method in the other Python script to test the document classifier
def doc_class(text):
    feats = find_features(text)
    
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)



    




