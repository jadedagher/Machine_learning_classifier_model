#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Subject: Classification algorithm

@author: Jade Dagher - 3CI
"""

from sklearn import linear_model, tree, neighbors, datasets
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import random
import numpy as np


#Load data
iris = datasets.load_iris()

#Select randomly % ech
random.seed(100)
ech_factor = 5
ech = random.sample(list(range(1, len(iris.target))), len(iris.target)/ech_factor)

#Training data
train_target = np.delete(iris.target, ech)
train_data = np.delete(iris.data, ech, axis=0)


#Testing data
test_target = iris.target[ech]
test_data = iris.data[ech]

#Difference between test_target and ech_prediction function
def diff(x,y):
    i = 0
    for a in range(len(x)):
        if  x[a] != y[a]:
            i = i+1 
    return i 

#Display detail
def details(test_target, ech_prediction): 
    print "test_target:\n", test_target
    print "ech_prediction:\n", ech_prediction
    '''Scoring'''
    print "Nb of error between test_target and ech_prediction = ", diff(ech_prediction, test_target)
    print "Prediction score in (%) = ", round(accuracy_score(test_target, ech_prediction)*100,1),"%"
    print "\n\n"
    

def LogisticR():
    #Initialisation
    clf = linear_model.LogisticRegression(C=1e5)
    clf = clf.fit(train_data, train_target)
    
    #Prediction
    ech_prediction = clf.predict(test_data)
        
    print "------------------Logistic Regression Classifier Model------------------"
    details(test_target, ech_prediction) 
    
    
def LinearD():
    #Initialisation
    clf = LinearDiscriminantAnalysis()
    clf = clf.fit(train_data, train_target)
    
    #Prediction
    ech_prediction = clf.predict(test_data)
    
    print "------------------Linear Discriminant Classifier Model------------------"
    details(test_target, ech_prediction) 


def Tree():
    #Initialisation
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_target)
    
    #Prediction
    ech_prediction = clf.predict(test_data)
    
    print "------------------Tree Classifier Model------------------"
    details(test_target, ech_prediction) 
    
    
def KNM():
    #Initialisation
    clf = neighbors.KNeighborsClassifier(2)
    clf = clf.fit(train_data, train_target) 

    #Prediction
    ech_prediction = clf.predict(test_data)
        
    print "------------------KNeighbors Classifier Model------------------"
    details(test_target, ech_prediction) 


def SVC_():
    #Initialisation
    clf = SVC()
    clf = clf.fit(train_data, train_target) 

    #Prediction
    ech_prediction = clf.predict(test_data)
        
    print "------------------SVC Classifier Model------------------"
    details(test_target, ech_prediction) 

def LSVC():
    #Initialisation
    clf = LinearSVC()
    clf = clf.fit(train_data, train_target) 

    #Prediction
    ech_prediction = clf.predict(test_data)
        
    print "------------------LinearSVC Classifier Model------------------"
    details(test_target, ech_prediction) 


#function call
LogisticR()
LinearD()
Tree()
KNM()
SVC_()
LSVC()



