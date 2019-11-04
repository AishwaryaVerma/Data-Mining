# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 22:45:16 2019

@author: DELL
"""


import  numpy as np
import pandas as pd
data=pd.read_csv("C:\\Users\\DELL\\Desktop\\semester_3\\data mining\\iris.csv")

def find_entropy(data):
    Class = data.keys()[-1] 
    entropy = 0
    category = data[Class].unique()
    for i in category:
        fraction = data[Class].value_counts()[i]/len(data[Class])
        entropy = entropy-fraction*np.log2(fraction)
    return entropy


def attribute_entropy(train,feature):
    Class = train.keys()[-1]   
    target_variables = train[Class].unique()  
    variables = train[feature].unique()
    entropy=0
    for j in variables:
            entropy_each_column=0
            for k in target_variables:
                num=len(train[feature][train[feature]==j][train.variety==k])
                denom=len(train[feature][train[feature]==j])
                p1=num/denom
                if(p1==0):
                    p1=1
                entropy_each_column=entropy_each_column-p1*np.log2(p1)
                
            p2=denom/len(train.variety)
            entropy+=entropy_each_column*p2
    return(entropy)    
def information_gain(data): 
    information_gain = []
    for key in data.keys()[:-1]:
        information_gain.append(find_entropy(data)-attribute_entropy(data,key))
    return data.keys()[:-1][np.argmax(information_gain)]

    
def find_gini(data):
    Class = data.keys()[-1]   
    entropy = 0
    values = data[Class].unique()
    for value in values:
        fraction = data[Class].value_counts()[value]/len(data[Class])
        entropy += fraction**2
    gini=1-entropy
    return gini


def attribute_gini(train,feature):
    Class = train.keys()[-1]   
    target_variables = train[Class].unique()  
    variables = train[feature].unique()
    gini=0
    for j in variables:
            gini_each_column=0
            for k in target_variables:
                num=len(train[feature][train[feature]==j][train.class_type==k])
                denom=len(train[feature][train[feature]==j])
                p1=num/denom
                gini_each_column+=p1**2
            gini1=1-gini_each_column    
            p2=denom/len(train.class_type)
            gini+=gini1*p2
    return(gini)    
def gini_gain(data): 
    IG = []
    for key in data.keys()[:-1]:
        IG.append(find_gini(data)-attribute_gini(data,key))
    return data.keys()[:-1][np.argmax(IG)]


def get_subtable(data,node,value):
  return data[data[node] == value].reset_index(drop=True)


def decision_tree_id3(data,tree=None):
    node = information_gain(data)
    attValue = np.unique(data[node])   
    if tree is None:
        tree={}
        tree[node] = {}

    for value in attValue:

        subtable = get_subtable(data,node,value)
        clValue,counts = np.unique(subtable['variety'],return_counts=True)

        if len(counts)==1:
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = decision_tree_id3(subtable) 

    return tree
        
        
        
            
            
def decision_tree_gini(data,tree=None):
    node = gini_gain(data)
    attValue = np.unique(data[node])   
    if tree is None:
        tree={}
        tree[node] = {}

    for value in attValue:

        subtable = get_subtable(data,node,value)
        clValue,counts = np.unique(subtable['class_type'],return_counts=True)

        if len(counts)==1:
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = decision_tree_gini(subtable) 

    return tree