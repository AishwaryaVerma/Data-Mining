# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:51:11 2019

@author: DELL
"""

import  numpy as np
import pandas as pd
data=pd.read_csv("C:\\Users\\DELL\\Desktop\\semester_3\\data mining\\zoo-animal-classification\\zoo.csv")
data=data.drop("animal_name",axis=1)


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
                num=len(train[feature][train[feature]==j][train.class_type==k])
                denom=len(train[feature][train[feature]==j])
                p1=num/denom
                if(p1==0):
                    p1=1
                entropy_each_column=entropy_each_column-p1*np.log2(p1)
                
            p2=denom/len(train.class_type)
            entropy+=entropy_each_column*p2
    return(entropy)    
def information_gain(data): 
    information_gain = []
    for key in data.keys()[:-1]:
        information_gain.append(find_entropy(data)-attribute_entropy(data,key))
    return data.keys()[:-1][np.argmax(information_gain)]

def decision_tree(train,tree=None):
    node=information_gain(train)
    attb_value=data[node].unique()
    if tree is None:
        tree={}
    tree[node]={}
    for i in attb_value:
       d=data[data[node]==i].reset_index(drop=True)
       clValue,counts = np.unique(d['class_type'],return_counts=True)
       if len(counts)==1:
            tree[node][i] = clValue[0]
       else:
            tree[node][i] = decision_tree(d) 

    return tree
        
       
    
        
        
        
            
            


    

