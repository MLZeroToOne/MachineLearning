# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 10:05:02 2018

@author: Python Learner
"""

import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DT
data = pd.read_csv(r'D:\[DataSet]\1_Titanic\train.csv')

# 定义数据处理函数
def dataProcess(data):
    mapTrans={'female':0,'male':1,'S':0,'C':1,'Q':2} #属性值转换
    data.Sex=data.Sex.map(mapTrans)
    data.Embarked=data.Embarked.map(mapTrans)
    
    data.Embarked=data.Embarked.fillna(data.Embarked.mode()[0]) #使用众数填充
    data.Age=data.Age.fillna(data.Age.mean()) #均值填充缺失年龄
    data.Fare=data.Fare.fillna(data.Fare.mean()) #均值填充缺失Fare
    return data

from numpy.random import shuffle 

def genSet(data): 
    data = data.as_matrix()    
    shuffle(data)    
    #75%测试集 25%验证集
    train_set = data[:int(0.75*len(data)), :] 
    test_set = data[int(0.75*len(data)):, :] 
    
    X_train = train_set[:, 1:]
    y_train = train_set[:, 0]
    X_test = test_set[:, 1:]
    y_test = test_set[:, 0]
    
    return X_train,y_train,X_test,y_test  

data = dataProcess(data)
data = data[['Survived','Pclass','Sex','Age','Fare','Embarked']] 

X_train,y_train,X_test,y_test = genSet(data)

#1.保持法
#from sklearn.model_selection import train_test_split
#X = data.iloc[:, 1:]
#y = data.iloc[:, 0]
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.25, random_state=17)

clf = DT() #建立模型
clf.fit(X_train,y_train)  #训练模型

print('Train accuracy: '+'%.2f' %(clf.score(X_train,y_train)*100)+'%')
print('Test  accuracy: ' +'%.2f' %(clf.score(X_test,y_test)*100)+'%')


#2. 交叉验证
X = data.iloc[:, 1:] #Feature
y = data.iloc[:, 0]  #Label

from sklearn.model_selection import cross_val_score
clf = DT() #建立模型
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()

#使用KFold
from sklearn.model_selection import KFold
kf=KFold(n_splits=5, shuffle=True)
y_pred=y.copy()
for train_index,test_index in kf.split(X):
    X_train, X_test=X.iloc[train_index],X.iloc[test_index]
    y_train=y.iloc[train_index]
    clf = DT()
    clf.fit(X_train,y_train)
    y_pred.iloc[test_index]=clf.predict(X_test)
np.mean(y == y_pred)



