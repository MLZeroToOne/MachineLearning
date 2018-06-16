# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 22:58:06 2018

@author: plPython Learner
"""

import pandas as pd
from sklearn.model_selection import cross_val_score #交叉验证
from sklearn.tree import DecisionTreeClassifier as DT#决策树
from sklearn.naive_bayes import GaussianNB as GNB#Naive Bayes
from sklearn.svm import SVC#SVM
from sklearn.neighbors import KNeighborsClassifier as KNN#KNN
from sklearn.neural_network import MLPClassifier as MLP#神经网络
from sklearn.ensemble import RandomForestClassifier as RF#随机森林
from sklearn.ensemble import GradientBoostingClassifier as GB#梯度提升
from sklearn.linear_model import LogisticRegression as LogR#逻辑回归

# 定义数据处理函数
def dataProcess(data):
    mapTrans={'female':0,'male':1,'S':0,'C':1,'Q':2} #属性值转换
    data.Sex=data.Sex.map(mapTrans)
    data.Embarked=data.Embarked.map(mapTrans)
    
    data.Embarked=data.Embarked.fillna(data.Embarked.mode()[0])#使用众数填充
    data.Age=data.Age.fillna(data.Age.mean()) #均值填充缺失年龄
    data.Fare=data.Fare.fillna(data.Fare.mean()) #均值填充缺失Fare
    
    return data

data = pd.read_csv(r'D:\[DataSet]\1_Titanic\train.csv')
data = dataProcess(data)
feature = ['Pclass','Sex','Age','Fare','Embarked']
X = data[feature] #Feature
y = data.Survived  #Label

modelDict = {'DT':DT(),'SVC':SVC(),'GNB':GNB(),'KNN':KNN(n_neighbors=3),
             'MLP':MLP(hidden_layer_sizes=(500,)),
             'LogR':LogR(C=1.0,penalty='l1',tol=1e-6),
             'RF':RF(),'GB':GB(n_estimators=500)}

for model in modelDict.keys():
    clf = modelDict.get(model)
    scores = cross_val_score(clf, X, y, cv=5)
    print (model +' accuracy: '+'%.3f'%(scores.mean()*100)+'%')
    
clf_GB = GB(n_estimators=500)
clf_GB.fit(X,y) #模型训练
data_sub = pd.read_csv(r'D:\[DataSet]\1_Titanic\test.csv') #加载测试数据
data_sub = dataProcess(data_sub)       #处理测试数据
X_sub = data_sub[feature]  #提取测试数据特征
y_sub = clf_GB.predict(X_sub) #使用模型预测
result = pd.DataFrame({'PassengerId':data_sub['PassengerId'].as_matrix(), 
                       'Survived':y_sub}) #形成要求格式
result.to_csv(r'D:\[DataSet]\1_Titanic\submission.csv', index=False) #输出至文件