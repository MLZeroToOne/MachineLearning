# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 21:11:52 2018

@author: Python Learner
"""

import pandas as pd 
data = pd.read_csv(r'D:\[DataSet]\1_Titanic\train.csv')

data[:5] #查看前5行数据 或者head(5)
data.iloc[0] #查看某一行数据
data.Pclass.unique() #查看某个字段的取值
data.Survived.value_counts() #查看字段取值的统计值
data.info() 

# 定义数据处理函数
def dataProcess(data):
    mapTrans={'female':0,'male':1,'S':0,'C':1,'Q':2} #属性值转换
    data.Sex=data.Sex.map(mapTrans)
    data.Embarked=data.Embarked.map(mapTrans)
    
    data.Embarked=data.Embarked.fillna(data.Embarked.mode()[0])#使用众数填充
    data.Age=data.Age.fillna(data.Age.mean()) #均值填充缺失年龄
    data.Fare=data.Fare.fillna(data.Fare.mean()) #均值填充缺失Fare
    return data

data = dataProcess(data)
feature = ['Pclass','Sex','Age','Fare','Embarked']
X = data[feature] #选择特征
y = data.Survived #标签

from sklearn.tree import DecisionTreeClassifier as DT
clf = DT() #建立模型
clf.fit(X,y)  #训练模型

print('%.3f' %(clf.score(X,y))) #准确率

from sklearn import metrics
metrics.confusion_matrix(y, clf.predict(X)) #混淆矩阵

data_sub = pd.read_csv(r'D:\[DataSet]\1_Titanic\test.csv') #加载测试数据
data_sub = dataProcess(data_sub)       #处理测试数据
X_sub = data_sub[feature]  #提取测试数据特征
y_sub = clf.predict(X_sub) #使用模型预测数据标签
result = pd.DataFrame({'PassengerId':data_sub['PassengerId'].as_matrix(), 
                       'Survived':y_sub}) #形成要求格式
result.to_csv(r'D:\[DataSet]\1_Titanic\submission.csv', index=False) #输出至文件






