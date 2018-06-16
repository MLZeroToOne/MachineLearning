# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 10:05:02 2018

@author: Python Learner
"""

import pandas as pd 
data = pd.read_csv(r'D:\[DataSet]\1_Titanic\train.csv')

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #设置中文显示
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure()
fig.set(alpha=0.65) # 设置图像透明度
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)

cou_Sex = pd.crosstab(data.Sex,data.Survived) 
#或者用counts_Sex = data.groupby(['Sex','Survived']).size().unstack
cou_Sex.rename_axis({0:'未生还',1:'生还'},axis=1,inplace=True)
cou_Sex.rename_axis({'female':'F','male':'M'},inplace=True)
pct_Sex = cou_Sex.div(cou_Sex.sum(1).astype(float),axis=0) #归一化
pct_Sex.plot(kind='bar',stacked=True,title=u'不同性别的生还情况',ax=ax1)

cou_Pclass = pd.crosstab(data.Pclass,data.Survived)
cou_Pclass.rename_axis({0:'未生还',1:'生还'},axis=1,inplace=True)
pct_Pclass = cou_Pclass.div(cou_Pclass.sum(1).astype(float),axis=0)
pct_Pclass.plot(kind='bar',stacked=True,title=u'不同等级的生还情况',
                ax=ax2,sharey=ax1)

cou_Embarked = pd.crosstab(data.Embarked,data.Survived) 
cou_Embarked.rename_axis({0:'未生还',1:'生还'},axis=1,inplace=True)
pct_Embarked = cou_Embarked.div(cou_Embarked.sum(1).astype(float),axis=0)
pct_Embarked.plot(kind='bar',stacked=True,title=u'不同登录点生还情况',
                  ax=ax3,sharey=ax1)

plt.scatter(data.Survived,data.Age)
plt.scatter(data.Survived,data.Fare)

fig = plt.figure()
fig.set(alpha=0.65) # 设置图像透明度
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)

bins=[0,14,30,45,60,80]
cats=pd.cut(data.Age.as_matrix(),bins) #Age离散化
data.Age=cats.codes

cou_Age = pd.crosstab(data.Age,data.Survived)
cou_Age.rename_axis({0:'未生还',1:'生还'},axis=1,inplace=True)
pct_Age = cou_Age.div(cou_Age.sum(1).astype(float),axis=0)
pct_Age.plot(kind='bar',stacked=True,title=u'不同年龄的生还情况',ax=ax1)

bins=[0,15,30,45,60,300]
cats=pd.cut(data.Fare.as_matrix(),bins) #Fare离散化
data.Fare=cats.codes

cou_Fare = pd.crosstab(data.Fare,data.Survived)
cou_Fare.rename_axis({0:'未生还',1:'生还'},axis=1,inplace=True)
pct_Fare = cou_Fare.div(cou_Fare.sum(1).astype(float),axis=0)
pct_Fare.plot(kind='bar',stacked=True,title=u'不同票价的生还情况',
              ax=ax2,sharey=ax1)

def dataProcess(data):
    mapTrans={'female':0,'male':1,'S':0,'C':1,'Q':2} #属性值转换
    data.Sex=data.Sex.map(mapTrans)
    data.Embarked=data.Embarked.map(mapTrans)
    
    data.Embarked=data.Embarked.fillna(data.Embarked.mode()[0]) #使用众数填充
    data.Age=data.Age.fillna(data.Age.mean()) #均值填充缺失年龄
    data.Fare=data.Fare.fillna(data.Fare.mean()) #均值填充缺失Fare
    return data

data = dataProcess(data)
data.iloc[:,1:].corr()['Survived']

import seaborn as sns #导入seaborn绘图库
sns.set(style='white', context='notebook', palette='deep')
sns.heatmap(data.iloc[:,1:].corr(),annot=True, fmt = ".2f", 
            cmap = "coolwarm")

from sklearn.feature_selection import SelectKBest ,chi2
feature = ['Pclass', 'Sex', 'Age', 'SibSp','Parch',  'Fare', 'Embarked']
selector = SelectKBest(chi2,k=5)
selector.fit(data[feature],data.Survived)
pd.concat([pd.Series(selector.scores_),pd.Series(feature)],axis=1)





