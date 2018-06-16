# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 09:50:12 2018

@author: Python Learner
"""
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model import LogisticRegression as LogR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.ensemble import VotingClassifier

train_file = r'D:\[DataSet]\1_Titanic\train.csv'
test_file = r'D:\[DataSet]\1_Titanic\test.csv'
data = pd.read_csv(train_file,index_col='PassengerId')
data_sub = pd.read_csv(test_file,index_col='PassengerId') 
data_copy = data.copy()
del data_copy['Survived']
data_all = pd.concat([data_copy,data_sub]) #数据合并

#根据Title填充Age空值
def get_title(name):
    title_search = re.search("([A-Za-z]+)\.",name)
    if title_search:
        return title_search.group(1)
    return ""
data_all['Title'] = data_all.Name.apply(get_title)
for title in data_all[data_all.Age.isnull()].Title.unique():
    title_age_mean = data_all[data_all.Title == title].Age.mean()
    data_all.loc[data_all.Age.isnull()*data_all.Title == title,'Age'] = \
    title_age_mean

#填充Fare与Embark空值
Fare_mean = data_all[data_all.Pclass == 3].Fare.mean() #计算均值
Embarked_mode = data_all.Embarked.mode()[0] #计算众数
data_all.Embarked=data_all.Embarked.fillna(Embarked_mode) #众数填充
data_all.Fare=data_all.Fare.fillna(Fare_mean) #均值填充

#年龄离散化
bins=[0,14,30,45,60,80]
cats=pd.cut(data_all.Age.as_matrix(),bins) 
data_all.Age=cats.codes

#Fare归一化
scaler=StandardScaler()
data_all.Fare=scaler.fit_transform(data_all.Fare.values.reshape(-1,1)) 

#data_all["Fare"] = data_all["Fare"].map(lambda i: np.log(i) if i > 0 else 0)


data_all['FamilySize'] = data_all.Parch + data_all.SibSp
  
data_all['Cabin_null'] = np.array(data_all.Cabin.isnull()).astype(np.int32)
data_all['Cabin_nnull'] = np.array(data_all.Cabin.notnull()).astype(np.int32)
        
Sex_dummies = pd.get_dummies(data_all.Sex, prefix= 'Sex')
Pclass_dummies = pd.get_dummies(data_all.Pclass,prefix= 'Pclass')
Embarked_dummies = pd.get_dummies(data_all.Embarked,prefix= 'Embarked')

data_all = pd.concat([data_all, Sex_dummies, Pclass_dummies,
                      Embarked_dummies], axis=1)

feature = [ 'Age','Fare','FamilySize',
           'Cabin_null','Cabin_nnull','Sex_female','Sex_male', 
           'Pclass_1','Pclass_2','Pclass_3',
           'Embarked_C','Embarked_Q','Embarked_S']

X = data_all.loc[data.index][feature] 
y = data.Survived

modelDict = {'DT':DT(),'SVC':SVC(),'GNB':GNB(),'KNN':KNN(n_neighbors=3),
             'MLP':MLP(hidden_layer_sizes=(500,)),
             'LogR':LogR(C=1.0,penalty='l1',tol=1e-6),
             'RF':RF(n_estimators=300),'GB':GB(n_estimators=500)}

for model in modelDict.keys():
    clf = modelDict.get(model)
    scores = cross_val_score(clf, X, y, cv=5)
    print (model +' accuracy: '+'%.3f' %(scores.mean()*100)+'%')
    
votingC = VotingClassifier(estimators=[('clf_GB', GB(n_estimators=500)), 
          ('clf_RF', RF(n_estimators=300)),('clf_SVC', SVC(probability=True)),
          ('clf_MLP',MLP(hidden_layer_sizes=(500,)))],voting='soft', n_jobs=4)

votingC = votingC.fit(X, y)

X_sub = data_all.loc[data_sub.index][feature]  #提取测试数据特征
y_sub = votingC.predict(X_sub) #使用模型预测数据标签
result = pd.DataFrame({'PassengerId':data_sub.index,'Survived':y_sub})
result.to_csv(r'D:\[DataSet]\1_Titanic\submission.csv', index=False) #0.78468