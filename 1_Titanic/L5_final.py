# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 09:50:12 2018

@author: Python Learner
"""
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLP
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

from sklearn.model_selection import GridSearchCV, StratifiedKFold
kfold = StratifiedKFold(n_splits=10)

clf_RF = RF()
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300,500],
              "criterion": ["gini"]}
gsRF = GridSearchCV(clf_RF,param_grid = rf_param_grid, cv=kfold, 
                    scoring="accuracy", n_jobs= 4, verbose = 1)
gsRF.fit(X,y)
rf_best = gsRF.best_estimator_


clf_SVC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}
gsSVC = GridSearchCV(clf_SVC,param_grid = svc_param_grid, cv=kfold, 
                    scoring="accuracy", n_jobs= 4, verbose = 1)
gsSVC.fit(X,y)
svm_best = gsSVC.best_estimator_

clf_GB = GB()
gb_param_grid = {'loss' : ['deviance'],
              'n_estimators' : [100,300,500],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1]}
gsGB = GridSearchCV(clf_GB,param_grid = gb_param_grid, cv=kfold, 
                    scoring="accuracy", n_jobs= 4, verbose = 1)
gsGB.fit(X,y)
gb_best = gsGB.best_estimator_

clf_MLP = MLP()
mlp_param_grid = {'hidden_layer_sizes' : [100,200,300,400,500],
              'activation' : ['relu'],
              'solver' : ['adam'],
              'learning_rate_init': [0.01, 0.001],
              'max_iter': [5000]}
gsMLP = GridSearchCV(clf_MLP,param_grid = mlp_param_grid, cv=kfold, 
                     scoring="accuracy", n_jobs= 4, verbose = 1)
gsMLP.fit(X,y)
mlp_best = gsMLP.best_estimator_

votingC = VotingClassifier(estimators=[('clf_GB', gb_best), 
          ('clf_RF', rf_best),('clf_SVC', svm_best),
          ('clf_MLP',mlp_best)],voting='soft', n_jobs=4)
votingC = votingC.fit(X, y)

X_sub = data_all.loc[data_sub.index][feature]  #提取测试数据特征
y_sub = votingC.predict(X_sub) #使用模型预测数据标签
result = pd.DataFrame({'PassengerId':data_sub.index,'Survived':y_sub})
result.to_csv(r'D:\[DataSet]\1_Titanic\submission.csv', index=False) 