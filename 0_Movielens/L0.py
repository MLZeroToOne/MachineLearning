# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 09:55:38 2018

@author: Python Learner
"""

import numpy as np
import pandas as pd

fpath = r'D:\[DataSet]\0_Movielens\\'

usercol = ['uid','sex','age','occupation','zip']
ratcol = ['uid','mid','rating','timestamp']
movcol = ['mid','title','genres']

users = pd.read_table(fpath+'users.dat',sep='::',header=None,
                      names=usercol,engine='python')
ratings = pd.read_table(fpath+'ratings.dat',sep='::',header=None,
                        names=ratcol,engine='python')
movies = pd.read_table(fpath+'movies.dat',sep='::',header=None,
                       names=movcol,engine='python')

#查看用户数据前5行
users.head()
users[:5]
users.iloc[:5]
users.loc[:4]
# 查看数据第5行
users.iloc[5]
users.loc[5]
# 按照步长查看特定行
users[1:5:2] #(start:end-1:step)
users.iloc[1:5:2] #(start:end-1:step)
users.loc[1:5:2] #(start:end:step)
# 根据条件选择特定行
users[(users.age>50)&(users.sex=='F')]
movies[movies.title.str.contains('Titanic')]
# 选定一组列
users.iloc[:,1:4]
users[['sex','age']]

#数据合并
data = pd.merge(pd.merge(users,ratings),movies)

#concat 示例
df1 = pd.DataFrame(np.arange(8).reshape(2,4))
df2 = pd.DataFrame(np.arange(12).reshape(3,4))
pd.concat([df1,df2])

#groupby示例
df = pd.DataFrame({'key1':['a','a','b','b','a'],
                   'key2':['one','two','one','two','one'],
                   'data1':np.random.randn(5),
                   'data2':np.random.randn(5)})
df.groupby(['key1','key2']).mean()


#计算电影评分
rating_mean = data.pivot_table('rating',index='title',aggfunc='mean')

#根据性别计算电影的平均分 返回DataFrame
rating_mean_sex = data.pivot_table('rating',index='title',
                               columns='sex',aggfunc='mean')
#或者使用groupby方法
rating_mean_sex = data.groupby(['sex','title']).rating.mean().unstack(level=0)

#计算电影评论数
rating_cou = data.groupby('title').size()
rating_cou.name = 'cou'

#计算评论数最多10部电影（及其得分）
hot_movie = rating_cou.sort_values(ascending=False)[:10]
pd.DataFrame(hot_movie).join(rating_mean)

#过滤评论数小于100条的电影
active_movie = rating_cou.index[rating_cou>=100]
rating_mean = rating_mean.loc[active_movie]

# 计算评分最高10部电影(及其评论数)
top_movie = rating_mean.sort_values(by='rating',ascending=False)[:10]
top_movie.join(rating_cou)

#计算女性评分最高的10部电影（及其评论数）
rating_mean_sex = rating_mean_sex.loc[active_movie]
top_movie_F = rating_mean_sex.sort_values(by='F',ascending=False)[:10]
top_movie_F.join(rating_cou)

#根据电影名称计算评分分歧
rating_std = data.groupby('title').rating.std()
rating_std = rating_std.loc[active_movie]
diff_movie = rating_std.sort_values(ascending=False)[:10]


# 分析电影时间
movies['year'] = movies.title.apply(lambda x : x[-5:-1])
# 分析电影类型
genre_iter = [set(x.split('|')) for x in movies.genres]
genre = sorted(set.union(*genre_iter))
genre_matrix = pd.DataFrame(np.zeros((len(movies),len(genre))),
                            columns=genre)
for i,gen in enumerate(movies.genres):
    genre_matrix.loc[i][gen.split('|')] = 1
movies_new = movies.join(genre_matrix)

#根据电影时间绘图 1991-2000年电影数量
movies.groupby('year').size()[-10:].plot()
#根据电影类型绘图
genre_cou = movies_new[movies_new.columns[-18:]].sum()
genre_cou.sort_values()[-10:].plot(kind='barh',fontsize=10)


#分析不同职业对于不同电影类型的喜好
data_new = pd.merge(pd.merge(users,ratings),movies_new)
homemaker = data_new.occupation==9
lawyer = data_new.occupation==11
engineer = data_new.occupation==17
tradesman = data_new.occupation==18
data_occupation = data_new[homemaker|lawyer|engineer|tradesman]
movie_genre = ['Action','Thriller','Romance','Horror','Adventure','Sci-Fi']
occup_cou = data_occupation.pivot_table(movie_genre,
                                        index='occupation',aggfunc='sum')
occup_pct = occup_cou.div(occup_cou.sum(1).astype(float),axis=0)
occup_pct.rename_axis({9:'homemaker',11:'lawyer',
                      17:'engineer', 18:'tradesman'},inplace=True)
occup_pct.plot(kind='barh',stacked=True)