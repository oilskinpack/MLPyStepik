import numpy as np
import pandas as pd

res = ''

#Загрузка данных
df = pd.read_csv(r'D:\Khabarov\Курс ML\03-Pandas\movie_scores.csv')

#1.1 Нулевое значение
res = np.nan

#1.2 Проверка null через равно -  False
res = np.nan == np.nan

#1.3 Проверка null через is - True
res = np.nan is np.nan

#2.1 проверка всех значений на Nan
#   first_name  last_name    age    sex  pre_movie_score  post_movie_score
#0       False      False  False  False            False             False
#1        True       True   True   True             True              True
res = df.isnull()

#2.2 Выбор только тех строк, где есть значения
#  first_name last_name   age sex  pre_movie_score  post_movie_score
#0        Tom     Hanks  63.0   m              8.0              10.0
#3      Oprah   Winfrey  66.0   f              6.0               8.0
#4       Emma     Stone  31.0   f              7.0               9.0
res = df [df['pre_movie_score'].notnull()]


#2.3 Выбор только тех строк, где одна колонка nan, а другая не Nun
#  first_name last_name   age sex  pre_movie_score  post_movie_score
#2       Hugh   Jackman  51.0   m              NaN               NaN
res = df [ (df['pre_movie_score'].isnull() & df['first_name'].notnull()) ]

#2.4 Удаление всех строк, у которых есть хотя бы один null
res = df.dropna()

#2.5 Удаление по условию - ДОЛЖНО БЫТЬ n количество значимых признаков
#Удалятся все строки, где есть меньше 2х значимых признаков (не Null)
res = df.dropna(thresh=2)

#2.6 Удаление всех строк, где значение колонки равно null
#Удалит все строки, где last_name is null
res = df.dropna(subset=['last_name'])


#3.1 Замена пустых значений в колонке на 0
#df['pre_movie_score'] = df['pre_movie_score'].fillna(0)
res = df

#3.2 Замена пустых значений в колонке на СРЕДНЕЕ ЗНАЧЕНИЕ
df['pre_movie_score'] = df['pre_movie_score'].fillna( df['pre_movie_score'].mean() )
res = df



print(res)