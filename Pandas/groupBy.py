import numpy as np
import pandas as pd

res = ''

#Загрузка данных
df = pd.read_csv(r'D:\Khabarov\Курс ML\03-Pandas\mpg.csv')
res = df

#1.1 Группирование и применение функции к группе
#                  mpg  cylinders  displacement       weight  acceleration    origin
#70          17.689655   6.758621    281.413793  3372.793103     12.948276  1.310345
#...
#82          31.709677   4.193548    128.870968  2453.548387     16.638710  1.645161
res = df.groupby('model_year').mean(numeric_only=True)


#1.2 Получение отдельной колонки
#model_year
#70    17.689655
#71    21.250000
res = df.groupby('model_year').mean(numeric_only=True)['mpg']

#1.3 Группировка по нескольким колонкам - Мультииндекс (или иерархический индекс)
#model_year cylinders
#70         4          25.285714    107.000000  2292.571429     16.000000  2.285714
#           6          20.500000    199.000000  2710.500000     15.500000  1.000000
#           8          14.111111    367.555556  3940.055556     11.194444  1.000000
#71         4          27.461538    101.846154  2056.384615     16.961538  1.923077
#           6          18.000000    243.375000  3171.875000     14.750000  1.000000
#           8          13.428571    371.714286  4537.714286     12.214286  1.000000
res = df.groupby(['model_year','cylinders']).mean(numeric_only=True)


#1.4 Получение значения из мультииндекса - получаем обычный датафрейм с индексами
#                 mpg  displacement       weight  acceleration    origin
#cylinders
#4          25.285714    107.000000  2292.571429     16.000000  2.285714
#6          20.500000    199.000000  2710.500000     15.500000  1.000000
#8          14.111111    367.555556  3940.055556     11.194444  1.000000

var = df.groupby(['model_year','cylinders']).mean(numeric_only=True)
res = var.loc[70]

#1.5 Получение нескольких значений
res = var.loc[[70,80]]

#1.6 Получение одной строки
#mpg               20.5
#displacement     199.0
#weight          2710.5
#acceleration      15.5
#origin             1.0
#Name: (70, 6), dtype: float64
res = var.loc[(70,6)]

#1.7 Получение по внутреннему ключу - получаем 4 цилиндра для всех лет
#                  mpg  displacement       weight  acceleration    origin
#model_year
#70          25.285714    107.000000  2292.571429     16.000000  2.285714
#71          27.461538    101.846154  2056.384615     16.961538  1.923077
four_cyl = var.xs(key=4,level='cylinders')
res = four_cyl


#1.8 Получение по нескольким внутренним ключам - для цилиндров 6 и 8
#Здесь имееет смысл сначала отфильтровать, потом группировать
#                            mpg  displacement       weight  acceleration    origin
#model_year cylinders
#70         6          20.500000    199.000000  2710.500000     15.500000  1.000000
#           8          14.111111    367.555556  3940.055556     11.194444  1.000000
res = df[df['cylinders'].isin([6,8])].groupby(['model_year','cylinders']).mean(numeric_only=True)

#1.9 Поменять внешний и внутренний ключи местами
#                            mpg  displacement       weight  acceleration    origin
#cylinders model_year
#4         70          25.285714    107.000000  2292.571429     16.000000  2.285714
#6         70          20.500000    199.000000  2710.500000     15.500000  1.000000
res = var.swaplevel()


#1.10 Поменять сортировку для внешнего ключа - По убыванию
res = var.sort_index(level='model_year',ascending=False)

#2.1 Агрегация по колонкам - Можно указать словарем для каких колонок какие функции сделать
#Где мы ничего не выбрали - будет Nan
#       mpg       weight
#max   46.6          NaN
#min    9.0          NaN
#mean   NaN  2970.424623
#std    NaN   846.841774
res = df.agg( {'mpg':['max','min'], 'weight':['mean','std']} )




print(res)