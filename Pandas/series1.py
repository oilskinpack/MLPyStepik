import numpy as np
import pandas as pd

res = ''

#Создание серии
myIndex = ['USA','Canada','Mexico']
myData = [1776,1867,1821]

#Создаем серию без индекса, только данные (индексация как в массиве)
#0    1776
#1    1867
#2    1821
#dtype: int64
res = pd.Series(data=myData)

#Создаем серию со своим индексом
#USA       1776
#Canada    1867
#Mexico    1821
#dtype: int64
res = pd.Series(data=myData,index=myIndex)

#Достаем значение по числовому индексу - 1776
    #res = res[0]

#Достаем значение по значению - 1776
    #res = res['USA']

#Превращаем словарь в серию
#Sam       5
#Frank    15
#Marty    21
#dtype: int6
ages = {'Sam':5, 'Frank': 15, 'Marty':21}
res = pd.Series(ages)

###Работа с series###
q1 = {'Japan':80,'China':450,'India': 200, 'USA': 250}
q2 = {'Brazil':100,'China':500,'India': 210, 'USA': 260}

sales_q1 = pd.Series(q1)
sales_q2 = pd.Series(q2)

#Получение ключей - Index(['Japan', 'China', 'India', 'USA'], dtype='object')
res = sales_q1.keys()

#Сложение данных по двум кварталам - без обработки нулевых значений
#Brazil      NaN
#China     950.0
#India     410.0
#Japan       NaN
#USA       510.0
#dtype: float64
res = sales_q1 + sales_q2

#Сложение данных по двум кварталам с обработкой
#Brazil    100.0
#China     950.0
#India     410.0
#Japan      80.0
#USA       510.0
#dtype: float64
res = sales_q1.add(sales_q2,fill_value=0)

#Получение типа данных - float64
res = res.dtype

print(res)