import numpy as np
import pandas as pd

res = ''

#Загрузка данных
df = pd.read_csv(r'D:\Khabarov\Курс ML\03-Pandas\tips.csv')

#1.1 Фильтрация по колонке (1 шаг) - выдает серию с результатом проверки по каждой строчке
#0      False
#1      False
#2      False
#3      False
#4      False
bool_series = df['total_bill'] > 40

#1.2 Фильтрация по колонке (2 шаг) - получаем только нужные строки, передав булевую маску на наш фрейм
res = df[bool_series]

#1.3 Фильтрация по колоке - в один шаг
res = df[df['sex'] == 'Male']

#2.1 Фильтрация по двум условиям
res = df[ (df['total_bill'] > 30) & (df['sex'] == 'Male') ] 

#2.2 Фильтрация по списку значений
options = ['Sat','Sun']
res = df[ df['day'].isin(options) ]


print(res)