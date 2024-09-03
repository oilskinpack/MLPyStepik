import numpy as np
import pandas as pd
from datetime import datetime

res = ''

#1.1 Преобразование в csv файле
#    a   b   c   d
#0   0   1   2   3
#1   4   5   6   7
#2   8   9  10  11
#3  12  13  14  15
df = pd.read_csv(r'D:\Khabarov\Курс ML\03-Pandas\example.csv')
res = df


#1.2 Данные без названий колонок
df = pd.read_csv(r'D:\Khabarov\Курс ML\03-Pandas\example.csv',header=None)
res = df

#1.3 Сохранение файла, с сохранением колонки индекса
df.to_csv(r'D:\Khabarov\Курс ML\03-Pandas\myExample.csv',index=True)

print(res)