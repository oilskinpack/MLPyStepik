import numpy as np
import pandas as pd

res = ''

#Загрузка данных
data_one = {'A': ['A0','A1','A2','A3'], 'B':['B0','B1','B2','B3']}
data_two = {'C': ['C0','C1','C2','C3'], 'D':['D0','D1','D2','D3']}
one = pd.DataFrame(data_one)
two = pd.DataFrame(data_two)

#1.1 Объединение по колонкам
#    A   B   C   D
#0  A0  B0  C0  D0
#1  A1  B1  C1  D1
#2  A2  B2  C2  D2
#3  A3  B3  C3  D3
res = pd.concat([one,two],axis=1)

#2.1 Объединение по строкам
#    A    B    C    D
#0   A0   B0  NaN  NaN
#1   A1   B1  NaN  NaN
#2   A2   B2  NaN  NaN
#3   A3   B3  NaN  NaN
#0  NaN  NaN   C0   D0
#1  NaN  NaN   C1   D1
#2  NaN  NaN   C2   D2
#3  NaN  NaN   C3   D3
res = pd.concat([one,two],axis=0)

#2.2 Как сделать одинаковые колонки и соединить дф
#    A   B
#0  A0  B0
#1  A1  B1
#2  A2  B2
#3  A3  B3
#0  C0  D0
#1  C1  D1
#2  C2  D2
#3  C3  D3
two.columns = one.columns
newDf = pd.concat([one,two],axis=0)

#2.3 После этого переназначаем индекс
#    A   B
#0  A0  B0
#1  A1  B1
#2  A2  B2
#3  A3  B3
#4  C0  D0
#5  C1  D1
#6  C2  D2
#7  C3  D3
newDf.index = range(len(newDf))
res = newDf


print(res)