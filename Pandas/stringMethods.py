import numpy as np
import pandas as pd

res = ''

#Вводим данные
#0    GOOG,APPL,AMZN
#1        JPN,BAC,GS
#dtype: object
tech_finance = ['GOOG,APPL,AMZN','JPN,BAC,GS']
tickers = pd.Series(tech_finance)

#1.1 Получаем часть от составного значения текстовой колонке
#0    GOOG
#1     JPN
#dtype: object
res = tickers.str.split(',').str[0]


#1.2 Превращаем каждое значение в колонку - на выходе df
#      0     1     2
#0  GOOG  APPL  AMZN
#1   JPN   BAC    GS
res = tickers.str.split(',',expand=True)


#1.3 Пример очистки строковых данных
#0    Andrew
#1      Bobo
#2    Claire
messy_names = pd.Series(['andrew  ','bo:bo','  claire  '])
res = messy_names.str.replace(':','').str.strip().str.capitalize()


print(res)