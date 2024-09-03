import numpy as np
import pandas as pd
from datetime import datetime

res = ''


#Создание DateTime
myYear = 1997
myMonth = 1
myDay = 1
myHour = 23
myMin = 30
mySec = 15

#1.1 Создание даты и времени  - 1997-01-01 23:30:15
mydatetime = datetime(myYear,myMonth,myDay,myHour,myMin,mySec)
res = mydatetime

#1.2 Как читать datetime из сторонних источников
#Форматы могут быть разные
#0   1990-11-03
#1   2000-01-01
#2          NaT
#dtype: datetime64[ns]
myser = pd.Series(['Nov 3, 1990','2000-01-01',None])
res = pd.to_datetime(myser,format='mixed')


#1.3 Разница между американским и европейским форматом
#Американский формат - месяц первый
#Европейский формат - день первый
euro_date = '10-12-2000'
res = pd.to_datetime(euro_date,dayfirst=True)

#1.4 Указание формата в проблемных случаях
#2000-12-22 00:00:00
style_date = '22--Dec--2000'
res = pd.to_datetime(style_date,format='%d--%b--%Y')

#1.5 Совсем кастомный формат
custom_date = '12th of Dec 2000'
res = pd.to_datetime(custom_date)

#1.6 Преобразование в csv файле
df = pd.read_csv(r'D:\Khabarov\Курс ML\03-Pandas\RetailSales_BeerWineLiquor.csv',parse_dates=[0])
res = df.info()


print(res)