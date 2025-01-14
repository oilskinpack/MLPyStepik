import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

res = ''
df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\bank-full.csv')


#region Исследование данных

#Смотрим данные
#  #   Column          Non-Null Count  Dtype
# ---  ------          --------------  -----
#  0   age             41188 non-null  int64
#  1   job             41188 non-null  object
#  2   marital         41188 non-null  object
#  3   education       41188 non-null  object
#  4   default         41188 non-null  object
#  5   housing         41188 non-null  object
#  6   loan            41188 non-null  object
#  7   contact         41188 non-null  object
#  8   month           41188 non-null  object
#  9   day_of_week     41188 non-null  object
#  10  duration        41188 non-null  int64
#  11  campaign        41188 non-null  int64
#  12  pdays           41188 non-null  int64
#  13  previous        41188 non-null  int64
#  14  poutcome        41188 non-null  object
#  15  emp.var.rate    41188 non-null  float64
#  16  cons.price.idx  41188 non-null  float64
#  17  cons.conf.idx   41188 non-null  float64
#  18  euribor3m       41188 non-null  float64
#  19  nr.employed     41188 non-null  float64
#  20  subscribed      41188 non-null  object
res = df.info()

#Проверяем разрезку по возрастным группам
#sns.histplot(data=df,x='age',bins=30,kde=True)

#Проверяем разрезку по возрастам и наличию/отсутствию кредита
#sns.histplot(data=df,x='age',bins=30,hue='loan')

#Количество дней с момента коммуникации с клиентом
#999 - никогда не контактировали, мы таких отсекаем
#sns.histplot(data=df[df['pdays'] != 999],x='pdays')

#Сколько шло общение если оно было - большая часть звонков шла не более 1к секунд
#sns.histplot(data=df,x='duration')

#Разбивка времени крайнего общения по способу общения - по мобильным говорили больше (cecular)
#sns.histplot(data=df,x='duration',hue='contact')

#Посмотрим работы - администраторы, синие воротнички и тех-работники самые популярные
# sns.countplot(data=df,x='job')
# plt.xticks(rotation=90)


#Посмотрим образование
# sns.countplot(data=df,x='education')
# plt.xticks(rotation=90)

#Как много людей имеют просрочки по кредиту - всего 3
# default
# no         32588
# unknown     8597
# yes            3
res = df['default'].value_counts()

#Как много людей имеют кредит - подавляющее большинство не имеет кредит
# loan
# no         33950
# yes         6248
# unknown      990
res = df['loan'].value_counts()

#

#endregion

print(res)
plt.show()