import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#df = pd.read_csv(r'D:\Khabarov\Курс ML\05-Seaborn\dm_office_sales.csv')
df = pd.read_csv(r'D:\Khabarov\Курс ML\05-Seaborn\application_record.csv')
res = df.info()


#===Задание 1===
#График scatterplot пытается показать зависимость между количеством дней на работе (days employed)
#и возрастом человека (DAYS_BIRTH) для всех, кроме безработных. Обратите внимание,
#что для этого графика Вам нужно будет сначала удалить безработных людей из набора данных. 
#Также обратите внимание на значения по осям - значения были преобразованы так, чтобы они были положительными.
#Также при желании можете поменять параметры alpha и linewidth, поскольку здесь много точек накладываются друг на друга
df01 = df[df['DAYS_EMPLOYED'] < 0]
df01['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'] * (-1)
df01['DAYS_BIRTH'] = df['DAYS_BIRTH'] * (-1)
#sns.scatterplot(data=df01,x='DAYS_BIRTH',y='DAYS_EMPLOYED',alpha = 0.01,size=2)


#===Задание 2===
df02 = df01
df02['Age in Years'] = (df02['DAYS_BIRTH'] / 365).round(decimals=0).astype(int)
res =  df02['Age in Years'].unique()
#sns.displot(data=df02,x='Age in Years',bins=47,color='red',edgecolor = 'black',linewidth=1)


#===Задание 3===
df03 = df02
size = len(df)
half = int(size/2)
df03 = df.nsmallest(half,'AMT_INCOME_TOTAL')
sns.boxplot(data=df03,x='NAME_FAMILY_STATUS',y='AMT_INCOME_TOTAL',hue='FLAG_OWN_REALTY')
res = df03



plt.show()
print(res)