import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def create_ages(mu=50,sigma=13,num_samples=100,seed=42):

    np.random.seed(seed)

    sample_ages = np.random.normal(loc = mu,scale=sigma,size=num_samples)
    sample_ages = np.round(sample_ages,decimals=0)

    return sample_ages

#Создаем нормально распределенные данные по возрасту
sample = create_ages()

#=====Визуализация данных
#Displot
# sns.distplot(sample,bins=20)
#Boxplot
# sns.boxplot(x = sample)

#Считаем интерквартильный размах
# count    100.00000
# mean      48.66000
# std       11.82039
# min       16.00000
# 25%       42.00000
# 50%       48.00000
# 75%       55.25000
# max       74.00000
# dtype: float64
ser = pd.Series(sample)
res = ser.describe()

#Собираем лимиты
IQR = 55.25 - 42.00
lover_limit = 42.00 -  1.5 * IQR
heiher_limit = 55.25 + 1.5 * IQR

#Ищем только нужные значения - 99 точек из 100
sample = sample[(sample > lover_limit) & (sample < heiher_limit)]
res = len(sample)

#Еще вариант как взять процентили - [55.5 42.5]
res = np.percentile(sample, [75,25])


#====Работа с реальными данными====
#          PID  MS SubClass MS Zoning  ...  Sale Type  Sale Condition SalePrice
# 0  526301100           20        RL  ...        WD           Normal    215000
# 1  526350040           20        RH  ...        WD           Normal    105000
# 2  526351010           20        RL  ...        WD           Normal    172000
# 3  526353030           20        RL  ...        WD           Normal    244000
# 4  527105010           60        RL  ...        WD           Normal    189900
df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\Ames_Housing_Data.csv')
res = df.head(5)
#SalePrice - наша целевая переменная

#Посмотрим корреляции
# ...
# Gr Liv Area        0.706780
# Overall Qual       0.799262
# SalePrice          1.000000
res = df.corr(numeric_only=True)['SalePrice'].sort_values()

#Проверяем отношение признака Overall Qual и SalePrice
sns.scatterplot(x='Overall Qual', y='SalePrice', data=df)
plt.clf()

#Проверяем отношение Gr Liv Area и SalePrice
sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=df)
plt.clf()

#Получаем индексы выбросов с первого графика
#             PID  MS SubClass MS Zoning  ...  Sale Type  Sale Condition SalePrice
# 1182  533350090           60        RL  ...        WD           Family    150000
# 1498  908154235           60        RL  ...        New         Partial    160000
# 2180  908154195           20        RL  ...        New         Partial    183850
# 2181  908154205           60        RL  ...        New         Partial    184750
drop_ind = df[(df['Overall Qual'] > 8) & (df['SalePrice'] < 200000)]
# res = drop_ind

#Получаем индексы выбросов со второго графика
#             PID  MS SubClass MS Zoning  ...  Sale Type  Sale Condition SalePrice
# 1498  908154235           60        RL  ...        New         Partial    160000
# 2180  908154195           20        RL  ...        New         Partial    183850
# 2181  908154205           60        RL  ...        New         Partial    184750
drop_ind2 = df[(df['Gr Liv Area'] > 4000) & (df['SalePrice'] < 200000)]
# res = drop_ind2

#Удаляем выбросы
df = df.drop(drop_ind2.index, axis=0)
res = df
#Проверяем отношение признака Overall Qual и SalePrice
sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=df)



print(res)
plt.show()
