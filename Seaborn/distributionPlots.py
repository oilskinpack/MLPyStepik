import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv(r'D:\Khabarov\Курс ML\05-Seaborn\dm_office_sales.csv')
res = df.head()

#1.RugPlot - ковровая диаграмма
#sns.rugplot(x='salary',data=df,height=0.5)

#2.Displot - гистограмма
#bins - количество интервалов
#style - сетка ['darkgreed','whitegreed','white','dark','ticks']
#color - цвет
#engecolot - цвет обводки
#linewidth - вес обводки
#sns.set(style='whitegrid')
#sns.displot(data=df,x='salary',bins=20,color='red',edgecolor='black',linewidth =4,kde=True)

#3.KDE
#Данные
np.random.seed(42)
sample_ages = np.random.randint(0,100,200)
sample_ages = pd.DataFrame(sample_ages,columns=['age'])
#Создание
#clip - ограничение по х
#bw_adjust - детализация
#fill - заполнение фигуры внутри
sns.kdeplot(data=sample_ages,x='age',bw_adjust=2,fill=True)


plt.show()
