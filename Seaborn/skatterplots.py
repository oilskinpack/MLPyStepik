import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv(r'D:\Khabarov\Курс ML\05-Seaborn\dm_office_sales.csv')
res = df.head()

#0.В этой части мы можем настроить figure
plt.figure(figsize=(12,4),dpi=200)


#1.График зависимости (skatterplot)
#x - колонка, которая возьмется за x (числовое значение)
#y - колонка, которая возьмется за y (числовое значение)
#hue - колонка по которой будет градиент/ цветовая гамма (числовое значение/категориальное значение)
#size - как hue только размер точек
#s - размер всех точек
#alpha - степень прозрачности
#style - колонка по которой будет выставлены разные маркеры для точек
sns.scatterplot(x='salary',y='sales',data=df,s=30,alpha = 0.3)



plt.show()
#print(res)

