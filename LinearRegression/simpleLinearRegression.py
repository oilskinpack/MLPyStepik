import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Загрузка данных - затраты на рекламу и продажи
#         TV  radio  newspaper  sales
# 0    230.1   37.8       69.2   22.1
# 1     44.5   39.3       45.1   10.4
df = pd.read_csv(r'D:\Khabarov\Курс ML\08-Linear-Regression-Models\Advertising.csv')
res = df

#Cчитаем общие продажи
df['total_spend'] = df['TV'] + df['radio'] + df['newspaper']
res = df

#Строим skatterplot чтобы понять есть ли связь между признаками
# sns.regplot(data=df,x='total_spend',y='sales')


#Приводим признак и лейбл в векторную форму
X = df['total_spend']
Y = df['sales']

#y = mx + b
#y = B1*x + B0
# [0.04868788 4.24302822]
# res = np.polyfit(X,Y,deg=1)

#Создание и проверка
#Создаем данные 1000 точек с одинаковым интервалом от 0 до 500
potential_spend = np.linspace(0,500,1000)

#Формула
predicted_sales = 0.04868788 * potential_spend + 4.24302822

# sns.scatterplot(data=df,x='total_spend',y='sales')
# plt.plot(potential_spend,predicted_sales,color = 'red')

#Вычисляем полином 3 степени
# [ 3.07615033e-07 -1.89392449e-04  8.20886302e-02  2.70495053e+00]
res = np.polyfit(X,Y,3)

#Формула для 3 степени
pot_spend = np.linspace(0,500,500)
pred_sales =   3.07615033e-07 * pot_spend       ** 3 + \
                   -1.89392449e-04 * pot_spend  ** 2 + \
                    8.20886302e-02 * pot_spend       + \
                    2.70495053e+00


sns.scatterplot(data=df,x='total_spend',y='sales')
plt.plot(pot_spend,pred_sales,color = 'red')




plt.show()
print(res)