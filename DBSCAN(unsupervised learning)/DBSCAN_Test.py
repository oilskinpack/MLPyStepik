import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from distributed.utils import palette
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler

res = ''
#Данные - информация по покупках людей в продуктовом магазине, где каждая строчка - человек и его покупки
df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\wholesome_customers_data.csv')


#region Просмотр и визуализация данных
#    Channel  Region  Fresh  Milk  Grocery  Frozen  Detergents_Paper  Delicassen
# 0        2       3  12669  9656     7561     214              2674        1338
# 1        2       3   7057  9810     9568    1762              3293        1776
# 2        2       3   6353  8808     7684    2405              3516        7844
# 3        1       3  13265  1196     4221    6404               507        1788
# 4        2       3  22615  5410     7198    3915              1777        5185
res = df.head()


#Зависимость между тратами на молоко и тратами на бакалею, в зависимости от channel (где были траты, отель/ресторан/кафе
# или ритейл
# sns.scatterplot(data=df,x='Milk',y='Grocery',hue='Channel',palette='Set1')


#Гистограмма трат на молоко
#multiple - stack  - накопленная гистограмма
# sns.histplot(data=df,x='Milk',hue='Channel',palette='Set1',multiple='stack')


#Кластерная карта по тратам
corr_df = df.drop(['Channel','Region'],axis=1).corr()
# sns.clustermap(data=corr_df,annot=True)


#Пэирплот для разных признаков
# sns.pairplot(data=df,hue='Region',palette='Set1')


#endregion
#region Подготовка данных и выбор eps

#Масштабирование признаков
scaler = StandardScaler()
scaled_X = scaler.fit_transform(df)

#Создание графика выбросов
outlier_percent = []
for eps in np.linspace(0.001,3,50):
    dbscan = DBSCAN(eps=eps,min_samples=2*scaled_X.shape[1])
    dbscan.fit(scaled_X)

    #Кол-во точек выбросов
    #Процент точек, классифицированных как выбросы
    percent_outliers = 100 * np.sum(dbscan.labels_ == -1) / len(dbscan.labels_)
    outlier_percent.append(percent_outliers)

#Сам график
# sns.lineplot(x=np.linspace(0.001,3,50),y=outlier_percent)


#endregion

#region Создание модели с заданным eps

#Создаем модель
dbscan = DBSCAN(eps=2,min_samples=2*scaled_X.shape[1])
dbscan.fit(scaled_X)

#Скаттер плот между Grocery и Milk по кластерам
# sns.scatterplot(data=df,x='Grocery',y='Milk',hue=dbscan.labels_,palette='Set1')


#Добавим колонку с кластерами
df['Labels'] = dbscan.labels_

#endregion
#region Анализ получившихся кластеров

#Средние значения признаков для каждого кластера
#         Channel    Region  ...  Detergents_Paper   Delicassen
# Labels                     ...
# -1         1.52  2.480000  ...      11173.560000  6707.160000
#  0         2.00  2.620155  ...       5969.581395  1498.457364
#  1         1.00  2.513986  ...        763.783217  1083.786713
cat_means = df.drop(['Channel','Region'],axis=1).groupby('Labels').mean()



#Строим тепловую карту и не берем выбросы
# sns.heatmap(data=cat_means.loc[[0,1]],annot=True,cmap='viridis')


#endregion

#region Анализ

#Приводим получившиеся данные по средним признаком к 0 - 1
#            Fresh      Milk   Grocery    Frozen  Detergents_Paper  Delicassen
# Labels
# -1      1.000000  1.000000  1.000000  1.000000          1.000000    1.000000
#  0      0.000000  0.280408  0.444551  0.000000          0.500087    0.073741
#  1      0.210196  0.000000  0.000000  0.166475          0.000000    0.000000

scaler = MinMaxScaler()
scaled_cat_data = scaler.fit_transform(cat_means)
scaled_cat_data_df = pd.DataFrame(data=scaled_cat_data,index=cat_means.index,columns=cat_means.columns)
res = scaled_cat_data_df

#Строим тепловую карту уже по масштабированному графику
sns.heatmap(data=scaled_cat_data_df,annot=True,cmap='viridis')



#endregion





print(res)
plt.show()