import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

res = ''

def display_categories(model,data):
    labels = model.fit_predict(data)
    sns.scatterplot(data=data,x='X1',y='X2',hue=labels,palette='Set1')

#region Загрузка и просмотр данных

two_blobs = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\cluster_two_blobs.csv')
two_blobs_outliers = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\cluster_two_blobs_outliers.csv')
# plt.scatter(data=two_blobs_outliers,x='X1',y='X2')

#endregion
#region Проверка

#Проверяем модель без параметров
# dbscan = DBSCAN()
# display_categories(dbscan,two_blobs_outliers)


#Процент точек-выбросов
# outliers_perc = 100 * np.sum(dbscan.labels_ == -1) / len(dbscan.labels_)
# res = outliers_perc

#Собираем информацию по кол-ву выбросов
outlier_percent = []
number_of_outliers = []

for eps in np.linspace(0.001,10,100):
    dbscan = DBSCAN(eps=eps)
    dbscan.fit(two_blobs_outliers)

    #Кол-во точек выбросов
    number_of_outliers.append(np.sum(dbscan.labels_ == -1))

    #Процент точек, классифицированных как выбросы
    percent_outliers = 100 * np.sum(dbscan.labels_ == -1) / len(dbscan.labels_)
    outlier_percent.append(percent_outliers)

sns.lineplot(x=np.linspace(0.001,10,100),y=number_of_outliers)


#endregion


print(res)
plt.show()