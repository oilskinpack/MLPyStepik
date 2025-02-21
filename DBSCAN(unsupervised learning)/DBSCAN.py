import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN

res = ''
#region Вспомогательные функции

def display_categories(model,data):
    labels = model.fit_predict(data)
    sns.scatterplot(data=data,x='X1',y='X2',hue=labels,palette='Set2')

#endregion
#region Загрузка данных
#Данные с двумя признаками (X1 и X2)
blobs = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\cluster_blobs.csv')
#Данные с полумесяцами
moons = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\cluster_moons.csv')
#Данные по кругам
circles = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\cluster_circles.csv')
#endregion
#region Визуализация
#Визуализация каплей
# .scatterplot(data=blobs,x='X1',y='X2')
#Визуализация месяцев
# sns.scatterplot(data=moons,x='X1',y='X2')
#Визуализация кругов
# sns.scatterplot(data=circles,x='X1',y='X2')
#endregion

#region Отработка KMeans

#Капли
model = KMeans(n_clusters=3)
# display_categories(model,blobs)

#Месяцы
model = KMeans(n_clusters=2)
# display_categories(model,moons)

#Круги
model = KMeans(n_clusters=2)
# display_categories(model,circles)

#endregion
#region Отработка DBSCAN

#Капли (берем по умолчанию)
model = DBSCAN()
# ddisplay_categories(model,blobs)

#Месяцы
model = DBSCAN(eps=0.15)
# display_categories(model,moons)

#Круги
model = DBSCAN(eps=0.15)
display_categories(model,circles)

#endregion


plt.show()

