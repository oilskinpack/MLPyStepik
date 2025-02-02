import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

res = ''
#Данные по моделям автомобилей
df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\cluster_mpg.csv')

#region Просмотр и подготовка данных

#Делаем дамми переменные
df_w_dummies = pd.get_dummies(df.drop(['name'],axis=1))
res = df_w_dummies

#Масштабируем
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_w_dummies)
scaled_df = pd.DataFrame(scaled_data,columns = df_w_dummies.columns)

#Тепловая карта
# sns.heatmap(scaled_df,cmap='viridis')

#Кластер карта
# sns.clustermap(scaled_df,cmap='viridis',col_cluster=False)


#endregion
#region Создание модели через количество кластеров

#Создание эстимейтора
model = AgglomerativeClustering(n_clusters=4)

#Получение значения кластеров
# [1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 0 0 0 3 2 2 2 2 2 0 1 1 1 1 3 0 3 0 0 0 0 0
#  1 1 1 1 1 1 1 0 0 0 0 0 2 2 2 3 3 2 0 3 0 2 0 0 1 1 1 1 1 1 1 1 1 3 1 1 1
cluster_labels = model.fit_predict(scaled_df)

#Просмотр зависимости веса машины от потребления топлива
# sns.scatterplot(data=df,x='mpg',y='weight',hue=cluster_labels,palette='viridis')


#Просмотр зависимости лошадиных сил машины от потребления топлива
# sns.scatterplot(data=df,x='mpg',y='horsepower',hue=cluster_labels,palette='viridis')



#endregion

#region Создание модели через threshold

#Максимально возможное евклидово расстояние (по формуле), считается из кол-ва признаков
max_dist = np.sqrt(len(df.columns))
res = max_dist

#Создание эстимейтора
model = AgglomerativeClustering(n_clusters=None,distance_threshold=0,compute_distances=True)
cluster_labels = model.fit_predict(scaled_df)

#endregion
#region Создание и анализ дендрограммы

#Данные дендрограммы в табличном виде, где каждая строка постепенно объединяет данные
# [[6.70000000e+01 6.80000000e+01 4.01977033e-02]
#  [2.32000000e+02 2.34000000e+02 4.12867038e-02]
#  [6.30000000e+01 7.40000000e+01 4.31686056e-02]
#  ...
#  [7.78000000e+02 7.79000000e+02 1.07410589e+01]
#  [7.75000000e+02 7.77000000e+02 1.21893455e+01]
#  [7.80000000e+02 7.81000000e+02 1.93074925e+01]]
from scipy.cluster import hierarchy
linkage_matrix = hierarchy.linkage(scaled_df.values,method='ward')
res = linkage_matrix

#Дендрограмма
#truncate_mode - по какому принципу определяется глубина дендрограммы
dendro = dendrogram(linkage_matrix,truncate_mode='lastp',p=10)

#endregion
#region Оценка расстояния

#Получение индекса машин с мин и макс mpg - 28 и 320
car_a_id = scaled_df['mpg'].idxmax()
car_b_id = scaled_df['mpg'].idxmin()

#Получение признаков этих машин
car_a = scaled_df.iloc[car_a_id]
car_b = scaled_df.iloc[car_b_id]

#Евклидово Расстояние между этими точками - 2.3852929970374714
distance = np.linalg.norm(car_a - car_b)
res = distance


#endregion


print(res)
plt.show()