import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

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



#endregion
#region Подготовка данных и алгоритма

#Создание dummy переменных
X = pd.get_dummies(df)

#Масштабирование признаков
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

#Параметры алгоритма
#max_iter - максимальное кол-во итераций (по деф 300, обычно хватает)
#n_clusters - количество кластеров
model = KMeans(n_clusters=2)



#endregion
#region Кластеризация

#Получаем значения кластеризации
# [1 1 1 ... 0 0 0]
cluster_labels = model.fit_predict(scaled_X)
# res = cluster_labels

#Добавляем результаты к признакам
#        age  duration  campaign  ...  subscribed_no  subscribed_yes  Clusters
# 0       56       261         1  ...           True           False         1
# 1       57       149         1  ...           True           False         1
# 2       37       226         1  ...           True           False         1
X['Cluster'] = cluster_labels
res = X

#endregion
#region Интерпретация

# previous               -0.478467
# poutcome_failure       -0.464295
# contact_cellular       -0.410476
# month_apr              -0.357923
# subscribed_yes         -0.294610
#                           ...
# poutcome_nonexistent    0.544377
# cons.price.idx          0.679372
# nr.employed             0.886155
# emp.var.rate            0.932612
# euribor3m               0.959297
res = X.corr()['Cluster'].iloc[:-1].sort_values()

#endregion

#region Поиск оптимального к
#region Сбор значений к

# ssd = []
# for k in range(2,10):
#     model = KMeans(n_clusters=k)
#     model.fit(scaled_X)
#     ssd.append(model.inertia_) #Сумма квадратов расстояний от точек до центров кластеров

#endregion
#region Сбор значений силуэтов

# silhouette = []
# for k in range(2,10):
#     model = KMeans(n_clusters=k)
#     model.fit(scaled_X)
#     silhouette.append(silhouette_score(scaled_X,model.labels_)) #Силуэты

#endregion
#region Визуализация

# plt.plot(range(2,10),ssd,'o--')

#endregion
#region Визуализация через силуэты

#Создаем модель
k = 2
kmeans = KMeans(n_clusters=k)
cluster_labels = kmeans.fit_predict(scaled_X)

#Считаем силуэт score
avg_silhouette_score = silhouette_score(scaled_X, cluster_labels)
sample_silhouette_values = silhouette_samples(scaled_X, cluster_labels)

#Создаем сабплот
fig, ax = plt.subplots(figsize=(10, 6))

y_lower = 10
ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == k]
ith_cluster_silhouette_values.sort()

cluster_size = ith_cluster_silhouette_values.shape[0]
y_upper = y_lower + cluster_size

ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values, alpha=0.7)
ax.text(-0.05, y_lower + 0.5 * cluster_size, str(k))

y_lower = 10
for i in range(k):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()

    cluster_size = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + cluster_size

    ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * cluster_size, str(i))

    y_lower = y_upper + 10

    ax.set_title(f"Silhouette Plot (n_clusters={k}, Avg Silhouette Score = {avg_silhouette_score:.2f})")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Label")
    ax.axvline(x=avg_silhouette_score, color="red", linestyle="--", label="Avg Silhouette Score")
    ax.legend()
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])

#endregion
#endregion

print(res)
plt.show()