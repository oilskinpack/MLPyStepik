import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import plotly.express as px
# import chart_studio.plotly as py

res = ''
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.max_colwidth', 1500)
# pd.set_option('display.max_rows', 20)

#region Функции

def show_silhouette_plt(k,model,scaled_X):
    cluster_labels = model.fit_predict(scaled_X)

    # Считаем силуэт score
    avg_silhouette_score = silhouette_score(scaled_X, cluster_labels)
    sample_silhouette_values = silhouette_samples(scaled_X, cluster_labels)

    # Создаем сабплот
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

#region Загрузка и просмотр данных

#Загрузка
df = pd.read_csv(r'D:\Khabarov\Репозиторий\MachineLearningCourse\DATA\CIA_Country_Facts.csv')

#Просмотр данных
#           Country                               Region  ...  Industry  Service
# 0     Afghanistan        ASIA (EX. NEAR EAST)           ...     0.240    0.380
# 1         Albania  EASTERN EUROPE                       ...     0.188    0.579
# 2         Algeria  NORTHERN AFRICA                      ...     0.600    0.298
# 3  American Samoa  OCEANIA                              ...       NaN      NaN
# 4         Andorra  WESTERN EUROPE                       ...       NaN      NaN
res = df.head()

#Типы данных
#  #   Column                              Non-Null Count  Dtype
# ---  ------                              --------------  -----
#  0   Country                             227 non-null    object
#  1   Region                              227 non-null    object
#  2   Population                          227 non-null    int64
#  3   Area (sq. mi.)                      227 non-null    int64
#  4   Pop. Density (per sq. mi.)          227 non-null    float64
#  5   Coastline (coast/area ratio)        227 non-null    float64
#  6   Net migration                       224 non-null    float64
#  7   Infant mortality (per 1000 births)  224 non-null    float64
#  8   GDP ($ per capita)                  226 non-null    float64
#  9   Literacy (%)                        209 non-null    float64
#  10  Phones (per 1000)                   223 non-null    float64
#  11  Arable (%)                          225 non-null    float64
#  12  Crops (%)                           225 non-null    float64
#  13  Other (%)                           225 non-null    float64
#  14  Climate                             205 non-null    float64
#  15  Birthrate                           224 non-null    float64
#  16  Deathrate                           223 non-null    float64
#  17  Agriculture                         212 non-null    float64
#  18  Industry                            211 non-null    float64
#  19  Service                             212 non-null    float64
res = df.info()


#Корреляции
res = df.describe()


#endregion
#region Визуализация данных

#Настройка
figsize = (10,6)
plot = plt.figure(figsize=figsize)
plt.subplots_adjust(bottom=0.3)
plt.xticks(rotation=90,fontsize=6)

#Гистограмма по кол-ву населения (уберем китай и индию, они значительно больше)
# sns.histplot(data=df[df['Population'] < 500000000],x='Population')


#Распределение по доходам в разных регионах
# ax = sns.barplot(data=df,y='GDP ($ per capita)',x='Region')


#Смотрим количество телефонов на 1000 человек в зависимости от ВВП
# sns.scatterplot(data=df,x='GDP ($ per capita)',y='Phones (per 1000)',hue='Region')


#Ищем выбросы с предыдущего графика - страну с большим кол-вом телефонов(но небольшим ВВП)
#А также страну с высоким ВВП, но низким кол-вом телефонов
#     Country                               Region  ...  Industry  Service
# 138  Monaco  WESTERN EUROPE                       ...       NaN      NaN
res = df[df['Phones (per 1000)'] > 900]

#Страна с самым большим ВВП на душу населения
#         Country                               Region  ...  Industry  Service
# 121  Luxembourg  WESTERN EUROPE                       ...      0.13     0.86
res = df[df['GDP ($ per capita)']>50000]

#Смотрим зависимость ВВП на душу от грамотности
# sns.scatterplot(data=df,x='GDP ($ per capita)',y='Literacy (%)',hue='Region')


#Тепловая карта корреляции между колонками
# sns.heatmap(df.corr(numeric_only=True))



#Иерархическая кластеризация
# sns.clustermap(df.corr(numeric_only=True))


#endregion
#region Заполнение пустых значений

#Смотрим нулевые значения
# Climate                               22
# Literacy (%)                          18
# Industry                              16
# Service                               15
# Agriculture                           15
# Deathrate                              4
# Phones (per 1000)                      4
# Infant mortality (per 1000 births)     3
# Net migration                          3
# Birthrate                              3
# Arable (%)                             2
# Crops (%)                              2
# Other (%)                              2
# GDP ($ per capita)                     1
# Region                                 0
# Coastline (coast/area ratio)           0
# Pop. Density (per sq. mi.)             0
# Area (sq. mi.)                         0
# Population                             0
# Country                                0
res = df.isnull().sum().sort_values(ascending=False)


#Смотрим какие страны имеют Agriculture == nan и что между ними общего
#Судя по всему это островные государства без развитой агрокультуры
# 3            American Samoa
# 4                   Andorra
# 78                Gibraltar
# 80                Greenland
# 83                     Guam
# 134                 Mayotte
# 140              Montserrat
# 144                   Nauru
# 153      N. Mariana Islands
# 171            Saint Helena
# 174    St Pierre & Miquelon
# 177              San Marino
# 208       Turks & Caicos Is
# 221       Wallis and Futuna
# 223          Western Sahara
res = df[df['Agriculture'].isnull()]['Country']

#Приравняем у них значение к 0
df[df['Agriculture'].isnull()] = df[df['Agriculture'].isnull()].fillna(0)
res = df[df['Agriculture'].isnull()]['Country']


#Смотрим страны с пустыми значениями климата
# 5             Angola
# 36            Canada
# 50           Croatia
# 66     Faroe Islands
# 101            Italy
# 115          Lebanon
# 118            Libya
# 120        Lithuania
# 121       Luxembourg
# 129            Malta
# 137          Moldova
# 138           Monaco
# 141          Morocco
# 145            Nepal
# 169           Russia
# 181           Serbia
# 186         Slovenia
# 200         Tanzania
res = df[df['Climate'].isnull()]['Country']


#Зная регион этих стран, мы можем взять среднюю температуру по региону
mean_values = df.groupby('Region')['Climate'].transform('mean')
df['Climate'] = df['Climate'].fillna(mean_values)

#Тоже самое делаем для грамотности
mean_values = df.groupby('Region')['Literacy (%)'].transform('mean')
df['Literacy (%)'] = df['Literacy (%)'].fillna(mean_values)

#Удалим несколько оставшихся стран с пустыми значениями - осталось 221
df = df.dropna()

#endregion
#region Подготовка данных

#Убираем колонку Country - она тут выступает индексом
X = df.drop('Country',axis=1)

#Создание dummy переменных
X = pd.get_dummies(X)

#Масштабирование признаков
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

#endregion

#region Создание и обучение модели

#Ищем k через метод локтя
# ssd = []
# for k in range(2,30):
#     model = KMeans(n_clusters=k)
#     model.fit(scaled_X)
#     ssd.append(model.inertia_)
# plt.plot(range(2,30),ssd,'o--')

#Пробуем исследовать через силуэты
k = 3
model = KMeans(n_clusters=k)
model.fit(scaled_X)
show_silhouette_plt(k=k,model=model,scaled_X=scaled_X)



#endregion

#region *Нанесение стран на карту

#Подгружаем ISO коды для стран
#                                              Country                                         ISO Code
# 0                                        Afghanistan                                              AFG
# 1    Akrotiri and Dhekelia – See United Kingdom, The  Akrotiri and Dhekelia – See United Kingdom, The
# 2                                      Åland Islands                                              ALA
# 3                                            Albania                                              ALB
# 4                                            Algeria                                              DZA
iso_codes = pd.read_csv(r'D:\Khabarov\Репозиторий\MachineLearningCourse\DATA\country_iso_codes.csv')


#Делаем название страны индексом
iso_codes = iso_codes.set_index('Country')['ISO Code']

#Превращаем в словарь
iso_dict = iso_codes.to_dict()
res = iso_dict

#Добавляем ISO код
#             Country                               Region  ...  Service  ISO CODE
# 0       Afghanistan        ASIA (EX. NEAR EAST)           ...    0.380       AFG
# 1           Albania  EASTERN EUROPE                       ...    0.579       ALB
df['Iso'] = df['Country'].map(iso_dict)

#Добавляем колонку с кластерами
df['Cluster'] = model.labels_

# fig = px.choropleth(data_frame=df, locations="Iso",
#                     color="Cluster", # lifeExp is a column of gapminder
#                     hover_name="Country", # column to add to hover information
#                     )
# fig.show()

#endregion

print(res)
# plt.show()