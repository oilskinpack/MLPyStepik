import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\iris.csv')


#===============АНАЛИЗ==================

#    sepal_length  sepal_width  petal_length  petal_width species
# 0           5.1          3.5           1.4          0.2  setosa
# 1           4.9          3.0           1.4          0.2  setosa
# 2           4.7          3.2           1.3          0.2  setosa
# 3           4.6          3.1           1.5          0.2  setosa
# 4           5.0          3.6           1.4          0.2  setosa
res = df.head()

#Информация
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   sepal_length  150 non-null    float64
#  1   sepal_width   150 non-null    float64
#  2   petal_length  150 non-null    float64
#  3   petal_width   150 non-null    float64
#  4   species       150 non-null    object
res = df.info()

#Распределения
#        sepal_length  sepal_width  petal_length  petal_width
# count    150.000000   150.000000    150.000000   150.000000
# mean       5.843333     3.054000      3.758667     1.198667
# std        0.828066     0.433594      1.764420     0.763161
# min        4.300000     2.000000      1.000000     0.100000
# 25%        5.100000     2.800000      1.600000     0.300000
# 50%        5.800000     3.000000      4.350000     1.300000
# 75%        6.400000     3.300000      5.100000     1.800000
# max        7.900000     4.400000      6.900000     2.500000
res = df.describe()


#Классы и баланс
# species
# setosa        50
# versicolor    50
# virginica     50
res = df['species'].value_counts()


#Скаттерплот для анализа трендов
sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=df)
plt.clf()

#Пэирплот
sns.pairplot(df, hue='species')
plt.clf()

#Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.clf()

#===============Создание==================

#Признаки
X = df.drop('species', axis=1)

#Классы
y = df['species']


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#Делим на обучающие и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

#Стандартизация признаков
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

#Создание модели
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
log_model = LogisticRegression(solver='saga',multi_class='ovr', max_iter=5000)

#Гиперпараметры
penalty = ['l1', 'l2']
l1_ratio = np.linspace(0, 1, 20)
C = np.logspace(0,10,20)

#Словарь для сетки
param_grid = {'penalty': penalty,'l1_ratio':l1_ratio,'C':C}

#Создание Grid
grid_model = GridSearchCV(log_model,param_grid)

#Обучение
grid_model.fit(scaled_X_train,y_train)

#Проверяем лучшие параметры
# {'C': 11.28837891684689, 'l1_ratio': 0.0, 'penalty': 'l1'}
res = grid_model.best_params_

#Высчитываем метрики
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay
y_pred = grid_model.predict(scaled_X_test)

#Acc
acc = accuracy_score(y_test,y_pred)

#Матрица ошибок
confM = confusion_matrix(y_test,y_pred)

#График для матрицы
confusion_matrix(grid_model,scaled_X_test,y_test)

#Отчет по метрикам
metricsRep = classification_report(y_test,y_pred)

print(res)
plt.show()