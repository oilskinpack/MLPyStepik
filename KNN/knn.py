import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.special.cython_special import huber
from scipy.stats import alpha
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from CrossValidation.Grid_Search import param_grid

res = ''

df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\gene_expression.csv')
res = df

#Создаем Scatterplot для проверки того насколько расходятся точки
sns.scatterplot(data=df,x='Gene One',y='Gene Two',hue='Cancer Present')
plt.clf()

#То же самое, но детальнее и больше визуализации
sns.scatterplot(data=df,x='Gene One',y='Gene Two',hue='Cancer Present',alpha=0.6,style='Cancer Present')
plt.xlim(2,6)
plt.ylim(4,8)
plt.clf()

#Pairplot
sns.pairplot(data=df,hue='Cancer Present')
plt.clf()


#Создание признаков
X = df.drop('Cancer Present',axis=1)
y = df['Cancer Present']

#Делим данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Масштабируем признаки
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Создание модели
knn_model = KNeighborsClassifier(n_neighbors=1)

#Обучение
knn_model.fit(X_train,y_train)

#Предсказание
y_pred = knn_model.predict(X_test)

#Матрица ошибок
conf_matr = confusion_matrix(y_test,y_pred)
res = conf_matr

#Отчет
rep = classification_report(y_test,y_pred)
res = rep



#========Пробуем найти подходящее значение K===========
test_error_rates = []

for k in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train,y_train)

    y_pred = knn_model.predict(X_test)
    test_error = 1 - accuracy_score(y_test,y_pred)
    test_error_rates.append(test_error)

res = test_error_rates

#Построение графика по методу локтя
plt.plot(range(1,30),test_error_rates)
plt.ylabel('Error rate')
plt.xlabel('К ближайших соседей')
plt.ylim(0,0.11)


#===Создание Pipeline=====
#Создание признаков
X = df.drop('Cancer Present',axis=1)
y = df['Cancer Present']

#Делим данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Создание pipeline => gridsearchCV
#Создаем пустые Scaler и Модель
scaler = StandardScaler()
knn_model = KNeighborsClassifier()

#Для справки - как посмотреть параметры модели
# dict_keys(['algorithm', 'leaf_size', 'metric', 'metric_params', 'n_jobs', 'n_neighbors', 'p', 'weights'])
res = knn_model.get_params().keys()

#Создаем операции, которые поместим в Pipeline
operations = [('scaler',scaler),('knn_model',knn_model)]

#Создание pipeline
pipe = Pipeline(operations)

#Задаем значения параметров
k_values = list(range(1,20))

#Создание словаря параметров для gsCv
param_grid = {'knn_model__n_neighbors':k_values}

#Создание gridsearchCV на основе pipeline
full_cv_classifier = GridSearchCV(pipe,param_grid,cv=5,scoring='accuracy')

#Обучение
full_cv_classifier.fit(X_train,y_train)

#Просмотр финальных параметров в лучшей модели
# {'memory': None,
#  'steps':
#      [('scaler', StandardScaler()),
#       ('knn_model', KNeighborsClassifier(n_neighbors=16))],
#  'verbose': False,
#  'scaler': StandardScaler(),
#  'knn_model': KNeighborsClassifier(n_neighbors=16),
#  'scaler__copy': True,
#  'scaler__with_mean': True,
#  'scaler__with_std': True,
#  'knn_model__algorithm': 'auto',
#  'knn_model__leaf_size': 30,
#  'knn_model__metric': 'minkowski',
#  'knn_model__metric_params': None,
#  'knn_model__n_jobs': None,
#  'knn_model__n_neighbors': 16,
#  'knn_model__p': 2,
#  'knn_model__weights': 'uniform'}
best_params = full_cv_classifier.best_estimator_.get_params()
res = best_params

#Оценка
full_pred = full_cv_classifier.predict(X_test)

#Репорт
#               precision    recall  f1-score   support
#
#            0       0.94      0.96      0.95       470
#            1       0.95      0.93      0.94       430
#
#     accuracy                           0.94       900
#    macro avg       0.94      0.94      0.94       900
# weighted avg       0.94      0.94      0.94       900
print(classification_report(y_test,full_pred))



print(res)
plt.show()