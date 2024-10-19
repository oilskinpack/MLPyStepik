import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from CrossValidation.Grid_Search import best_model

res = ''

#region Загрузка и проверка данных
#      class cap-shape cap-surface  ... spore-print-color population habitat
# 0        p         x           s  ...                 k          s       u
# 1        e         x           s  ...                 n          n       g
# class - p(poisoned) e(edible)
df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\mushrooms.csv')

#Проверяем насколько сбалансированный датасет
# class
# e    4208
# p    3916
res = df['class'].value_counts()

#Проверяем какие значения есть в колонках
#count - количество ненулевых значений
#unique - количество уникальных значений
#top - самое частое значение
#fred - количество появления самого частого значения

#        class cap-shape cap-surface  ... spore-print-color population habitat
# count   8124      8124        8124  ...              8124       8124    8124
# unique     2         6           4  ...                 9          6       7
# top        e         x           y  ...                 w          v       d
# freq    4208      3656        3244  ...              2388       4040    3148
res = df.describe()

#Смотрим по каким колонкам больше всего уникальных значений
#Они скорее всего дадут больше всего информации для моделей
#                        index count unique top  freq
# 9                 gill-color  8124     12   b  1728
# 3                  cap-color  8124     10   n  2284
# 20         spore-print-color  8124      9   w  2388
unique_val_df = df.describe().transpose().reset_index().sort_values('unique',ascending=False)

#Построим countplot по полученному разбросу уникальных значений
# plt.figure(figsize=(12,2.5),dpi=200)
# plt.xticks(rotation=90,fontsize=3)
# plt.yticks(fontsize=4)
# plt.ylabel(ylabel='unique',fontsize=5)
# ax = sns.barplot(data=unique_val_df,x='index',y='unique',palette='Set2')

#Проверка пустых значений - таких нет
res = df.isnull().sum()

#endregion
#region Разбивка данных

#Получение признаков
X = df.drop(columns='class',axis=1)
X = pd.get_dummies(X,drop_first=True)
res = X

#Получение целевой переменной
y = df['class']

#Деление данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

#



#endregion
#region Создание модели

#Создаем адабуст
#Здесь создаем цикл, который будет проверять - какое будет количество ошибок при каком наборе признаков
#Так как стандартным алгоритмом здесь выступает простейшее дерево, то Кол-во алгоритмов = Кол-во исп. признаков
# error_rates = []
# for n in range(1,50):
#     model = AdaBoostClassifier(n_estimators=n)
#     model.fit(X_train,y_train)
#     preds = model.predict(X_test)
#     err = 1 - accuracy_score(y_test,preds)
#     error_rates.append(err)
# plt.plot(range(1,50),error_rates)


#Создаем модель наилучшим количеством деревьев
best_model = AdaBoostClassifier(n_estimators=20)
best_model.fit(X_train,y_train)
preds = best_model.predict(X_test)

#Интерпретация значений
best_f = best_model.feature_importances_
df_best_f = pd.DataFrame(data=best_f,index=X.columns,columns=['Важность'])
df_best_f = df_best_f[df_best_f['Важность'] > 0]
df_best_f = df_best_f.sort_values('Важность',ascending=False)
df_best_f = df_best_f.reset_index()

#Визуализация
plt.figure(figsize=(12,2.5),dpi=200)
plt.xticks(rotation=90,fontsize=3)
plt.yticks(fontsize=4)
plt.ylabel(ylabel='unique',fontsize=5)
ax = sns.barplot(data=df_best_f,x='index',y='Важность',palette='Set2')


#endregion


print(res)
plt.show()