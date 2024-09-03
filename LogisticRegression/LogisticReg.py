import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\hearing_test.csv')

#     age  physical_score  test_result
# 0  33.0            40.7            1
# 1  50.0            37.2            1
# 2  52.0            24.7            0
# 3  56.0            31.0            0
# 4  35.0            42.9            1
res = df.head()


# test_result
# 1    3000
# 0    2000
res = df['test_result'].value_counts()


#Создание countplot
sns.countplot(x='test_result', data=df)
plt.clf()

#Создаем boxplot по возрасту
sns.boxplot(x='test_result', y='age',data=df)
plt.clf()

#Создаем boxplot по физическому состоянию
sns.boxplot(x='test_result', y='physical_score',data=df)
plt.clf()

#Создаем scatterplot зависимости между возрастом и физ состоянием
sns.scatterplot(data=df,x='age',y='physical_score',hue='test_result',alpha=0.5)
plt.clf()

#Создаем паирплот для поиска корреляций
sns.pairplot(data=df,hue='test_result')
plt.clf()

#Создаем тепловую карту
sns.heatmap(data=df.corr(),annot=True)
plt.clf()

#Создаем 3д график по трем признакам
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df['age'],df['physical_score'],df['test_result'],c=df['test_result'])


#Берем признаки
X = df.drop('test_result',axis=1)

#Берем целевую переменную
y = df['test_result']

#Масштабируемся
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Делим на обучающие и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

#Масшатбирование
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

#Создаем модель
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()

#Обучаем модель
log_model.fit(scaled_X_train,y_train)

#Коэффициенты Бета - [[-0.95017725  3.46148946]]
y_pred = log_model.coef_
res = y_pred


#Получаем предсказания для тестового набора - Массив нулей и единиц
# [1 1 0 1 0 0 1 1 0 1 1 1 1 0 1 1 0 1 1 0 0 1 0 1 1 0 1 1 0 1 1 1 1 1 1 0 1
#  1 0 1 1 1 1 0 0 1 1 1 1 1 1 0 0 0 1 0 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 1 0
y_pred = log_model.predict(scaled_X_test)
res = y_pred

#Получаем вероятности - Массив массивов с вероятностями принадлежности к тому или иному классу
# [[2.38051656e-02 9.76194834e-01]
#  [2.68854070e-02 9.73114593e-01]
y_pred_proba = log_model.predict_proba(scaled_X_test)
res = y_pred_proba

#############МЕТРИКИ#################
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
#Acc - 0.93
accuracy = accuracy_score(y_test,y_pred)
res = accuracy

#Матрица
# [[172  21]
#  [ 14 293]]
confusion_matrix = confusion_matrix(y_test,y_pred)
res = confusion_matrix

#Визуализация матрицы
from sklearn.metrics import ConfusionMatrixDisplay
confusion_matrix(log_model,scaled_X_test,y_test)


print(res)
plt.show()