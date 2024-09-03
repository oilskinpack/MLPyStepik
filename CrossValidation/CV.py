import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r'D:\Khabarov\Курс ML\08-Linear-Regression-Models\Advertising.csv')
res = df

#Берем признаки
X = df.drop('sales', axis=1)

#Берем целевую переменную
y = df['sales']


#Разбиваем на обучающие и Валидационные+Тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Создаем Scaler и учим нормализовывать значения
scaler = StandardScaler()
scaler.fit(X_train)

#Стандартизируем данные для трейн и тест
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Создаем модель и обучаем
model = Ridge(alpha=1)

#Метод делит данные на количество частей и для каждого разбиения обучает модель + считает метрику
#model - выбранная нами модель
#Трейн + валид признаки
#Трейн + валид целевая переменная
#Метрика (ключом)
#На сколько частей бьем данные
scores = cross_val_score(model, X_train,y_train,scoring='neg_mean_squared_error',cv=5)
#[ -9.32552967  -4.9449624  -11.39665242  -7.0242106   -8.38562723]
res = scores

#Считаем общую метрику - 8.215396464543607
res = abs(scores.mean())

#Уже обучаем модель
model.fit(X_train, y_train)

#Получаем ФИНАЛЬНУЮ МЕТРИКУ
#Получаем предсказания для тестовых признаков
y_final_test_pred = model.predict(X_test)

#Оценка работы
#2.3190215794287514 (1)
MSE_final = mean_squared_error(y_test, y_final_test_pred)
res = MSE_final

print(res)