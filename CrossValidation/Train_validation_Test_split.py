import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r'D:\Khabarov\Курс ML\08-Linear-Regression-Models\Advertising.csv')
res = df

#Берем признаки
X = df.drop('sales', axis=1)

#Берем целевую переменную
y = df['sales']

#Разбиваем на обучающие и Валидационные+Тестовые данные
X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.3, random_state=101)

#Разбиваем Валидационные и тестовые данные
X_eval, X_test, y_eval, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=101)

#Создаем Scaler и учим нормализовывать значения
scaler = StandardScaler()
scaler.fit(X_train)

#Нормализуем признаки
X_train = scaler.transform(X_train)
X_eval = scaler.transform(X_eval)
X_test = scaler.transform(X_test)

#Создаем модель и обучаем
model = Ridge(alpha=100)
model.fit(X_train, y_train)

#Получаем предсказания для тестовых признаков
y_eval_pred = model.predict(X_eval)

#Оценка работы
#7.320101458823869 (1)
MSE = mean_squared_error(y_eval, y_eval_pred)
res = MSE

#==========Тут мы подстроили гиперпараметры и устаканили модел===================

#Получаем ФИНАЛЬНУЮ МЕТРИКУ
#Получаем предсказания для тестовых признаков
y_final_test_pred = model.predict(X_test)

#Оценка работы
#7.363450119244385 (1)
MSE_final = mean_squared_error(y_test, y_final_test_pred)
res = MSE_final

print(res)