import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate


desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)

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

#Создаем модель
model = Ridge(alpha=100)

#Создаем экземпляр scores, который сделает кросс валидацию и посчитает метрики
#Модель
#Признаки обуч + валид
#ЦФ обучающие + валид
#Метрики списком
#На сколько делим входящие данные
scores = cross_validate(model,X_train,y_train,scoring=['neg_mean_squared_error','neg_mean_absolute_error'],cv=10)

#Получаем дф с метриками для каждого разбиения
#    fit_time  score_time  test_neg_mean_squared_error  test_neg_mean_absolute_error
# 0  0.000998    0.000998                    -6.060671                     -1.810212
# 1  0.000998    0.000997                   -10.627031                     -2.541958
# 2  0.000997    0.000000                    -3.993426                     -1.469594
# 3  0.000000    0.000997                    -5.009494                     -1.862769
# 4  0.000997    0.000997                    -9.141800                     -2.520697
# 5  0.000998    0.000000                   -13.086256                     -2.459995
# 6  0.000000    0.001029                    -3.839405                     -1.451971
# 7  0.000998    0.000997                    -9.058786                     -2.377395
# 8  0.000000    0.000997                    -9.055457                     -2.443344
# 9  0.000998    0.000998                    -5.778882                     -1.899797
scores = pd.DataFrame(scores)
res = scores


#Средние метрики
# fit_time                        0.000900
# score_time                      0.000705
# test_neg_mean_squared_error    -7.565121
# test_neg_mean_absolute_error   -2.083773
res = scores.mean()

print(res)
