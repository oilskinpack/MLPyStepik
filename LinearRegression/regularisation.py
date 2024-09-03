import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r'D:\Khabarov\Курс ML\08-Linear-Regression-Models\Advertising.csv')

#Разбиваем признаки
X = df.drop('sales', axis=1)
y = df['sales']

#Создаем конвертер для полиномиальной регресси
polynomial_features = PolynomialFeatures(degree = 3,include_bias=False)
poly_features = polynomial_features.fit_transform(X)

#Разбиение данных
X_train, X_test, y_train, y_test = train_test_split(poly_features,y,test_size=0.3,random_state=101)

#Объект для масштабирования признаков
scaler = StandardScaler()

#Вычисляем данные для тренировочного куска данных
scaler.fit(X_train)

#Получаем новые тренировочные данные (Отмасштабируемые)
X_train = scaler.transform(X_train)
#Получаем новые тестовые данные (Отмасштабируемые)
X_test = scaler.transform(X_test)

#Импортируем модуль для ridge регрессии
from sklearn.linear_model import Ridge
#Создаем модель
ridge_model = Ridge(alpha = 10)

#Обучаем модель
ridge_model.fit(X_train,y_train)

#Вычисляем y с крышкой для тестовых данных
test_prediction = ridge_model.predict(X_test)

#Импортируем модуль метрик
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#Средняя абсолютная ошибка
#MAE:0.5774404204714166 . RMSE:0.8946386461319648
MAE = mean_absolute_error(y_test,test_prediction)
RMSE = np.sqrt(mean_squared_error(y_test,test_prediction))
res = fr'MAE:{MAE} . RMSE:{RMSE}'
print(res)


#МОЖНО ЕЩЕ СОБРАТЬ МОДЕЛЬ С РЕГУЛЯРИЗАЦИЕЙ
#С ИСПОЛЬЗОВАНИЕМ КРОСС ВАЛИДАЦИИ
from sklearn.linear_model import RidgeCV
#Создаем модель, которая с помощью кросс валидации и перебора а
#найдет оптимальные значения
#Параметры
#------------
#alphas - какие значения а будут проверяться
#cv - значение k для кросс валидации
#scoring - можно указать метрику оценки (все названия в scorers)
ridge_cv_model = RidgeCV(alphas=[0.1,1.0,10],scoring='neg_mean_squared_error')

#Обучаем модель
ridge_cv_model.fit(X_train,y_train)

#Получаем наилучшее значение альфа
a = ridge_cv_model.alpha_

#Вычисляем y с крышкой для тестовых данных
test_prediction = ridge_cv_model.predict(X_test)

#Средняя абсолютная ошибка
#MAE:0.42737748843375084 . RMSE:0.6180719926926787
MAE = mean_absolute_error(y_test,test_prediction)
RMSE = np.sqrt(mean_squared_error(y_test,test_prediction))
#res = fr'MAE:{MAE} . RMSE:{RMSE}'

#КОэффициенты
# [ 5.40769392  0.5885865   0.40390395 -6.18263924  4.59607939 -1.18789654
#  -1.15200458  0.57837796 -0.1261586   2.5569777  -1.38900471  0.86059434
#   0.72219553 -0.26129256  0.17870787  0.44353612 -0.21362436 -0.04622473
#  -0.06441449]
res = ridge_cv_model.coef_


#Scorers - словарь с метриками, всегда в большую сторону
from sklearn.metrics._scorer import _SCORERS as SCORERS
#Так можем посмотреть названия метрик
res = SCORERS.keys()



#=======LASSO регрессия======
from sklearn.linear_model import LassoCV

#Создаем и обучаем модель
lasso_cv_train = LassoCV(eps=0.1,n_alphas=100,cv=5)
lasso_cv_train.fit(X_train,y_train)

#Значение альфа - 0.004943070909225831
res = lasso_cv_train.alpha_

#у c крышкой
test_prediction = lasso_cv_train.predict(X_test)

#Метрики
#Средняя абсолютная ошибка
#MAE:0.6541723161252868 . RMSE:1.1308001022762548
MAE = mean_absolute_error(y_test,test_prediction)
RMSE = np.sqrt(mean_squared_error(y_test,test_prediction))
# res = fr'MAE:{MAE} . RMSE:{RMSE}'

#Коэффициенты
# [1.002651   0.         0.         0.         3.79745279 0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.        ]
res = lasso_cv_train.coef_



#=======L1+L2 регрессия======
from sklearn.linear_model import ElasticNetCV

#Создание и обучение модели
elastic_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],eps=0.1,n_alphas=100,max_iter = 1000000)
elastic_model.fit(X_train,y_train)

#Наилучшее значение L1 - 1.0
res = elastic_model.l1_ratio_

#Наилучшее значение альфа - 0.4943070909225831
res = elastic_model.alpha_

#у c крышкой
test_prediction = elastic_model.predict(X_test)

#Метрики
#Средняя абсолютная ошибка
#MAE:0.6541723161252868 . RMSE:1.1308001022762548
MAE = mean_absolute_error(y_test,test_prediction)
RMSE = np.sqrt(mean_squared_error(y_test,test_prediction))
res = fr'MAE:{MAE} . RMSE:{RMSE}'

print(res)




