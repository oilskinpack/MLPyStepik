import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from CrossValidation.Grid_Search import grid_model, best_model
from LinearRegression.regularisation import test_prediction

res = ''

df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\AMES_Final_DF.csv')

#    Lot Frontage  Lot Area  ...  Sale Condition_Normal  Sale Condition_Partial
# 0         141.0     31770  ...                      1                       0
res = df.head()

#Признаки
X = df.drop('SalePrice',axis=1)
res = X

#Целевая переменная
y = df['SalePrice']
res = y

#Деление на тренировочный и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=101)

#Масштабирование признаков
scaler = StandardScaler()
scaler.fit(X_train)

#Отмасштабированные данные
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Создание модели
elastic_model = ElasticNet()

#Создание словаря
param_grid = {'alpha':[1,2,3,5,7,11],'l1_ratio':[0.1,0.3,0.5,0.7,0.9,1]}

#Создание сетки
grid_model = GridSearchCV(estimator=elastic_model
                          ,param_grid=param_grid
                          ,scoring='neg_mean_squared_error'
                          ,cv=5
                          ,verbose=0)

#Обучение
grid_model.fit(X_train,y_train)

#Берем лучшую модель - ElasticNet(alpha=2, l1_ratio=0.9)
best_model = grid_model.best_estimator_

#Получаем целевую переменную для x_test
test_prediction = best_model.predict(X_test)

#Оценка метрик
# MAE:19551.7295104113
# RMSE:28984.055592445373
MAE = mean_absolute_error(y_test,test_prediction)
MSE = mean_squared_error(y_test,test_prediction)
RMSE = np.sqrt(MSE)
print(f'MAE:{MAE}\nRMSE:{RMSE}')


print(res)