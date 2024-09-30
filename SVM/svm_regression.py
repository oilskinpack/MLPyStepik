import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVR, SVR


res = ''

#    Cement   Slag  ...  FLOW(cm)  Compressive Strength (28-day)(Mpa)
# 0   273.0   82.0  ...      62.0                               34.99
# 1   163.0  149.0  ...      20.0                               41.14
# 2   162.0  148.0  ...      20.0                               41.81
# 3   162.0  148.0  ...      21.5                               42.08
df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\cement_slump.csv')
res = df.head()

#Корреляции
sns.heatmap(df.corr())

#Деление
X = df.drop('Compressive Strength (28-day)(Mpa)',axis=1)
y = df['Compressive Strength (28-day)(Mpa)']

#Делим на обучающие и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Масштабирование
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

#Создание и обучение модели
base_model = SVR()
base_model.fit(scaled_X_train,y_train)

#Оценка
base_pred = base_model.predict(X_test)
mse = mean_absolute_error(y_test,base_pred)
print(f'Ошибка базовой модели {mse}')


#Кросс валидация
param_grid = {'C':[0.001,0.01,0.1,0.5,1],
              'kernel':['linear','rbf','poly'],
              'gamma':['scale','auto'],
              'degree':[2,3,4],
              'epsilon':[0,0.01,0.1,0.5,1,2]}
svr = SVR()
grid_model = GridSearchCV(estimator=svr,param_grid=param_grid)
grid_model.fit(scaled_X_train,y_train)
print(f'Лучшие параметры: {grid_model.best_params_}')

#Оценка
grid_pred = grid_model.predict(scaled_X_test)
mse = mean_absolute_error(y_test,grid_pred)
print(f'Ошибка через CV: {mse}')


# Ошибка базовой модели 6.762776104941544
# Лучшие параметры: {'C': 1, 'degree': 2, 'epsilon': 2, 'gamma': 'scale', 'kernel': 'linear'}
# Ошибка через CV: 2.5128012210762365
