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
from sklearn.linear_model import ElasticNet


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

#Создаем экземпляр эстимейтора - алгоритма (в нем же смотрим гиперпараметры)
base_elastic_net_model = ElasticNet()

#Создаем словарь с Гиперпараметрами под конкретную модель (и их значения)
param_grid = {'alpha':[0.1,1,5,50,100],'l1_ratio':[0.1,0.5,0.7,0.95,0.99,1]}

#Создаем модель
from sklearn.model_selection import GridSearchCV
#est - алгоритм
#param_grid - словарь с гиперпараметрами и их значениями
#scoring - ключ для метрики
#cv - на сколько частей бьем при кросс-валидации
#verbose - как много информации выводить рил тайм при обучении (0 - не выводить)
grid_model = GridSearchCV(estimator=base_elastic_net_model,
                          param_grid = param_grid,
                          scoring='neg_mean_squared_error',
                          cv=5,
                          verbose=2)

#Обучение
grid_model.fit(X_train, y_train)

#Берем лучшую - ElasticNet(alpha=0.1, l1_ratio=1)
best_model = grid_model.best_estimator_
res = best_model

print(res)