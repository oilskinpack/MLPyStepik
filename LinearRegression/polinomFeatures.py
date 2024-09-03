import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv(r'D:\Khabarov\Курс ML\08-Linear-Regression-Models\Advertising.csv')
res = df.head()

#Выделяем признаки и целевую функцию
X = df.drop('sales',axis=1)
y = df['sales']

#Создаем экземпляр полиномиального конвертера - он переведет наши фичи в новую форму
#degree - степень полинома
#include_bias - оставлять ли константу, когда все признаки равны 0
#interaction_only - оставляются только значения перемноженных признаков
polynomial_converter = PolynomialFeatures(degree=2,include_bias=False)

#Метод fit - анализирует данные
polynomial_converter.fit(X)

#Получаем матрицу новых признаков
poly_features = polynomial_converter.transform(X)
#Таким вот образом можем преобразовать их в датафрейм
dfPolyf = pd.DataFrame(poly_features,columns=polynomial_converter.get_feature_names_out(X.columns))


#Разбиваем данные на тестовые и тренировочные - вместо X указываем новые признаки poly_features
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.33, random_state=101)

#Создаем модель линейной регрессии
from sklearn.linear_model import LinearRegression
model = LinearRegression()

#Тренируем модель
model.fit(X_train,y_train)

#Получаем предсказание
test_predictions = model.predict(X_test)

#Проверяем коэффициенты
# [ 5.25319441e-02  1.42773271e-02  1.47528851e-02 -1.12739168e-04
#   1.13231490e-03 -5.42180033e-05  6.26813126e-05  8.93347558e-05
#  -3.52004070e-05]
# res = model.coef_

#Оценка полиномиальной модели
#Средняя абсолютная ошибка: 0.48428105352155904. Среднеквадратичное отклонение 0.6482912032533674 - Полиномиальная регрессия
#Средняя абсолютная ошибка: 1.213. Среднеквадратичное отклонение 1.516                            - Линейная регрессия
from sklearn.metrics import mean_absolute_error,mean_squared_error
MAE = mean_absolute_error(y_test,test_predictions)
MSE = mean_squared_error(y_test,test_predictions)
RMSE = np.sqrt(MSE)
# res = f'Средняя абсолютная ошибка: {MAE}. Среднеквадратичное отклонение {RMSE}'


#Проверка коэффициентов B
#          TV     radio  newspaper      TV^2  TV radio  TV newspaper   radio^2  radio newspaper  newspaper^2
# 0  0.052532  0.014277   0.014753 -0.000113  0.001132     -0.000054  0.000063         0.000089    -0.000035
dfCoefB = pd.DataFrame([model.coef_],columns=polynomial_converter.get_feature_names_out(X.columns))


#Определеяем сложность модели

#1.Создать различные степени полинома
#2.Разбить данные на обучающий и тестовый наборы данных
#3. Обученить модель
#4. Сохранить метрики RMSE для обучающего и тестового набора данных
#5.Нарисовать график с результатами - ошибка по степеням полинома

#Среднеквадратические отклонения для обучающего набора данных
train_rmse_errors = []
#Среднеквадратическое отклонения для тестового набора данных
test_rmse_errors = []

#Проверяем степени полинома
for d in range(1,10):

    #Создаем полиномиальный конвертер
    poly_converter = PolynomialFeatures(degree=d,include_bias=False)
    poly_features = poly_converter.fit_transform(X)
    #Разбиваем данные на тестовые и тренировочные
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.33, random_state=101)

    #Создаем модель
    model = LinearRegression()
    model.fit(X_train,y_train)

    #Делаем предсказание для тестовых и тренировочных данных
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    #Вычисляем ошибку для тренировочных данных
    train_rmse = np.sqrt(mean_squared_error(y_train,train_pred))
    #Вычисляем ошибку для тестовых данных
    test_rmse = np.sqrt(mean_squared_error(y_test,test_pred))

    #Добавляем в списки
    train_rmse_errors.append(train_rmse)
    test_rmse_errors.append(test_rmse)

#Нанесем данные на график
plt.plot(range(1,6), train_rmse_errors[:5],label = 'Train RMSE')
plt.plot(range(1,6), test_rmse_errors[:5],label = 'Test RMSE')


final_poly_converter = PolynomialFeatures(degree=3,include_bias=False)
final_model = LinearRegression()

full_converted_X = final_poly_converter.fit_transform(X)
final_model.fit(full_converted_X,y)

#Сохраняем модель и конвертер
from joblib import dump,load
dump(final_model,r'D:\Khabarov\Скрипты\pythonScripts\LinearRegression\final_poly_model.joblib')
dump(final_poly_converter,r'D:\Khabarov\Скрипты\pythonScripts\LinearRegression\final_poly_converter.joblib') 

#Загрузка модели
loaded_converter = load(r'D:\Khabarov\Скрипты\pythonScripts\LinearRegression\final_poly_converter.joblib')
loaded_model = load(r'D:\Khabarov\Скрипты\pythonScripts\LinearRegression\final_poly_model.joblib')

campaign = [[149,22,12]]
transformed_data = loaded_converter.fit_transform(campaign)
loaded_model.predict(transformed_data)



#plt.show()



 

print(res)