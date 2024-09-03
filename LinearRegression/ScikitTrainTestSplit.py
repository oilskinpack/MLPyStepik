import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from joblib import dump,load


#Загрузка данных - затраты на рекламу и продажи
#         TV  radio  newspaper  sales
# 0    230.1   37.8       69.2   22.1
# 1     44.5   39.3       45.1   10.4
df = pd.read_csv(r'D:\Khabarov\Курс ML\08-Linear-Regression-Models\Advertising.csv')
res = df


#Построим pairgrid
# g = sns.PairGrid(data=df,corner=True)
# g = g.map_upper(sns.scatterplot)
# g = g.map_diag(sns.kdeplot)
# g = g.map_lower(sns.scatterplot)

#Получаем признаки (X)
X = df[['TV','radio','sales']]
res = X

#Получаем лейбл
y = df['sales']
res = y

#Разбивка данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Создаем модель (пока без каких либо параметров/гиперпараметров)
model = LinearRegression()

#Тренируем модель
model.fit(X_train,y_train)

#Предсказываем лейбл
# [16.9 22.4 21.4  7.3 24.7 12.6 22.3  8.4 11.5 14.9  9.5  8.7 11.9  5.3
#  10.3 11.7  5.5 16.6 11.3 18.9 19.7 12.5 10.9 22.2  9.3  8.1 21.7 13.4
#  10.6  5.7 10.6 11.3 23.7  8.7 16.1 20.7 11.6 20.8 11.9  6.9 11.  12.8
#  10.1  9.7 11.6  7.6 10.5 14.6 10.4 12.  14.6 11.7  7.2  6.6  9.4 11.
#  10.9 25.4  7.6 11.7]
test_predictions = model.predict(X_test)


#Анализ данных - 14.0225
#res = df['sales'].mean()
#sns.histplot(data=df,x='sales',bins=20)

#1 - 2.412884706852007e-15
res = mean_absolute_error(y_test,test_predictions)

#2 - 2.9345751904939687e-15
res = np.sqrt(mean_squared_error(y_test,test_predictions))

#Оценка остатков - (y - y^)
test_residuals = y_test - test_predictions
# sns.scatterplot(x=y_test,y=test_residuals)
# plt.axhline(y=0,color='r',ls='--')

sns.displot(test_residuals,bins=25,kde=True)


#Создание финальной модели
final_model = LinearRegression()
final_model.fit(X,y)

#Проверка коэф бета - для TV,radio,newspaper по порядку
#[0.04576456  0.18853002  -0.00103749]
res = final_model.coef_

#Сохранение модели
# dump(final_model,r'D:\Khabarov\Скрипты\pythonScripts\LinearRegression\final_sales.joblib')

#Загрузка данных
loaded_model = load(r'D:\Khabarov\Скрипты\pythonScripts\LinearRegression\final_sales.joblib')
#Применение модели
campaign = [[149,22,12]]
dfCampaign = pd.DataFrame(campaign,columns=['TV','radio','newspaper'])
res = loaded_model.predict(dfCampaign)



# plt.show()
print(res)