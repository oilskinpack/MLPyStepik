import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

from CrossValidation.Grid_Search import grid_model

res = ''

df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\mushrooms.csv')

#region Разбивка данных

#Получение признаков
X = df.drop(columns='class',axis=1)
X = pd.get_dummies(X,drop_first=True)
res = X

#Получение целевой переменной
y = df['class']

#Деление данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

#endregion
#region Создание модели

#Создаем параметры для сетки
param_grid = {'n_estimators':[50,100],
              'learning_rate':[0.1,0.05,0.2],
              'max_depth':[3,4,5]}

#Создаем алгоритм
est = GradientBoostingClassifier()

#Создаем сетку
grid_model = GridSearchCV(param_grid=param_grid,estimator=est)

#Обучение
grid_model.fit(X_train,y_train)

#Предсказания
pred = grid_model.predict(X_test)



#endregion
#region Метрики
#Лучший алгоритм
best_est = grid_model.best_estimator_

#Лучшие параметры
best_params = grid_model.best_params_
print(best_params)

#Репорт
report = classification_report(y_test,pred)
print(report)

#Важность признаков
best_f = best_est.feature_importances_
df_best_f = pd.DataFrame(data=best_f,index=X.columns,columns=['Важность'])
df_best_f = df_best_f[df_best_f['Важность'] > 0.0005]
df_best_f = df_best_f.sort_values('Важность',ascending=False)
df_best_f = df_best_f.reset_index()

#Визуализация
plt.figure(figsize=(12,2.5),dpi=200)
plt.xticks(rotation=90,fontsize=3)
plt.yticks(fontsize=4)
plt.ylabel(ylabel='unique',fontsize=5)
ax = sns.barplot(data=df_best_f,x='index',y='Важность',palette='Set2')


#endregion


print(res)
plt.show()