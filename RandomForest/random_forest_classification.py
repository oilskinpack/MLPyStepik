import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.metrics._classification import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

res = ''

#region Без кросс-валидации

df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\penguins_size.csv')

#Чистка
df = df.dropna()

#Создаем dummy переменные
X = df.drop('species',axis=1)
X = pd.get_dummies(X,drop_first=True)
y = df['species']

#Деление
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Создание модели
#n_estimators - количество деревьев в лесу
#max_features - какое количество признаков брать ['auto','sqrt','log2'] или float
rfc = RandomForestClassifier(n_estimators=10,max_features='sqrt',random_state=101)

#Обучение модели
rfc.fit(X_train,y_train)

#Предсказания
pred = rfc.predict(X_test)
res = pred

#Вывод матрицы
# ConfusionMatrixDisplay(confusion_matrix(pred, y_test)).plot()


#Вывод матрицы
#               precision    recall  f1-score   support
#
#       Adelie       0.97      0.95      0.96        41
#    Chinstrap       0.92      0.96      0.94        23
#       Gentoo       1.00      1.00      1.00        37
#
#     accuracy                           0.97       101
#    macro avg       0.96      0.97      0.97       101
# weighted avg       0.97      0.97      0.97       101
# print(classification_report(y_test,pred))

#endregion
#region С кросс валидацией
df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\data_banknote_authentication.csv')

#  #   Column            Non-Null Count  Dtype
# ---  ------            --------------  -----
#  0   Variance_Wavelet  1372 non-null   float64
#  1   Skewness_Wavelet  1372 non-null   float64
#  2   Curtosis_Wavelet  1372 non-null   float64
#  3   Image_Entropy     1372 non-null   float64
#  4   Class             1372 non-null   int64 (0 - действительные, 1 - фальшивки)
# info = df.info()


#Pairplot
# sns.pairplot(df,hue='Class')


#Делим данные
X = df.drop('Class',axis=1)
y = df['Class']

#Разбиение данных на холд-аут и тест+трейн
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

#Создание параметров для поиска по сетке
n_estimators = [64,100,128,200]
max_features = [2,3,4]
bootstrap = [True,False]
oob_score = [True,False]
param_grid = {'n_estimators':n_estimators,
              'max_features':max_features,
              'oob_score':oob_score,
              'bootstrap':bootstrap}

#Создание базовой модели
rfc = RandomForestClassifier()

#Создание сетки
grid_model = GridSearchCV(param_grid=param_grid,estimator=rfc)

#Обучение модели
grid_model.fit(X_train,y_train)

#Лучшие параметры
best_params = grid_model.best_params_
res = best_params
print(res)

#Создание финальной модели с лучшими параметрами
best_model = RandomForestClassifier(**best_params)

#Обучение финальной модели
best_model.fit(X_train,y_train)

#Предсказания
# {'bootstrap': True, 'max_features': 2, 'n_estimators': 100, 'oob_score': True}
pred = best_model.predict(X_test)

#Вывод матрицы
# ConfusionMatrixDisplay(confusion_matrix(pred, y_test)).plot()

#Вывод матрицы
#               precision    recall  f1-score   support
#
#            0       1.00      0.98      0.99       124
#            1       0.98      1.00      0.99        82
#
#     accuracy                           0.99       206
#    macro avg       0.99      0.99      0.99       206
# weighted avg       0.99      0.99      0.99       206
print(classification_report(y_test,pred))

#Проверяем зависимость количества ошибок от количества
errors = []
misclassifications = []

for n in range(1,100):
    rfc = RandomForestClassifier(n_estimators = n,max_features=2)
    rfc.fit(X_train,y_train)
    preds = rfc.predict(X_test)

    err = 1 - accuracy_score(y_test,preds)
    n_missed = np.sum(preds != y_test)
    errors.append(err)
    misclassifications.append(n_missed)

#График ошибок от роста деревьев в лесу
# plt.plot(range(1,100),errors)

#График неправильного определения от роста деревьев в лесу
plt.plot(range(1,100),misclassifications)




#endregion

print(res)
plt.show()