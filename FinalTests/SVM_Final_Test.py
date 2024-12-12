import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from CrossValidation.Grid_Search import grid_model

res = ''

#region Постановка задачи

# Компания-дистрибьютор вина недавно столкнулась с подделками.
# В итоге был проведён аудит различных вин с помощью химического анализа.
# Компания занимается экспортом очень качественных и дорогих вин,
# но один из поставщиков попытался передать дешёвое вино под видом более дорогого.
# Компания-дистрибьютор наняла Вас, чтобы Вы создали модель машинного обучения,
# которая предскажет низкое качество вина (то есть, "подделку"). Они хотят узнать,
# возможно ли определить разницу между дешёвыми и дорогими винами.\

#Задача - **ЗАДАНИЕ: Обшая цель - используя данные ниже, разработайте модель машинного обучения,
# которая будет предсказывать на основе некоторых химических тестов, является ли вино настоящим или поддельным.
# Выполните задания ниже.**

#endregion
#region Загрузка и просмотр данных

df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\wine_fraud.csv')

#    fixed acidity  volatile acidity  citric acid  ...  alcohol  quality  type
# 0            7.4              0.70         0.00  ...      9.4    Legit   red
# 1            7.8              0.88         0.00  ...      9.8    Legit   red
# 2            7.8              0.76         0.04  ...      9.8    Legit   red
# 3           11.2              0.28         0.56  ...      9.8    Legit   red
# 4            7.4              0.70         0.00  ...      9.4    Legit   red
res = df.head()

#  #   Column                Non-Null Count  Dtype
# ---  ------                --------------  -----
#  0   fixed acidity         6497 non-null   float64
#  1   volatile acidity      6497 non-null   float64
#  2   citric acid           6497 non-null   float64
#  3   residual sugar        6497 non-null   float64
#  4   chlorides             6497 non-null   float64
#  5   free sulfur dioxide   6497 non-null   float64
#  6   total sulfur dioxide  6497 non-null   float64
#  7   density               6497 non-null   float64
#  8   pH                    6497 non-null   float64
#  9   sulphates             6497 non-null   float64
#  10  alcohol               6497 non-null   float64
#  11  quality               6497 non-null   object
#  12  type                  6497 non-null   object
res = df.info()


# quality
# Legit    6251
# Fraud     246
res = df['quality'].value_counts()

#endregion


#region Исследование данных

#region Соотношение классов
# sns.countplot(data=df,x='quality',palette='Set2')
#endregion
#region Какое вино подделывают чаще
# sns.countplot(data=df,x='type',palette='Set2',hue='quality')
#endregion
#region Какой процент вин (красного и белого) подделан

red_df = df[df['type'] == 'red']
white_df = df[df['type'] == 'white']

# 3.9399624765478425
res = (len(red_df[red_df['quality'] == 'Fraud']) / len(red_df)) * 100
# 3.7362188648427925
res = (len(white_df[white_df['quality'] == 'Fraud']) / len(white_df)) * 100

#endregion
#region Корреляции переменных с классами

qual_map = {'Fraud':1,'Legit':0}
df['quality_num'] = df['quality'].map(qual_map)
corr_qual = df.corr(numeric_only=True)['quality_num'].sort_values(ascending=True).reset_index()
# sns.barplot(data=corr_qual,x='index',y='quality_num',palette='Set2')


#endregion
#region Корреляция переменных

# sns.clustermap(df.corr(numeric_only=True),cmap='viridis')

#endregion

#endregion

#region Создание dummy переменных и деление признака с целевой переменной

y = df['quality']
df = df.drop(columns=['quality_num','quality'],axis=1)
X = pd.get_dummies(df,drop_first=True,dtype=int)
res = y



#endregion
#region Деление данных

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=101)

#endregion
#region Создание звеньев для Pipeline

scaler = StandardScaler()
est = SVC(class_weight='balanced')

#endregion
#region Создание Pipeline

operations = [('scaler',scaler),('est',est)]
pipe = Pipeline(steps=operations)

#endregion
#region Создание GridSearchCV

# c = [0.15,0.3,0.45,0.6,0.75,0.9,1.05]
# gamma = ['scale','auto']

c = [1.05]
gamma = ['scale']
param_grid = {'est__C':c
              ,'est__gamma':gamma}
grid_model = GridSearchCV(estimator=pipe
                          ,param_grid=param_grid
                          ,return_train_score=True
                          ,scoring='f1_weighted'
                          ,verbose=2
                          ,cv=10)




#endregion
#region Обучение модели и предсказание

grid_model.fit(X_train,y_train)
y_pred = grid_model.predict(X_test)


#Лучшие параметры - k = С = 1.05 gamma = scale
# {'memory': None, 'steps': [('scaler', StandardScaler()), ('est', SVC(C=1.05, class_weight='balanced'))],
#  'verbose': False, 'scaler': StandardScaler(), 'est': SVC(C=1.05, class_weight='balanced'),
#  'scaler__copy': True, 'scaler__with_mean': True, 'scaler__with_std': True, 'est__C': 1.05,
#  'est__break_ties': False, 'est__cache_size': 200, 'est__class_weight': 'balanced',
#  'est__coef0': 0.0, 'est__decision_function_shape': 'ovr', 'est__degree': 3,
#  'est__gamma': 'scale', 'est__kernel': 'rbf', 'est__max_iter': -1,
#  'est__probability': False, 'est__random_state': None,
#  'est__shrinking': True, 'est__tol': 0.001, 'est__verbose': False}
res = grid_model.best_estimator_.get_params()


#endregion

#region Оценка
#Матрица ошибок
# [[ 17  10]
#  [ 91 532]]
confM = confusion_matrix(y_test,y_pred)
res = confM

#Визуализация матрицы ошибок
ConfusionMatrixDisplay(confusion_matrix(y_pred, y_test)).plot()

#Отчет по метрикам
#               precision    recall  f1-score   support
#
#        Fraud       0.16      0.63      0.25        27
#        Legit       0.98      0.85      0.91       623
#
#     accuracy                           0.84       650
#    macro avg       0.57      0.74      0.58       650
# weighted avg       0.95      0.84      0.89       650
metricsRep = classification_report(y_test,y_pred)
print(metricsRep)

#endregion



print(res)
plt.show()