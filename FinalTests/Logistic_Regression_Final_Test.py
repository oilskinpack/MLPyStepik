
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay

res = ''

#region Задача данных и постановка задачи
# В базе данных содержатся 14 атрибутов о физическом тестировании пациентов.
# Они сдают кровь и выполняют небольшой физический тест.
# Колонка "goal" указывает на наличие заболевания сердца у пациента - 0 означает заболевания нет,
# 1 означает заболевание есть. В общем случае, подтвердить на 100% заболевание сердца это очень инвазивный процесс,
# поэтому если мы сможем построить модель, которая достаточно точно оценивает вероятность заболевания,
# то это поможет избежать дорогих инвазивных процедур.

# Информация об атрибутах:
#
# * age - возраст
# * sex - пол
# * cp - chest pain - тип боли в груди (4 значения)
# * trestbps - resting blood pressure - давление в состоянии покоя
# * chol - уровень холистерина в крови, в mg/dl
# * fbs - fasting blood sugar - уровень сахара в крови, > 120 mg/dl
# * restecg - resting electrocardiographic results - результаты электрокардиограммы (значнеия 0,1,2)
# * thalach - максимальный пульс
# * exang - exercise induced angina - возникновение ангины в результате упражнений
# * oldpeak = пиковые значения в электрокардиограмме, как результат упражнений (по сравнению с состоянием покоя)
# * slope - наклон пикового значения в электрокардиограмме, как результат упражнений (по сравнению с состоянием покоя)
# * ca - количество крупных сосудов (0-3), окрашенных флурозопией
# * thal -  3 = нормально; 6 = фиксированный дефект; 7 = обратимый дефект
# * target - 0 означает отсутствие заболевания сердца, 1 означает наличие заболевания сердца
df = pd.read_csv('D:\Khabarov\Курс ML\DATA\heart.csv')
#endregion

#region Анализ и визуализация данных

#region Проверка нулевых значений

# age         0.0
# sex         0.0
#................
# thal        0.0
# target      0.0
#Отсутствующих данных нет
has_null_cols = df.isnull().sum() / len(df)

#endregion
#region Проверяем насколько сбалансированы данные - Countplot
# sns.countplot(data=df,x='target',palette='Set2')
#endregion
#region Проверяем корреляцию по колонкам через pairplot
pair_df = df[['age','trestbps','chol','thalach','target']]
# sns.pairplot(data=pair_df,corner=True,hue='target')
#endregion
#region Проверяем корреляцию через heatmap

# sns.heatmap(data=pair_df.corr(),linewidths=0.5,annot=True,cmap="viridis")
#endregion

#endregion
#region Построение модели

#region Разбиение данных

X = df.drop(columns=['target'],axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

#endregion
#region Создание стоковых элементов для Pipeline
scaler = StandardScaler()
est = LogisticRegression()

#endregion
#region Создание Pipeline

operations = [('scaler',scaler)
              ,('est',est)]
pipeline = Pipeline(operations)

#endregion
#region Создание GridSearchCV

param_grid = {'est__penalty':['l1','l2']
              ,'est__C':[0.1,0.3,0.5,0.8,1.3,2.1,3.4,5.5]
              ,'est__solver':['liblinear']}
p_grid_model = GridSearchCV(estimator=pipeline
                                ,param_grid=param_grid
                                , scoring='f1_weighted'
                                , verbose=2
                                , cv=10)

#endregion
#region Обучение

p_grid_model.fit(X_train,y_train)
res = p_grid_model.best_estimator_.get_params()

#endregion
#region Оценка

y_pred = p_grid_model.predict(X_test)

#Матрица ошибок
# [[12  3]
 # [ 2 14]]
confM = confusion_matrix(y_test,y_pred)

#Визуализация матрицы ошибок
# ConfusionMatrixDisplay(confusion_matrix(y_pred, y_test)).plot()

#Отчет по метрикам
#               precision    recall  f1-score   support
#
#            0       0.86      0.80      0.83        15
#            1       0.82      0.88      0.85        16
#
#     accuracy                           0.84        31
#    macro avg       0.84      0.84      0.84        31
# weighted avg       0.84      0.84      0.84        31
metricsRep = classification_report(y_test,y_pred)
res = metricsRep





#endregion

#endregion



print(res)
plt.show()