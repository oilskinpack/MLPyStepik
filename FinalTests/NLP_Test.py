import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

res = ''

#region Постановка задачи

# На входе комментарии к фильмам. Нужно построить модель обучения, которая определяет положительный отзыв или отрицательный



#endregion
#region Загрузка и просмотр данных

df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\moviereviews.csv')

#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   label   2000 non-null   object
#  1   review  1965 non-null   object
res = df.info()


# label
# neg    1000
# pos    1000
res = df['label'].value_counts()

df['review'] = df['review'].astype(str)



#   label                                             review
# 0   neg  how do films like mouse hunt get into theatres...
# 1   neg  some talented actresses are blessed with a dem...
# 2   pos  this has been an extraordinary year for austra...
# 3   pos  according to hollywood movies made in last few...
# 4   neg  my first press screening of 1998 and already i...
res = df.head()

#endregion


#region Исследование и подготовка данных

#region Поиск и удаление Nan значений
# label      0
# review    35
res = df.isnull().sum()
df = df.dropna(axis=0,subset='review')
# res = len(df)

res = len(df[df['review'].str.len() <= 3])
df['review']= np.where(df['review'].str.len() <= 3,np.nan,df['review'])
df = df.dropna(axis=0,subset='review')

res = len(df)


#endregion
#region Поиск самых популярных слов в позитивных и негативных комментариях

#region Позитивные
#Создаем датафрейм положительных слов
df_pos = df[df['label'] == 'pos']

#Создаем мешок слов в виде разряженной матрицы, возьмем только 20 самых популярных слов
cv = CountVectorizer(stop_words='english',max_features=20)
sparse_matrix = cv.fit_transform(df_pos['review'])

#Названия колонок
feature_names = cv.get_feature_names_out()
res = feature_names
#endregion
#region Негативные
#Создаем датафрейм положительных слов
df_neg = df[df['label'] == 'neg']

#Создаем мешок слов в виде разряженной матрицы, возьмем только 20 самых популярных слов
cv = CountVectorizer(stop_words='english',max_features=20)
sparse_matrix = cv.fit_transform(df_neg['review'])

#Названия колонок
feature_names = cv.get_feature_names_out()
res = feature_names
#endregion


#endregion

#endregion

#region Деление данных

X = df['review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#endregion
#region Создание звеньев для Pipeline

tfidf = TfidfVectorizer(stop_words='english')
est = SVC()


#endregion
#region Создание Pipeline

operations = [('tfidf',tfidf),('est',est)]
pipe = Pipeline(steps=operations)

#endregion
#region Создание GridSearchCV


c = [0.15,0.3,0.45,0.6,0.75,0.9,1.05]
gamma = ['scale','auto']

# c = [1.05]
# gamma = ['scale']
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


#endregion

#region Оценка
#Матрица ошибок
# [[161  30]
#  [ 38 159]]
confM = confusion_matrix(y_test,y_pred)
print(confM)

#Визуализация матрицы ошибок
ConfusionMatrixDisplay(confusion_matrix(y_pred, y_test)).plot()

#Отчет по метрикам
#               precision    recall  f1-score   support
#
#          neg       0.81      0.84      0.83       191
#          pos       0.84      0.81      0.82       197
#
#     accuracy                           0.82       388
#    macro avg       0.83      0.83      0.82       388
# weighted avg       0.83      0.82      0.82       388
metricsRep = classification_report(y_test,y_pred)
print(metricsRep)

#endregion



print(res)
plt.show()