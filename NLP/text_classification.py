import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

res = ''
plt.figure(figsize=(7,3))

#region Для функций

def report(model,X_test_tfidf):
    preds = model.predict(X_test_tfidf)
    print(classification_report(y_test,preds))
    conf_matrix = confusion_matrix(y_test, preds, labels=model.classes_)
    ConfusionMatrixDisplay(conf_matrix,display_labels=model.classes_).plot()

#endregion
#region Загрузка данных
df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\airline_tweets.csv')

#  #   Column                        Non-Null Count  Dtype
# ---  ------                        --------------  -----
#  0   tweet_id                      14640 non-null  int64
#  1   airline_sentiment             14640 non-null  object
#  2   airline_sentiment_confidence  14640 non-null  float64
#  3   negativereason                9178 non-null   object
#  4   negativereason_confidence     10522 non-null  float64
#  5   airline                       14640 non-null  object
#  6   airline_sentiment_gold        40 non-null     object
#  7   name                          14640 non-null  object
#  8   negativereason_gold           32 non-null     object
#  9   retweet_count                 14640 non-null  int64
#  10  text                          14640 non-null  object
#  11  tweet_coord                   1019 non-null   object
#  12  tweet_created                 14640 non-null  object
#  13  tweet_location                9907 non-null   object
#  14  user_timezone                 9820 non-null   object
# res = df.info()
#endregion
#region Анализ данных

#Смотрим разбивку по эмоциональной окраске твита
#sns.countplot(data=df,x='airline_sentiment',palette='Set2')

#По какой причине твит был негативным
#sns.countplot(data=df,x='negativereason',palette='Set2')

#Смотрим для каких авиакомпаний какая разбивка по окраске твита
#sns.countplot(data=df,x='airline',hue='airline_sentiment',palette='Set2')

#endregion
#region Создаем признаки

#Берем только целевую переменную и текст
#       airline_sentiment                                               text
# 0               neutral                @VirginAmerica What @dhepburn said.
# 1              positive  @VirginAmerica plus you've added commercials t...
data = df[['airline_sentiment','text']]

#Разбиваем на тест-трейн
X = data['text']
y = data['airline_sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

#Векторизуем трейн данные
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(X_train)
X_train_tfidf = tfidf.transform(X_train)

#Векторизуем тест данные
X_test_tfidf = tfidf.transform(X_test)



#endregion
#region Создание и обучение моделей (будем сравнивать)

#Алгоритм наивного байеса
nb = MultinomialNB()
#Алгоритм логистической регрессии
log_model = LogisticRegression()
#Алгоритм опорных векторов
rbf_svc = SVC()

#Тренируем байеса
nb.fit(X_train_tfidf,y_train)
#Тренируем логист регрессию
log_model.fit(X_train_tfidf,y_train)
#Тренируем модель опорных векторов
rbf_svc.fit(X_train_tfidf,y_train)

#endregion
#region Сравнение

report(nb,X_test_tfidf)

#endregion
#region Создание Pipeline

#Создаем пайплайн
pipe = Pipeline([('tfidf',TfidfVectorizer())
                 ,('np',MultinomialNB())])

#Обучаем уже на всех данных
pipe.fit(X,y)

#Проверка
res = pipe.predict(['good flight'])

#endregion



print(res)
plt.xticks(rotation=45,fontsize=8)
plt.show()