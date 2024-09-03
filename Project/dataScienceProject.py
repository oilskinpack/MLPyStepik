import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

#Вы планируете посмотреть какой-то фильм. Можете ли Вы доверять онлайн-рейтингам и отзывам о фильмах?
#Особенно если та компания, которая занимается рейтингами и отзывами, также зарабатывает на продаже билетов на фильмы.
#Есть ли у таких компаний тенденция выдавать завышенные или заниженные рейтинги?

#Цель проекта - выполнить шаги по мотивам статьи на сайте fivethirtyeight.com о рейтингах и посмотреть,
#сможем ли мы прийти к тем же выводам, которые приведены в статье.
#Вы будете применять Ваши навыки работы с pandas и навыки визуализации данных для того,
#чтобы определить, предоставляла ли компания Fandango завышенные рейтинги в 2015 году для того, чтобы продавать больше билетов.

#=====Часть 1: Исследуем данные======
#Файл all_sites_scores.csv
#Колонка	Определение
#FILM	Название фильма
#RottenTomatoes	Оценка "Rotten Tomatoes Tomatometer" для этого фильма
#RottenTomatoes_User	Оценка "Rotten Tomatoes user" для этого фильма
#Metacritic	Оценка "Metacritic" для этого фильма
#Metacritic_User	Оценка "Metacritic user" для этого фильма
#IMDB	Оценка "IMDb user" для этого фильма
#Metacritic_user_vote_count	Количество голосов за этот фильм от пользователей Metacritic
#IMDB_user_vote_count	Количество голосов за этот фильм от пользователей IMDb

#fandango_scrape.csv
#Колонка	Определение
#FILM	Название фильма
#STARS	Количество звёзд на Fandango.com
#RATING	Рейтинг Fandango - значение, прочитанное с HTML-страницы. Это средний рейтинг фильма.
#VOTES	Количество голосов пользователей, которые написали отзыв о фильме (на момент выгрузки данных).

#Загружаем данные остальных сайтов
dfOther = pd.read_csv(r'D:\Khabarov\Курс ML\06-Capstone-Project\all_sites_scores.csv')
#   Column                      Non-Null Count  Dtype
# 0   FILM                        146 non-null    object
# 1   RottenTomatoes              146 non-null    int64
# 2   RottenTomatoes_User         146 non-null    int64
# 3   Metacritic                  146 non-null    int64
# 4   Metacritic_User             146 non-null    float64
# 5   IMDB                        146 non-null    float64
# 6   Metacritic_user_vote_count  146 non-null    int64
# 7   IMDB_user_vote_count        146 non-null    int64

#Загружаем данные фандарго
dfFandargo = pd.read_csv(r'D:\Khabarov\Курс ML\06-Capstone-Project\fandango_scrape.csv')
 #   Column  Non-Null Count  Dtype
#  0   FILM    504 non-null    object
#  1   STARS   504 non-null    float64
#  2   RATING  504 non-null    float64
#  3   VOTES   504 non-null    int64


# ЗАДАНИЕ: Давайте изучим связь между популярностью фильма и его рейтингом. 
# Нарисуйте график scatterplot, показывающий связь между колонками RATING и VOTES.
# Можете поменять стилизацию графика по Вашему вкусу.
plt.figure(figsize=[4,2.5],dpi=200)
#sns.scatterplot(data=dfFandargo,x='RATING',y='VOTES')

# ЗАДАНИЕ: Вычислите корреляцию между колонками:
correl = dfFandargo.corr(numeric_only=True)

#Создайте новую колонку, в ней возьмите из строки FILM только год, и назовите эту новую колонку YEAR
def findYear(movieName):
    res = ''
    parts = str(movieName).split(sep='(')
    for part in parts:
        if (')' in part and part[0:2] in ['19','20']):
            res = part.split(sep=')')[0]
        else:
            continue
    return res
    

#ЗАДАНИЕ: Сделаем предположение, что каждая строка в колонке FILM содержит значение в следующем формате:
dfFandargo['YEAR'] = dfFandargo['FILM'].apply(findYear)

#ЗАДАНИЕ: Сколько фильмов содержится в наборе данных Fandango, в разбивке по годам?
# YEAR
# 2015    478
# 2014     23
# 2016      1
# 1964      1
# 2012      1
#res = dfFandargo['YEAR'].value_counts()
#sns.countplot(data=dfFandargo,x='YEAR')

#ЗАДАНИЕ: Какие 10 фильмов получили наибольшее количество голосов (votes)?
#                                                FILM  STARS  RATING  VOTES  YEAR
# 0                       Fifty Shades of Grey (2015)    4.0     3.9  34846  2015
# 1                             Jurassic World (2015)    4.5     4.5  34390  2015
# 2                            American Sniper (2015)    5.0     4.8  34085  2015
# 3                                  Furious 7 (2015)    5.0     4.8  33538  2015
# 4                                 Inside Out (2015)    4.5     4.5  15749  2015
# 5  The Hobbit: The Battle of the Five Armies (2014)    4.5     4.3  15337  2014
# 6               Kingsman: The Secret Service (2015)    4.5     4.2  15205  2015
# 7                                    Minions (2015)    4.0     4.0  14998  2015
# 8                    Avengers: Age of Ultron (2015)    5.0     4.5  14846  2015
# 9                             Into the Woods (2014)    3.5     3.4  13055  2014
#res = dfFandargo.nlargest(columns='VOTES',n=10)

# ЗАДАНИЕ: Сколько фильмов имеет нулевое количество голосов (votes)?
res = len(dfFandargo[dfFandargo['VOTES'] == 0])

#ЗАДАНИЕ: Создайте DataFrame только с теми фильмами, которые имеют голоса (votes) - то есть, удалите те фильмы, у которых нет ни одного голоса.
dfFandargo = dfFandargo[ dfFandargo['VOTES'] != 0]
res = len(dfFandargo)

#ЗАДАНИЕ: Создайте график KDE plot (или несколько таких графиков), который отображает распределение отображаемых рейтингов (STARS)
#  и истинных рейтингов на основе голосов пользователей (RATING).
#  Обрежьте диапазон KDE в пределах 0-5.
# sns.kdeplot(data=dfFandargo,x='RATING',clip=[0,5],fill=True,label='True Rating')
# sns.kdeplot(data=dfFandargo,x='STARS',clip=[0,5],fill=True,label='Stars Displayed')


# ЗАДАНИЕ: Теперь давайте посчитаем эту разницу в численном виде;
# Создайте новую колонку, в которой сохраните разницу между колонками STARS и RATING с помощью обычного вычитания STARS-RATING,
# а также выполните округление до одной десятичной цифры после запятой
dfFandargo ['STARS_DIFF'] = round((dfFandargo['STARS'] - dfFandargo['RATING']),1)
res = dfFandargo

# ЗАДАНИЕ: Нарисуйте график count plot для отображения того, сколько раз встречается то или иное значение разницы между STAR и RATING:
#sns.countplot(data=dfFandargo,x='STARS_DIFF',palette='magma')


# ЗАДАНИЕ: На этом графике мы видим, что один из фильмов имеет разницу в 1 звезду между отображаемым рейтингом и истинным рейтингом! Найдите этот фильм.
# Turbo Kid (2015)
res = dfFandargo[dfFandargo['STARS_DIFF'] == 1] ['FILM']

# ЗАДАНИЕ: Нарисуйте график scatterplot, изображающий для Rotten Tomatoes связь между рейтингами от критиков и пользовательскими рейтингами.
# sns.scatterplot(data=dfOther,x='RottenTomatoes',y='RottenTomatoes_User')

# ЗАДАНИЕ: Создайте новую колонку, в которой сохраните разницу между рейтингом от критиков и пользовательским рейтингом для Rotten Tomatoes.
# Используйте формулу RottenTomatoes-RottenTomatoes_User.
# 15.095890410958905
dfOther ['Rotten_Diff'] = dfOther['RottenTomatoes'] - dfOther['RottenTomatoes_User']
dfOther ['Rotten_Diff_Mdl'] = (dfOther['RottenTomatoes'] - dfOther['RottenTomatoes_User']).abs()
res = abs(dfOther['Rotten_Diff'] ).mean()


#ЗАДАНИЕ: Нарисуйте график распределения разницы между рейтингами от критиков и пользовательскими рейтингами для Rotten Tomatoes.
#  На этом графике будут отрицательные значения.
#  Для отображения этого распределения можете использовать как KDE, так и гистограммы.
# sns.displot(data=dfOther,x='Rotten_Diff_Mdl',kde=True,bins=25,label='Abs Diff')

# ЗАДАНИЕ: Какие 5 фильмов в среднем были выше всего оценены пользователями, по сравнению с оценками от критиков:
#                           FILM  RottenTomatoes  RottenTomatoes_User  Metacritic  Metacritic_User  IMDB  Metacritic_user_vote_count  IMDB_user_vote_count  Rotten_Diff  Rotten_Diff_Mdl
# 3       Do You Believe? (2015)              18                   84          22              4.7   5.4                          31                  3136          -66               66  
# 85           Little Boy (2015)              20                   81          30              5.9   7.4                          38                  5927          -61               61  
# 105    Hitman: Agent 47 (2015)               7                   49          28              3.3   5.9                          67                  4260          -42               42  
# 134    The Longest Ride (2015)              31                   73          33              4.8   7.2                          49                 25214          -42               42  
# 125  The Wedding Ringer (2015)              27                   66          35              3.3   6.7                         126                 37292          -39               39 
res = dfOther.nsmallest(n=5,columns='Rotten_Diff')



#                                  FILM  RottenTomatoes  RottenTomatoes_User  Metacritic  ...  Metacritic_user_vote_count  IMDB_user_vote_count  Rotten_Diff  Rotten_Diff_Mdl
# 69                  Mr. Turner (2014)              98                   56          94  ...                          98                 13296           42               42
# 112                 It Follows (2015)              96                   65          83  ...                         551                 64656           31               31
# 115          While We're Young (2015)              83                   52          76  ...                          65                 17647           31               31
# 37               Welcome to Me (2015)              71                   47          67  ...                          33                  8301           24               24
# 40   I'll See You In My Dreams (2015)              94                   70          75  ...                          14                  1151           24               24
res = dfOther.nlargest(n=5,columns='Rotten_Diff')

# ЗАДАНИЕ: Нарисуйте график scatterplot для сравнения рейтингов Metacritic и Metacritic User.
#sns.scatterplot(data=dfOther,x='Metacritic',y='Metacritic_User')

#ЗАДАНИЕ: Нарисуйте график scatterplot для изображения связи между количеством голосов на MetaCritic и количеством голосов на IMDB.
# sns.scatterplot(data=dfOther,x='Metacritic_user_vote_count',y='IMDB_user_vote_count')

# ЗАДАНИЕ: Какой фильм получил наибольшее количество голосов на IMDB?
#                          FILM  RottenTomatoes  RottenTomatoes_User  Metacritic  Metacritic_User  IMDB  Metacritic_user_vote_count  IMDB_user_vote_count  Rotten_Diff  Rotten_Diff_Mdl
# 14  The Imitation Game (2014)              90                   92          73              8.2   8.1                         566                334164           -2                2 
res = dfOther.nlargest(n=1,columns='IMDB_user_vote_count')

# ЗАДАНИЕ: Какой фильм получил наибольшее количество голосов на Metacritic?
res = dfOther.nlargest(n=1,columns='Metacritic_user_vote_count')

# ЗАДАНИЕ: Объедините таблицу fandango с таблицей all_sites. Не каждый фильм в таблице Fandango найдётся в таблице all_sites,
#  потому что некоторые фильмы Fandango имеют очень мало отзывов или вообще не имеют отзывов. Но мы хотим сравнивать только те фильмы,
#  которые есть в обеих таблицах. Поэтому сделайте объединение "inner merge" двух наборов данных, сопоставляя строки по колонке FILM.
dfUnion = pd.merge(left=dfFandargo,right=dfOther,how='inner',on='FILM')
res = dfUnion

# ЗАДАНИЕ: Добавьте дополнительные колонки в all_sites, и запишите в них нормализованные значения рейтингов от 0 до 5. Это можно сделать разными способами.
dfUnion['RT_Norm'] = round(dfUnion['RottenTomatoes'] / 20,1)
dfUnion['RTU_Norm'] = round(dfUnion['RottenTomatoes_User'] / 20,1)
dfUnion['Meta_Norm'] = round(dfUnion['Metacritic'] / 20,1)
dfUnion['Meta_U_Norm'] = round(dfUnion['Metacritic_User'] / 2,1)
dfUnion['IMDB_Norm'] = round(dfUnion['IMDB'] / 2,1)


# ЗАДАНИЕ: Далее создайте DataFrame с названием norm_scores, в котором оставьте только нормализованные рейтинги. 
# Также оставьте колонки STARS и RATING из исходной таблицы fandango.
norm_scores = dfUnion[['STARS','RATING','RT_Norm','RTU_Norm','Meta_Norm','Meta_U_Norm','IMDB_Norm']]


# ЗАДАНИЕ: Нарисуйте график, сравнивающий распределения нормализованных рейтингов от всех компаний.
#  Это можно сделать разными способами, но попробуйте использовать Seaborn KDEplot (посмотрите документацию).
#  Не беспокойтесь, если Ваш график будет выглядеть немного иначе нашего примера.
#  Главное, чтобы были видны отличия между отдельными распределениями
sns.kdeplot(data=norm_scores,fill=True,clip=[0,5],legend=True)


print(res)
plt.title('График')
# plt.legend(loc=(0.01,0.5))
plt.show()