import numpy as np
import pandas as pd

res = ''

#Загрузка данных
df = pd.read_csv(r'D:\Khabarov\Курс ML\03-Pandas\tips.csv')

#1.1 Создаем функцию для apply() - функция берет номер карты и возвращает последние 4 цифры
def last_four(num):
    return str(num)[-4:]

#1.2 - получаем серию с последними четырьмя цифрами карт
res = df['CC Number'].apply(last_four)

#1.3 - добавляем эту серию в наш датафрейм
#     total_bill   tip     sex smoker   day    time  size  price_per_person          Payer Name         CC Number Payment ID last_four
#0         16.99  1.01  Female     No   Sun  Dinner     2              8.49  Christy Cunningham  3560325168603410    Sun2959      3410
df['last_four'] = df['CC Number'].apply(last_four)
res = df

#2.1 Пишем функцию для градации цены ресторана по чеку
def yelp(price):
    if price < 10:
        return '$'
    elif price >= 10 and price < 30:
        return '$$'
    else:
        return 'SSS'
    
#2.2 Применяем
#     total_bill   tip     sex smoker   day    time  size  price_per_person          Payer Name         CC Number Payment ID last_four yelp
#0         16.99  1.01  Female     No   Sun  Dinner     2              8.49  Christy Cunningham  3560325168603410    Sun2959      3410   $$    
#1         10.34  1.66    Male     No   Sun  Dinner     3              3.45      Douglas Tucker  4478071379779230    Sun4608      9230   $$
df['yelp'] = df['total_bill'].apply(yelp)
res = df

#3.1 Использование лямбда выражений в методе apply
#Cначала пример лямбда выражения - увеличивает чек на 2
#lambda bill : bill*2
#Использование
res = df[ 'total_bill' ].apply(lambda bill: bill*2)

#3.2 Бродкаст по колонке функцией в которой более 1 параметра
#Пишем функцию - она принимает общий чек и чаевые, высчитывает Щедрые чаевые или Обычные
def quality(total_bill,tip):
    if tip/total_bill > 0.25:
        return 'Щедрые чаевые'
    else:
        return "Обычные чаевые"
#Используем нашу функцию
#df ['quality'] = df[ ['total_bill','tip'] ].apply (lambda df: quality (df['total_bill'],df['tip']), axis=1  )
res = df

#.3.3 Векторизируем функцию
df ['quality'] = np.vectorize(quality) (df['total_bill'],df['tip'])


#4.1 Сортировка данных - по возрастанию
res = df.sort_values('tip')

#4.2 Сортировка данных - по убыванию
res = df.sort_values('tip',ascending=False)

#4.3 Сортировка данных по двум и более колонкам
res = df.sort_values( ['tip','size'] )

#4.4 Получение максимального значения
res = df['total_bill'].max()

#4.5 Получение индекса элемента, у которого макс значение - 170
res = df['total_bill'].idxmax()

#4.6 Все значения айтема по индексу
res = df.iloc[170]

#4.7 Матрица корреляций между колонками - можно только числовые
#                  total_bill       tip      size  price_per_person  CC Number
#total_bill          1.000000  0.675734  0.598315          0.647554   0.104576
#tip                 0.675734  1.000000  0.489299          0.347405   0.110857
#size                0.598315  0.489299  1.000000         -0.175359  -0.030239
#price_per_person    0.647554  0.347405 -0.175359          1.000000   0.135240
#CC Number           0.104576  0.110857 -0.030239          0.135240   1.000000
res = df.corr(numeric_only=True)

#4.8 Количество строк для всех возможных значений
#sex
#Male      157
#Female     87
#Name: count, dtype: int64
res = df['sex'].value_counts()

#4.9 Список уникальных значений в колоке - ['Sun' 'Sat' 'Thur' 'Fri']
res = df['day'].unique()

#4.10 Замена в колонке одних значений на другие ЧЕРЕЗ REPLACE
#0      F
#1      M
res = df['sex'].replace( ['Female','Male'],['F','M'] )

#4.11 Замена через МАППИНГ
myMap = {'Female':'F','Male':'M'}
res = df['sex'].map(myMap)

#4.12 Проверка дубликатов - Была ли уже такая строка
#0      False
#1      False
res = df.duplicated()

#4.13 Удаление дубликатов
res = df.drop_duplicates()

#4.14 Проверка - попадает ли значение в диапазон
#0       True
#1       True
#2      False
res = df['total_bill'].between(10,20,inclusive='both')

#4.15 N строк с наибольшим значением нужной колонки
#     total_bill    tip   sex smoker  day    time  ...       Payer Name         CC Number Payment ID  last_four yelp         quality
#170       50.81  10.00  Male    Yes  Sat  Dinner  ...    Gregory Clark  5473850968388236    Sat1954       8236  SSS  Обычные чаевые
#212       48.33   9.00  Male     No  Sat  Dinner  ...  Alex Williamson      676218815212    Sat4590       5212  SSS  Обычные чаевые
#23        39.42   7.58  Male     No  Sat  Dinner  ...   Lance Peterson  3542584061609808     Sat239       9808  SSS  Обычные чаевые
res = df.nlargest(3,'tip')

#4.16 N строк с наименьшим значением нужной колонки
#     total_bill  tip     sex smoker  day    time  ...     Payer Name         CC Number Payment ID  last_four yelp         quality
#67         3.07  1.0  Female    Yes  Sat  Dinner  ...  Tiffany Brock  4359488526995267    Sat3455       5267    $   Щедрые чаевые
#92         5.75  1.0  Female    Yes  Fri  Dinner  ...   Leah Ramirez  3508911676966392    Fri3780       6392    $  Обычные чаевые
#111        7.25  1.0  Female     No  Sat  Dinner  ...    Terri Jones  3559221007826887    Sat4801       6887    $  Обычные чаевые
res = df.nsmallest(3,'tip')

#4.17 Получение рандомного среза для тестов - По количеству строк
res = df.sample(2)

#4.17 Получение рандомного среза для тестов - В процентах
res = df.sample(frac=0.1)



print(res)



