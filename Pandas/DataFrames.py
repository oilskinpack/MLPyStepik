import numpy as np
import pandas as pd

res = ''

#1 Создание Датафрейма
#Создаем значения
np.random.seed(101)
myData = np.random.randint(0,101,(4,3))
#Создаем индексы
myIndex = ['CA','NY','AZ','TX']
#Создаем колонки
myColumns = ['Jan','Feb','Mar']
#Создаем датафрейм:
#    Jan  Feb  Mar
#CA   95   11   81
#NY   70   63   87
#AZ   75    9   77
#TX   40    4   63
df = pd.DataFrame(data=myData,index=myIndex,columns=myColumns)
res = df

#2.1 Получение инфы по датафрейму:
#<class 'pandas.core.frame.DataFrame'>
#Index: 4 entries, CA to TX
#Data columns (total 3 columns):
# #   Column  Non-Null Count  Dtype
#---  ------  --------------  -----
# 0   Jan     4 non-null      int32
# 1   Feb     4 non-null      int32
# 2   Mar     4 non-null      int32
#dtypes: int32(3)
#memory usage: 80.0+ bytes
res = df.info()

#2.2 Загрузка csv файла
df = pd.read_csv(r'D:\Khabarov\Курс ML\03-Pandas\tips.csv')

#3.1 Список колонок
#Index(['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size',
#       'price_per_person', 'Payer Name', 'CC Number', 'Payment ID'],
#      dtype='object')
res = df.columns

#3.2 Просмотр индекса - RangeIndex(start=0, stop=244, step=1)
res = df.index

#3.3 Просмотр нескольких первых строк таблицы (по умолчанию 5)
res = df.head()

#3.4 Просмотр нескольких последних строк таблицы (по умолчанию 5)
res = df.tail()

#3.5 Статистика по колонкам
#       total_bill         tip        size  price_per_person     CC Number  - КОЛОНКИ
#count  244.000000  244.000000  244.000000        244.000000  2.440000e+02  - Количество значений
#mean    19.785943    2.998279    2.569672          7.888197  2.563496e+15  - Среднее
#std      8.902412    1.383638    0.951100          2.914234  2.369340e+15  - Среднеквадратичное отклонение
#min      3.070000    1.000000    1.000000          2.880000  6.040679e+10  - мин
#25%     13.347500    2.000000    2.000000          5.800000  3.040731e+13  - 25% всех данных меньше указанного числа
#50%     17.795000    2.900000    2.000000          7.255000  3.525318e+15  - 50% всех данных меньше указанного числа [МЕДИАНА]
#75%     24.127500    3.562500    3.000000          9.390000  4.553675e+15  - 75% всех данных меньше указанного числа
#max     50.810000   10.000000    6.000000         20.270000  6.596454e+15  - макс
res = df.describe()

#3.6 Транспонированная таблица со статистикой
res = df.describe().transpose()



#4.1 Получение колонки - колонка типа Series
res = df['total_bill']

#4.2 Получение двух колонок
res = df[['total_bill','tip']]

#4.3 Операция с колонками - процент чаевых от всего счета
res = 100 * (df['tip'] / df['total_bill'])

#4.4 Добавление новой колонки
df ['tip_percentage'] = 100 * (df['tip'] / df['total_bill'])

#4.5 Округление колонки
df ['tip_percentage'] = np.round( 100 * (df['tip'] / df['total_bill']),2)


#4.6 Удаление колонки - выдает новый датафрейм
res = df.drop('tip_percentage',axis=1)


#5.1 Получение индекса
res = df.index

#5.2 Сделаем индексом другую колонку
df = df.set_index('Payment ID')

#5.3 Сброс индекса
#res = df.reset_index()

#5.4 Получаем строку по индексу - получаем объект Series:
#total_bill                       16.99
#tip                               1.01
#sex                             Female
#smoker                              No
#day                                Sun
#time                            Dinner
#size                                 2
#price_per_person                  8.49
#Payer Name          Christy Cunningham
#CC Number             3560325168603410
#Payment ID                     Sun2959
#tip_percentage                    5.94
#Name: 0, dtype: object
#res = df.iloc[0]

#5.5 Получаем строку по наименованному индексу
res = df.loc['Sat1766']

#5.6 Получаем несколько строк:
#        total_bill   tip     sex smoker  day    time  size  price_per_person          Payer Name         CC Number  tip_percentage
#Payment ID
#Sun2959          16.99  1.01  Female     No  Sun  Dinner     2              8.49  Christy Cunningham  3560325168603410            5.94        
#Sun5260          23.68  3.31    Male     No  Sun  Dinner     2             11.84    Nathaniel Harris  4676137647685994           13.98    
res = df.loc[['Sun2959','Sun5260']]

#5.7 Удаление строки
res = df.drop('Sun4608', axis=0)

#5.8 Добавление строки
one_row = df.iloc[0]
res = pd.concat( [df , pd.DataFrame([one_row])] ,axis=0)

print(res)