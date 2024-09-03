import numpy as np
import pandas as pd

res = ''

#Загружаем данные
hotels = pd.read_csv(r'D:\Khabarov\Курс ML\03-Pandas\hotel_booking_data.csv')
res = hotels.head()
res = hotels.info()

#1.1 Количество строк - 119390
res = hotels.__len__()

#1.2 Количество пропущенных строк - 112593
res =hotels.isnull().sum()

#1.3 Удаление Company из набора данных
res = hotels.drop('company',axis=1)

#1.4 Пять стран, которые встречаются чаще всего
res = hotels['country'].value_counts()[:5]

#1.5 Выведите имя человека, который заплатил наибольшую сумму за одни сутки (ADR - average daily rate)? Какое было значение этого ADR
res = hotels.columns
res = hotels.iloc[hotels['adr'].idxmax()] [['name','adr']] 

#1.6 adr это средняя сумма за сутки для одного бронирования. Каково будет среднее значение adr для всех бронирований в этом наборе данных
res = hotels['adr'].mean()

#1.7 Каково среднее (mean) количество ночей, если сделать усреднение по всем бронированиям? Можете округлить до двух знаков после запятой
res = (hotels['stays_in_week_nights'] + hotels['stays_in_weekend_nights']).mean().__round__(2)

#1.8 Выведите имена и адреса email тех, кто сделал ровно 5 специальных запросов ("Special Requests")?
res = hotels[hotels['total_of_special_requests'] == 5] [['name','email']]

#1.9 Какой процент бронирований классифицированы как бронирования "повторными гостями"? (Для этого не следует использовать имя, воспользуйтесь колонкой is_repeated_guest)
res = (hotels[hotels['is_repeated_guest'] == 1].__len__() / hotels.__len__()) * 100

#1.10 Какие 5 фамилий встречаются в этом наборе данных наиболее часто? 
res = hotels['name'].str.split(' ',expand=True) [1].value_counts()

#1.11 Выведите имена людей, которые бронировали номер для наибольшего количества детей и младенцев?
hotels ['all_children'] = hotels['children'] + hotels['babies']
res = hotels.nlargest(3,'all_children') ['name']

#1.12 Какие 3 кода области (это первые три цифры телефона) встречаются в наборе данных наиболее часто?
res = hotels['phone-number'].str.split('-',expand=True)[0].value_counts()

#1.13 Посчитайте количество заселений в отель между 1м и 15м числами месяца (включая 1 и 15)?
res = hotels[hotels['arrival_date_day_of_month'] <= 15].__len__()

#1.14  Создайте таблицу, запишите в неё количество заселений в отель в тот или иной день недели
daysDf = hotels[['arrival_date_day_of_month','arrival_date_month','arrival_date_year']]
daysDf ['dayTime'] = (hotels['arrival_date_day_of_month'].astype(str) +' '+ hotels['arrival_date_month'].astype(str) + ' '+ hotels['arrival_date_year'].astype(str))
daysDf ['dateTime_conv'] = pd.to_datetime(daysDf ['dayTime'])
res = daysDf['dateTime_conv']. dt.day_name ().value_counts()

print(res)