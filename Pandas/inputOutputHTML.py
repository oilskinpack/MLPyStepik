import numpy as np
import pandas as pd

res = ''

#1.1 Загружаем таблицы со страницы сайта
url = 'https://en.wikipedia.org/wiki/World_population'
tables = pd.read_html(url)
df = tables[4]
res = df

#1.2 Проверка на мультииндекс (если таблица многоуровневая)
# Тут будет мультииндекс или обычные колонки
res = tables[4].columns

#1.3 Убираем мультииндекс, чтобы работать чисто с данными
#df = df['NameOfMultiIndex']

#1.4 Удаляем лишнее
res = df.drop('Source (official or from the United Nations)',axis=1)

#1.5 Сохраняем без индекса
df.to_html(r'D:\Khabarov\Курс ML\03-Pandas\myHTML.html',index=False)


print(res)