import numpy as np
import pandas as pd

res = ''

#1.1 Читаем данные - по умолчанию открывает первый лист
#ИНОГДА стоит добавить параметр engine = 'openpyxl'
df = pd.read_excel(r'D:\Khabarov\Курс ML\03-Pandas\my_excel_file.xlsx')
res = df

#1.2 Явно указываем лист
df = pd.read_excel(r'D:\Khabarov\Курс ML\03-Pandas\my_excel_file.xlsx',sheet_name='Second_Sheet')

#1.3 Узнаем какие листы есть - ['First_Sheet', 'Second_Sheet']
res = pd.ExcelFile(r'D:\Khabarov\Курс ML\03-Pandas\my_excel_file.xlsx').sheet_names

#1.4 Сохранение файла
res = df.to_excel(r'D:\Khabarov\Курс ML\03-Pandas\myExcel.xlsx',sheet_name='First_Sheet')

print(res)