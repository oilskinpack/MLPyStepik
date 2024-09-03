import numpy as np
import pandas as pd

#encoding=ISO-8859-1 - если не грузится англ язык
#encoding='cp1251' - если не грузится русский язык

df = pd.read_csv(r'D:\Khabarov\RVT\ЮК11\UKV_GP11_Квартирография_СЦО_29.05.2024_10.20.csv',sep=';',encoding='cp1251')

def sumByObject(areaStr):
    area =  float(str(areaStr).replace(',','.'))
    return area
    

#res = df[df['Дуплекс'] == 1] ['Наименование помещениия']
res = df['Общая площадь, кв.м.'].apply(sumByObject).sum()

print(res)