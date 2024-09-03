import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#Загружаем датафрейм
#    MS SubClass MS Zoning  Lot Frontage  ...  Sale Type Sale Condition SalePrice
# 0           20        RL         141.0  ...        WD          Normal    215000
# 1           20        RH          80.0  ...        WD          Normal    105000
# 2           20        RL          81.0  ...        WD          Normal    172000
df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\Ames_NO_Missing_Data.csv')
res = df.head(3)

#Меняем числовые значения на строковые (там где это нужно)
df['MS SubClass'] = df['MS SubClass'].astype(str)

#Принцип создания dummy переменных
#       Up
# 0   True
# 1   True
# 2  False
direction = pd.Series(['Up','Up','Down'])
# res = pd.get_dummies(direction,drop_first=True)


#Делим датафрейм на числовые и категориальные колонки
my_object_df = df.select_dtypes(include='object')
my_numeric_df = df.select_dtypes(exclude='object')

#Преобразуем категориальные колонки в dummy переменные
df_objects_dummies = pd.get_dummies(my_object_df,drop_first=True,dtype=int)

#Добавляем к dummy датафрейму наши числовые колонки
final_df = pd.concat([my_numeric_df,df_objects_dummies],axis=1)
res = final_df

#Корреляции
# Garage Cars           0.648488
# Total Bsmt SF         0.660983
# Gr Liv Area           0.727279
# Overall Qual          0.802637
# SalePrice             1.000000
res = final_df.corr() ['SalePrice'].sort_values()


print(res)