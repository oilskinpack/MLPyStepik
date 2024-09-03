import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



#Открываем файл с описанием колонок
with open(r'D:\Khabarov\Курс ML\DATA\Ames_Housing_Feature_Description.txt','r') as f:
    print(f.read())

#Загружаем датафрейм
#          PID  MS SubClass MS Zoning  ...  Sale Type  Sale Condition SalePrice
# 0  526301100           20        RL  ...        WD           Normal    215000
df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\Ames_outliers_removed.csv')
res = df.head(10)

#Удаляем колонку с идентификатором
df = df.drop('PID',axis=1)

#Смотрим в какой колонке какое количество null значений
# MS SubClass         0
# MS Zoning           0
# Lot Frontage      490
# Lot Area            0
# Street              0
#                  ...
# Mo Sold             0
# Yr Sold             0
# Sale Type           0
# Sale Condition      0
# SalePrice           0
res = df.isnull().sum()



#Посчитаем процент отсутствующих значений
# MS SubClass        0.00000
# MS Zoning          0.00000
# Lot Frontage      16.74069
res = 100 * df.isnull().sum() / len(df)

#Переведем в функцию
def percent_missing(df):
    res = res = 100 * df.isnull().sum() / len(df)
    res = res[res>0].sort_values()
    return res
def show_percent_missing(df):
    percent_nan = percent_missing(df)
    sns.barplot(x=percent_nan.index, y=percent_nan,palette='magma')
    plt.xticks(rotation=90)
    plt.show()
    plt.clf()

#Получаем список параметров и процент nan Для них
# Electrical         0.034165
# Garage Cars        0.034165
# ...
# Misc Feature      96.412709
# Pool QC           99.590024
percent_nan = percent_missing(df)
res = percent_nan

#Визуализация
# sns.barplot(x=percent_nan.index,y=percent_nan)
# plt.xticks(rotation=90)
# plt.ylim(0,1)


#Ищем признаки, у которых nan меньше 1%
# Electrical         1.0
# Garage Cars        1.0
# BsmtFin SF 1       1.0
# Garage Area        1.0
# BsmtFin SF 2       1.0
# Bsmt Unf SF        1.0
# Total Bsmt SF      1.0
# Bsmt Half Bath     2.0
# Bsmt Full Bath     2.0
# Mas Vnr Area      23.0
res = percent_nan[percent_nan < 1] / (100/len(df))

#Удалим строки с пустым значением колонок Electrical и Garage Cars И смотрим новый процент признаков
df = df.dropna(axis=0,subset=['Electrical','Garage Cars'])
percent_nan = percent_missing(df)

# Bsmt Unf SF       0.034188
# Total Bsmt SF     0.034188
# BsmtFin SF 2      0.034188
# BsmtFin SF 1      0.034188
# Bsmt Full Bath    0.068376
# Bsmt Half Bath    0.068376
# Mas Vnr Area      0.786325
res = percent_nan[percent_nan < 1]


#Замещение nan числовых строк на 0
num_cols = ['BsmtFin SF 1','BsmtFin SF 2','Bsmt Unf SF','Total Bsmt SF','Bsmt Full Bath','Bsmt Half Bath']
df[num_cols] = df[num_cols].fillna(0)

#Замещаем nan для строковых колонок на None
str_cols = ['Bsmt Qual','Bsmt Cond','Bsmt Exposure','BsmtFin Type 1','BsmtFin Type 2']
df[str_cols] = df[str_cols].fillna('None')

#Смотрим результат
percent_nan = percent_missing(df)
res = percent_nan
# sns.barplot(x=percent_nan.index,y=percent_nan)
# plt.xticks(rotation=90)
# plt.ylim(0,1)
# plt.show()

#Создаем заполнение еще для двух признаков
df['Mas Vnr Type'] = df['Mas Vnr Type'].fillna('None')
df['Mas Vnr Area'] = df['Mas Vnr Area'].fillna(0)
percent_nan = percent_missing(df)
# plt.close()
# sns.barplot(x=percent_nan.index,y=percent_nan)
# plt.xticks(rotation=90)
# plt.show()


#Категориальные характеристики про гараж, где пропущено много значений
gar_str_cols = ['Garage Type','Garage Finish','Garage Qual','Garage Cond']
df[gar_str_cols] = df[gar_str_cols].fillna('None')

#Числовые характеристики про гараж,где пропущено много значений
df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(0)

#Удаляем колонки, где пропущено много значений
df = df.drop(['Pool QC','Misc Feature','Alley','Fence'],axis=1)

#Заменяем еще одну строку
df['Fireplace Qu'] = df['Fireplace Qu'].fillna('None')
percent_nan = percent_missing(df)

#Анализируем колонку Lot Frontage
# sns.boxplot(x='Lot Frontage',y='Neighborhood',data=df,orient='h')
# plt.show()

def changeToMean(value):
    res = value.fillna(value.mean())
    return res

# res = df ['Lot Frontage']
res  = df.groupby('Neighborhood')['Lot Frontage'].transform(changeToMean)

df['Lot Frontage'] = df['Lot Frontage'].fillna(0)
percent_nan = percent_missing(df)
res = percent_nan

print(res)