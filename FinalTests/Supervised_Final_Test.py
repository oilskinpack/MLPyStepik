import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 300)
res = ''



#region Постановка задачи

# Необходимо провести анализ отток клиентов для провайдера интернет-услуг и телефонии

#Задача: Определить, что человек в ближайшее время уйдет в отток

#endregion
#region Загрузка и просмотр данных

df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\Telco-Customer-Churn.csv')


#  #   Column            Non-Null Count  Dtype
# ---  ------            --------------  -----
#  0   customerID        7032 non-null   object
#  1   gender            7032 non-null   object
#  2   SeniorCitizen     7032 non-null   int64
#  3   Partner           7032 non-null   object
#  4   Dependents        7032 non-null   object
#  5   tenure            7032 non-null   int64
#  6   PhoneService      7032 non-null   object
#  7   MultipleLines     7032 non-null   object
#  8   InternetService   7032 non-null   object
#  9   OnlineSecurity    7032 non-null   object
#  10  OnlineBackup      7032 non-null   object
#  11  DeviceProtection  7032 non-null   object
#  12  TechSupport       7032 non-null   object
#  13  StreamingTV       7032 non-null   object
#  14  StreamingMovies   7032 non-null   object
#  15  Contract          7032 non-null   object
#  16  PaperlessBilling  7032 non-null   object
#  17  PaymentMethod     7032 non-null   object
#  18  MonthlyCharges    7032 non-null   float64
#  19  TotalCharges      7032 non-null   float64
#  20  Churn             7032 non-null   object
res = df.info()


#    customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService     MultipleLines InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling              PaymentMethod  MonthlyCharges  TotalCharges Churn
# 0  7590-VHVEG  Female              0     Yes         No       1           No  No phone service             DSL             No          Yes               No          No          No              No  Month-to-month              Yes           Electronic check           29.85         29.85    No
# 1  5575-GNVDE    Male              0      No         No      34          Yes                No             DSL            Yes           No              Yes          No          No              No        One year               No               Mailed check           56.95       1889.50    No
# 2  3668-QPYBK    Male              0      No         No       2          Yes                No             DSL            Yes          Yes               No          No          No              No  Month-to-month              Yes               Mailed check           53.85        108.15   Yes
# 3  7795-CFOCW    Male              0      No         No      45           No  No phone service             DSL            Yes           No              Yes         Yes          No              No        One year               No  Bank transfer (automatic)           42.30       1840.75    No
# 4  9237-HQITU  Female              0      No         No       2          Yes                No     Fiber optic             No           No               No          No          No              No  Month-to-month              Yes           Electronic check           70.70        151.65   Yes
res = df.head()



# Churn
# No     5163
# Yes    1869
res = df['Churn'].value_counts()

#        SeniorCitizen       tenure  MonthlyCharges  TotalCharges
# count    7032.000000  7032.000000     7032.000000   7032.000000
# mean        0.162400    32.421786       64.798208   2283.300441
# std         0.368844    24.545260       30.085974   2266.771362
# min         0.000000     1.000000       18.250000     18.800000
# 25%         0.000000     9.000000       35.587500    401.450000
# 50%         0.000000    29.000000       70.350000   1397.475000
# 75%         0.000000    55.000000       89.862500   3794.737500
# max         1.000000    72.000000      118.750000   8684.800000
res = df.describe()

#endregion


#region Исследование и подготовка данных

#region Проверка на nan
# customerID          0
# gender              0
# SeniorCitizen       0
# Partner             0
# Dependents          0
# tenure              0
# PhoneService        0
# MultipleLines       0
# InternetService     0
# OnlineSecurity      0
# OnlineBackup        0
# DeviceProtection    0
# TechSupport         0
# StreamingTV         0
# StreamingMovies     0
# Contract            0
# PaperlessBilling    0
# PaymentMethod       0
# MonthlyCharges      0
# TotalCharges        0
# Churn               0
res = df.isnull().sum()
#endregion
#region Проверка баланса значений целевой переменной

# sns.countplot(data=df,x='Churn')

#endregion
#region Распределение колонки TotalCharges (общей суммы расходов) в зависимости от оттока

#Через виолинплот - оптимал
# sns.violinplot(data=df,y='TotalCharges',x='Churn',palette='Set2')

#Через боксплот
# sns.boxplot(data=df,y='TotalCharges',x='Churn',palette='Set2')

#Через свамплот - не стоит, оч много точек
# sns.swarmplot(data=df,y='TotalCharges',x='Churn',palette='Set2',size=1)


#endregion
#region Распределение колонки TotalCharges для различных типов контрактов в зависимости от оттока

# sns.boxplot(data=df,x='Contract',y='TotalCharges',hue='Churn',palette='Set2')

#endregion
#region Корреляция категориальных колонок с целевой переменной

cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines',
     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'InternetService',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod','Churn']
df_dummy = df[cols]
df_dummy = pd.get_dummies(df_dummy,drop_first=False,dtype=int)


df_dummy = df_dummy.corr().sort_values('Churn_Yes',ascending=True).reset_index()
res = df_dummy
# sns.barplot(data=df_dummy,y='Churn_Yes',x='index',palette='Set2')
# plt.xticks(rotation=90)

#endregion


#endregion
#region Анализ оттока

#region Просмотр возможных типов контрактов

# Contract
# Month-to-month    3875
# Two year          1685
# One year          1472
res = df['Contract'].value_counts()

#endregion
#region Гистограмма по колонке tenure - сколько месяцев являлся абонетном

# sns.displot(data=df,x='tenure',bins=60)

#endregion
#region Гистограмма по колонке tenure - c разбитием по Churn и Contract

# plt.figure(figsize=(10,3),dpi=200)
# sns.displot(data=df,x='tenure',bins=70,col='Contract',row='Churn')

#endregion
#region Диаграмма с данными по Total Charges и Monthly Charges

# sns.scatterplot(data=df,x='MonthlyCharges',y='TotalCharges',hue='Churn',palette='Set2')

#endregion
#region Процент оттока людей для каждого tenure

# no_churn = df.groupby(['Churn','tenure']).count().transpose()['No']
no_churn = df.groupby(['Churn','tenure']).count().transpose()['No']
yes_churn = df.groupby(['Churn','tenure']).count().transpose()['Yes']

churn_rate = (yes_churn)/(no_churn + yes_churn) * 100
churn_rate = churn_rate.transpose()['customerID']

#endregion
#region График процента оттока в зависимости от tenure (кол-ва месяцев)

# churn_rate.plot()
# plt.ylabel('Кол-во месяцев')

#endregion
#region Построение более крупных когорт

def cohorting(month_total):
    res = ''
    if(month_total <= 12):
        res = '0-12 Months'
    elif(month_total <= 24):
        res = '13-24 Months'
    elif (month_total <= 48):
        res = '25-48 Months'
    else:
        res = 'Over 48 Months'
    return  res

df['Tenure Cohort'] = df['tenure'].apply(cohorting)

#endregion
#region Диаграмма с данными по Total Charges и Monthly Charges c разбивкой по большим когортам

# sns.scatterplot(data=df,x='MonthlyCharges',y='TotalCharges',hue='Tenure Cohort',palette='Set2')

#endregion
#region Проверка процента ушедших в когорте

# sns.countplot(data=df,x='Tenure Cohort',hue='Churn',palette='Set2')

#endregion
#region Подсчет количества людей в оттоке с разбиквой по когортам и типу контракта

# plt.figure(figsize=(10,3),dpi=200)
sns.catplot(data=df,x='Tenure Cohort',hue='Churn',col='Contract',kind='count')

#endregion

#endregion




print(res)
plt.show()