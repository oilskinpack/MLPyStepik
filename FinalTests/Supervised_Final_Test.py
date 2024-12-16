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
sns.violinplot(data=df,y='TotalCharges',x='Churn',palette='Set2')

#Через боксплот
# sns.boxplot(data=df,y='TotalCharges',x='Churn',palette='Set2')

#Через свамплот - не стоит, оч много точек
# sns.swarmplot(data=df,y='TotalCharges',x='Churn',palette='Set2',size=1)


#endregion

#endregion




print(res)
plt.show()