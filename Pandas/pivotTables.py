import numpy as np
import pandas as pd

res = ''

df = pd.read_csv(r'D:\Khabarov\Курс ML\03-Pandas\Sales_Funnel_CRM.csv')
res = df


#1.1 Берем колонки, которые нам понадобятся
#         Company          Product  Licenses
#0         Google        Analytics       150
#1         Google       Prediction       150
#2         Google         Tracking       300
#3           BOBO        Analytics       150
#4           IKEA        Analytics       300
#5     Tesla Inc.        Analytics       300
#6     Tesla Inc.       Prediction       150
licenses = df[['Company','Product','Licenses']]
res = licenses


#1.2 Используем pivot
res = pd.pivot(data=licenses,index='Company',columns='Product',values='Licenses')



print(res)