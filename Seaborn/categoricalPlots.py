import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#df = pd.read_csv(r'D:\Khabarov\Курс ML\05-Seaborn\dm_office_sales.csv')
df = pd.read_csv(r'D:\Khabarov\Курс ML\05-Seaborn\StudentsPerformance.csv')

res = df.head()

#===CountPlot===
#Здесь можем выставить комфортные значения figure
plt.figure(figsize=(10,4),dpi=200)

#Здесь создаем график
#sns.countplot(data=df,x='level of education',palette='Set2',hue='division')

#===BarPlot===
#sns.barplot(data=df,x='level of education',y='salary',estimator=np.mean,ci='sd')


#===Boxplot===
#sns.boxplot(data=df,y='math score',x='parental level of education')


#===Violinplot===
#sns.violinplot(data=df,y='math score',x='parental level of education',inner='quartile')

#===swarmplot===
sns.swarmplot(data=df,y='math score',x='gender',hue='test preparation course',dodge=True,size=2)


plt.show()