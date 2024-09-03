import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#df = pd.read_csv(r'D:\Khabarov\Курс ML\05-Seaborn\dm_office_sales.csv')
df = pd.read_csv(r'D:\Khabarov\Курс ML\05-Seaborn\country_table.csv')


#Подготовка данных
df = df.set_index('Countries')
heatData = df.drop('Life expectancy',axis=1)

#===heatplot===
#linewidth - толщина разграничителей
#annot - показывать ли значения
sns.heatmap(data=heatData,linewidths=0.5,annot=True)


plt.show()