import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#df = pd.read_csv(r'D:\Khabarov\Курс ML\05-Seaborn\dm_office_sales.csv')
df = pd.read_csv(r'D:\Khabarov\Курс ML\05-Seaborn\StudentsPerformance.csv')

#===catplot===
#sns.catplot(data=df,x='gender',y='math score',kind = 'box',row='lunch')

#===pairgrid===
g = sns.PairGrid(data=df)
g = g.map_upper(sns.scatterplot)
g = g.map_lower(sns.kdeplot)
g = g.map_diag(sns.histplot)

plt.show()