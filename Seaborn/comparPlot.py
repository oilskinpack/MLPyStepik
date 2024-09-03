import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#df = pd.read_csv(r'D:\Khabarov\Курс ML\05-Seaborn\dm_office_sales.csv')
df = pd.read_csv(r'D:\Khabarov\Курс ML\05-Seaborn\StudentsPerformance.csv')

#===jointPlot===
#kind - вид скаттер плота (hex,hist,kde)
#shade - тени для пересечения kde (true)
#hue - разбивка по категориям
#sns.jointplot(data=df,x='math score',y='reading score',hue='gender')


#===pairPlot===
#hue - разбивка по категориям
#corner - удаление дубликатов
sns.pairplot(data=df,corner=True,hue='gender')


plt.show()

