import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

res = ''
#Данные по моделям автомобилей
df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\cluster_mpg.csv')

#region Просмотр и подготовка данных

#Делаем дамми переменные
df_w_dummies = pd.get_dummies(df.drop(['name'],axis=1))
res = df_w_dummies

#Масштабируем
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_w_dummies)
scaled_df = pd.DataFrame(scaled_data,columns = df_w_dummies.columns)

#Тепловая карта
# sns.heatmap(scaled_df,cmap='viridis')

#Кластер карта
sns.clustermap(scaled_df,cmap='viridis',col_cluster=False)


#endregion


print(res)
plt.show()