import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

res = ''

#region Постановка задачи

# Компания-дистрибьютор вина недавно столкнулась с подделками.
# В итоге был проведён аудит различных вин с помощью химического анализа.
# Компания занимается экспортом очень качественных и дорогих вин,
# но один из поставщиков попытался передать дешёвое вино под видом более дорогого.
# Компания-дистрибьютор наняла Вас, чтобы Вы создали модель машинного обучения,
# которая предскажет низкое качество вина (то есть, "подделку"). Они хотят узнать,
# возможно ли определить разницу между дешёвыми и дорогими винами.\

#Задача - **ЗАДАНИЕ: Обшая цель - используя данные ниже, разработайте модель машинного обучения,
# которая будет предсказывать на основе некоторых химических тестов, является ли вино настоящим или поддельным.
# Выполните задания ниже.**

#endregion
#region Загрузка и просмотр данных

df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\wine_fraud.csv')

#    fixed acidity  volatile acidity  citric acid  ...  alcohol  quality  type
# 0            7.4              0.70         0.00  ...      9.4    Legit   red
# 1            7.8              0.88         0.00  ...      9.8    Legit   red
# 2            7.8              0.76         0.04  ...      9.8    Legit   red
# 3           11.2              0.28         0.56  ...      9.8    Legit   red
# 4            7.4              0.70         0.00  ...      9.4    Legit   red
res = df.head()

#  #   Column                Non-Null Count  Dtype
# ---  ------                --------------  -----
#  0   fixed acidity         6497 non-null   float64
#  1   volatile acidity      6497 non-null   float64
#  2   citric acid           6497 non-null   float64
#  3   residual sugar        6497 non-null   float64
#  4   chlorides             6497 non-null   float64
#  5   free sulfur dioxide   6497 non-null   float64
#  6   total sulfur dioxide  6497 non-null   float64
#  7   density               6497 non-null   float64
#  8   pH                    6497 non-null   float64
#  9   sulphates             6497 non-null   float64
#  10  alcohol               6497 non-null   float64
#  11  quality               6497 non-null   object
#  12  type                  6497 non-null   object
res = df.info()


# quality
# Legit    6251
# Fraud     246
res = df['quality'].value_counts()

#endregion


#region Исследование данных

#region Соотношение классов
# sns.countplot(data=df,x='quality',palette='Set2')
#endregion
#region Какое вино подделывают чаще
# sns.countplot(data=df,x='type',palette='Set2',hue='quality')
#endregion
#region Какой процент вин (красного и белого) подделан

red_df = df[df['type'] == 'red']
white_df = df[df['type'] == 'white']

# 3.9399624765478425
res = (len(red_df[red_df['quality'] == 'Fraud']) / len(red_df)) * 100
# 3.7362188648427925
res = (len(white_df[white_df['quality'] == 'Fraud']) / len(white_df)) * 100

#endregion
#region Корреляции переменных с классами

qual_map = {'Fraud':1,'Legit':0}
df['quality_num'] = df['quality'].map(qual_map)
corr_qual = df.corr(numeric_only=True)['quality_num'].sort_values(ascending=True).reset_index()
# sns.barplot(data=corr_qual,x='index',y='quality_num',palette='Set2')


#endregion
#region Корреляция переменных

# sns.clustermap(df.corr(numeric_only=True),cmap='viridis')

#endregion

#endregion



print(res)
plt.show()