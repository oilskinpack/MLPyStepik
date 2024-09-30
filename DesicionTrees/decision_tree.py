import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


res = ''

df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\penguins_size.csv')

#   species     island  culmen_length_mm  ...  flipper_length_mm  body_mass_g     sex
# 0  Adelie  Torgersen              39.1  ...              181.0       3750.0    MALE
res = df.head()

#Породы пингвинов - ['Adelie' 'Chinstrap' 'Gentoo']
res = df['species'].unique()

#Проверка null значений
# species               0
# island                0
# culmen_length_mm      2
# culmen_depth_mm       2
# flipper_length_mm     2
# body_mass_g           2
# sex                  10
res = df.isnull().sum()

res = df['island'].unique()

res =  df['sex'].unique()

#Определяем единственную строку, где пол не определен
res = df[df['sex'] == '.']
res = df[df['species'] == 'Gentoo'].groupby('sex').describe().transpose()
df.at[336,'sex'] = 'Female'
res = df.loc[336]

#Создаем парные графики, чтобы увидеть можно ли разделить классы
sns.pairplot(data=df,hue='species')
plt.clf()

#Создаем признаки, создавая так же dummy переменные
X = pd.get_dummies(df.drop('species',axis=1),drop_first=True)
#Создаем целевую переменную
y = df['species']

#МОЖНО НЕ МАСШТАБИРОВАТЬ, ТАК КАК В КАЖДОМ УЗЛЕ БУДЕТ ТОЛЬКО 1 ПРИЗНАК

#Деление данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Создание алгоритма
model = DecisionTreeClassifier()

#Обучение
model.fit(X_train,y_train)

#Предсказания
base_pred = model.predict(X_test)

#Метрики
print(classification_report(y_test,base_pred))

#Интерпретация модели
coefs = model.feature_importances_
coef_columns = X.columns
features_importance_df = (pd.DataFrame(data=coefs,index=coef_columns,columns=['Важность признаков'])
                          .sort_values('Важность признаков'))

#                    Важность признаков
# island_Dream                 0.000000
# island_Torgersen             0.000000
# sex_Female                   0.000000
# sex_MALE                     0.009785
# body_mass_g                  0.040623
# culmen_depth_mm              0.070005
# culmen_length_mm             0.368144
# flipper_length_mm            0.511443
res = features_importance_df

#Визуализация дерева
plot_tree(model,feature_names=X.columns,filled=True)
plt.clf()

#Гиперпараметр - Макс количество уровней дерева
pruned_tree = DecisionTreeClassifier(max_depth=2)
pruned_tree.fit(X_train,y_train)
plot_tree(pruned_tree,feature_names=X.columns,filled=True)
plt.clf()


#Гиперпараметр - Максимальное количество листовых узлов
max_leaf_tree = DecisionTreeClassifier(max_leaf_nodes=3)
max_leaf_tree.fit(X_train,y_train)
plot_tree(max_leaf_tree,feature_names=X.columns,filled=True)


print(res)
plt.show()