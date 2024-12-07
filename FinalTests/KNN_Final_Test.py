import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Pandas.pandasTest import daysDf

res = ''

#region Постановка задачи

#В данной задаче у нас есть данные по 60 сканируемым (сонаром) объектам. Какие то из них камни, а какие то - мины.
#Наша задача - построить модель МО для определения


#endregion
#region Загрузка и просмотр данных
#    Freq_1  Freq_2  Freq_3  Freq_4  ...  Freq_58  Freq_59  Freq_60  Label
# 0  0.0200  0.0371  0.0428  0.0207  ...   0.0084   0.0090   0.0032      R
# 1  0.0453  0.0523  0.0843  0.0689  ...   0.0049   0.0052   0.0044      R
# 2  0.0262  0.0582  0.1099  0.1083  ...   0.0164   0.0095   0.0078      R
# 3  0.0100  0.0171  0.0623  0.0205  ...   0.0044   0.0040   0.0117      R
# 4  0.0762  0.0666  0.0481  0.0394  ...   0.0048   0.0107   0.0094      R
df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\sonar.all-data.csv')
res = df.head()

# #   Column   Non-Null Count  Dtype
# ---  ------   --------------  -----
#  0   Freq_1   208 non-null    float64
#  1   Freq_2   208 non-null    float64
#  2   Freq_3   208 non-null    float64
#  3   Freq_4   208 non-null    float64
#  4   Freq_5   208 non-null    float64
#  5   Freq_6   208 non-null    float64
#  6   Freq_7   208 non-null    float64
#  7   Freq_8   208 non-null    float64
#  8   Freq_9   208 non-null    float64
#  9   Freq_10  208 non-null    float64
#  10  Freq_11  208 non-null    float64
#  11  Freq_12  208 non-null    float64
#  12  Freq_13  208 non-null    float64
#  13  Freq_14  208 non-null    float64
#  14  Freq_15  208 non-null    float64
#  15  Freq_16  208 non-null    float64
#  16  Freq_17  208 non-null    float64
#  17  Freq_18  208 non-null    float64
#  18  Freq_19  208 non-null    float64
#  19  Freq_20  208 non-null    float64
#  20  Freq_21  208 non-null    float64
#  21  Freq_22  208 non-null    float64
#  22  Freq_23  208 non-null    float64
#  23  Freq_24  208 non-null    float64
#  24  Freq_25  208 non-null    float64
#  25  Freq_26  208 non-null    float64
#  26  Freq_27  208 non-null    float64
#  27  Freq_28  208 non-null    float64
#  28  Freq_29  208 non-null    float64
#  29  Freq_30  208 non-null    float64
#  30  Freq_31  208 non-null    float64
#  31  Freq_32  208 non-null    float64
#  32  Freq_33  208 non-null    float64
#  33  Freq_34  208 non-null    float64
#  34  Freq_35  208 non-null    float64
#  35  Freq_36  208 non-null    float64
#  36  Freq_37  208 non-null    float64
#  37  Freq_38  208 non-null    float64
#  38  Freq_39  208 non-null    float64
#  39  Freq_40  208 non-null    float64
#  40  Freq_41  208 non-null    float64
#  41  Freq_42  208 non-null    float64
#  42  Freq_43  208 non-null    float64
#  43  Freq_44  208 non-null    float64
#  44  Freq_45  208 non-null    float64
#  45  Freq_46  208 non-null    float64
#  46  Freq_47  208 non-null    float64
#  47  Freq_48  208 non-null    float64
#  48  Freq_49  208 non-null    float64
#  49  Freq_50  208 non-null    float64
#  50  Freq_51  208 non-null    float64
#  51  Freq_52  208 non-null    float64
#  52  Freq_53  208 non-null    float64
#  53  Freq_54  208 non-null    float64
#  54  Freq_55  208 non-null    float64
#  55  Freq_56  208 non-null    float64
#  56  Freq_57  208 non-null    float64
#  57  Freq_58  208 non-null    float64
#  58  Freq_59  208 non-null    float64
#  59  Freq_60  208 non-null    float64
#  60  Label    208 non-null    object
res = df.info()

# Label
# M    111 - Mine
# R     97 - Rock
res = df['Label'].value_counts()

#endregion

#region Исследование данных

#region Построение heatmap
# sns.heatmap(data = df.corr(numeric_only=True),cmap="viridis")
#endregion
#region Корреляция с целевой переменной

df['Label_num'] = np.where(df['Label'] == 'R',0,1)
corr_label = np.abs(df.corr(numeric_only=True)['Label_num'].sort_values(ascending=True)).tail(6)
res = corr_label

#endregion

#endregion

#region Деление данных

y = df['Label']
X = df.drop(columns=['Label','Label_num'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 101)

#endregion
#region Создание звеньев для Pipeline

scaler = StandardScaler()
knn_est = KNeighborsClassifier()

#endregion
#region Создание Pipeline

operations = [('scaler',scaler)
            ,('knn_est',knn_est)]
pipeline = Pipeline(operations)

#endregion
#region Создание GridSearchCV

param_grid = {'knn_est__n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                              12, 13, 14, 15, 16, 17, 18, 19,
                                              20, 21, 22, 23, 24, 25, 26, 27,
                                              28, 29]}
grid_model = GridSearchCV(estimator=pipeline
                          ,param_grid=param_grid
                          ,return_train_score=True
                          ,scoring='f1_weighted'
                          ,verbose=2
                          ,cv=10)

#endregion
#region Обучение модели и предсказание

grid_model.fit(X_train,y_train)
y_pred = grid_model.predict(X_test)

#Лучшие параметры - k = 3
# {'memory': None, 'steps': [('scaler', StandardScaler()),
#                            ('knn_est', KNeighborsClassifier(n_neighbors=3))],
#  'verbose': False, 'scaler': StandardScaler(), 'knn_est': KNeighborsClassifier(n_neighbors=3),
#  'scaler__copy': True, 'scaler__with_mean': True, 'scaler__with_std': True, 'knn_est__algorithm': 'auto',
#  'knn_est__leaf_size': 30, 'knn_est__metric': 'minkowski', 'knn_est__metric_params': None, 'knn_est__n_jobs': None,
#  'knn_est__n_neighbors': 3, 'knn_est__p': 2, 'knn_est__weights': 'uniform'}
res = grid_model.best_estimator_.get_params()

#endregion

#region Поиск лучшего n - количества точек

# test_error_rates = []
#
# for k in range (1,25):
#     k_search_scaler = StandardScaler()
#     k_search_scaler.fit(X_train,y_train)
#     X_train_scaled = k_search_scaler.transform(X_train)
#     X_test_scaled = k_search_scaler.transform(X_test)
#
#     test_model = KNeighborsClassifier(n_neighbors=k)
#     test_model.fit(X_train_scaled,y_train)
#     y_k_search_pred = test_model.predict(X_test_scaled)
#
#     test_error = 1 - accuracy_score(y_test, y_k_search_pred)
#     test_error_rates.append(test_error)
#
# #Построение графика по методу локтя
# plt.plot(range(1,25),test_error_rates)
# plt.ylabel('Error rate')
# plt.xlabel('К ближайших соседей')
# # plt.ylim(0,0.11)

#region Получаем оценки для каждой модели
scores = grid_model.cv_results_
#    param_knn_est__n_neighbors  mean_test_score
# 0                           1         0.853640
# 1                           3         0.863742
# 2                           5         0.809737
# 3                           8         0.750840
# 4                          13         0.706927
# 5                          21         0.683129
# 6                          34         0.722459
means_test_score_df = pd.DataFrame(data=scores)[['param_knn_est__n_neighbors','mean_test_score']]
means_test_score_arr = means_test_score_df['mean_test_score']
#Построение графика по методу локтя
plt.plot(range(1,30),means_test_score_arr)
plt.ylabel('Error rate')
plt.xlabel('К ближайших соседей')
# plt.ylim(0,0.11)



#endregion


#endregion
#region Оценка
#Матрица ошибок
# [[9 2]
#  [1 9]]
confM = confusion_matrix(y_test,y_pred)
res = confM

#Визуализация матрицы ошибок
ConfusionMatrixDisplay(confusion_matrix(y_pred, y_test)).plot()

#Отчет по метрикам
#               precision    recall  f1-score   support
#
#            M       0.90      0.82      0.86        11
#            R       0.82      0.90      0.86        10
#
#     accuracy                           0.86        21
#    macro avg       0.86      0.86      0.86        21
# weighted avg       0.86      0.86      0.86        21
metricsRep = classification_report(y_test,y_pred)
# res = metricsRep

#endregion



print(res)
plt.show()