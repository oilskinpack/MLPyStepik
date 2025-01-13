import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, mean_absolute_error, \
    mean_squared_error
from sklearn.metrics._classification import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from CrossValidation.Grid_Search import param_grid
from KNN.knn import k_values

res = ''

def run_model(model,X_train,y_train,X_test,y_test):
    #Обучение модели
    model.fit(X_train,y_train)
    #Вычисление метрики
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test,preds))
    mae = mean_absolute_error(y_test,preds)
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    #Построить график с результатами
    signal_range = np.arange(0, 100)
    signal_pred = model.predict(signal_range.reshape(-1, 1))
    plt.plot(signal_range, signal_pred)
    sns.scatterplot(x='Signal', y='Density', data=df,color='black')

#region Подготовка данных
#Данные о плотности породы при бурении при разных частотах
df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\rock_density_xray.csv')

#Переименуем колонки
df.columns = ['Signal','Density']

#Просмотр данных
# sns.scatterplot(x='Signal',y='Density',data=df)

#Вывод признака и целевой переменной
X = df['Signal'].values.reshape(-1,1)
res = X
y = df['Density']

#Деление
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
#endregion
#region Линейная регрессия
lr_model = LinearRegression()
lr_model.fit(X_train,y_train)
lr_preds = lr_model.predict(X_test)

#Метрика
mae =  mean_absolute_error(y_test,lr_preds)

#Проверка на графике
# run_model(lr_model,X_train,y_train,X_test,y_test)

#endregion
#region Полиномиальная регрессия
pipe = make_pipeline(PolynomialFeatures(degree=6),LinearRegression())
# run_model(pipe,X_train,y_train,X_test,y_test)

#endregion
#region К ближайших соседей
k_values = 10
k_model = KNeighborsRegressor(n_neighbors=10)
# run_model(k_model, X_train, y_train, X_test, y_test)



#endregion
#region Дерево решений
tree_model = DecisionTreeRegressor()
# run_model(tree_model, X_train, y_train, X_test, y_test)

#endregion
#region Метод опорных векторов - здесь придется перебрать параметры (сильно зависят от входных данных)
svr = SVR()
param_grid = {'C':[0.01,0.1,1,5,10,15,100]
              ,'gamma':['auto','scale']}
grid = GridSearchCV(param_grid=param_grid,estimator=svr)
# run_model(grid, X_train, y_train, X_test, y_test)

#endregion
#region Случайные леса
rfr = RandomForestRegressor(n_estimators=10)
# run_model(rfr, X_train, y_train, X_test, y_test)
#endregion
#region Расширяемые деревья
boost_model = GradientBoostingRegressor()
# run_model(boost_model, X_train, y_train, X_test, y_test)

ada_boost_model = AdaBoostRegressor()
run_model(ada_boost_model, X_train, y_train, X_test, y_test)
#endregion


plt.show()
print(res)