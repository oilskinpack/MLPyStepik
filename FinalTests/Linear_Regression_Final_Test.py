import FeatureImportance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


res = ''
usePipeline = False
usePolynomial = True
showResidualsPlot = False

#region Загрузка данных

df = pd.read_csv(r'D:\Khabarov\Курс ML\Data\AMES_Final_DF.csv')
res = df.head()

#endregion
#region Выделение признаков и целевой переменной

X = df.drop('SalePrice',axis=1)
y = df['SalePrice']

#endregion
#region Делим данные Train-Test

#Делим на Train-Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


#endregion

#region Создание через GridSearch
if((usePipeline == False) and (usePolynomial == False)):
    #region Стандартизуем признаки

    #Создаем скейлер
    scaler = StandardScaler()

    #Обучаем скейлер стандартизовывать числовые признаки
    scaler.fit(X_train)

    #Стандартизируем признаки
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    #Получаем


    #endregion
    #region Создаем эстимейтор

    est = ElasticNet()

    #endregion
    #region Создание GridSearch

    param_grid = {'alpha':[150]
                  ,'l1_ratio':[1]}
    grid_model = GridSearchCV(param_grid=param_grid
                              ,estimator=est
                              ,scoring='neg_mean_squared_error'
                              ,verbose=2
                              ,cv=10)

    #endregion
    #region Обучение

    grid_model.fit(X_train,y_train)
    res = grid_model.best_params_

    #endregion
    #region Метрики

    preds = grid_model.predict(X_test)
    MAE = mean_absolute_error(y_test,preds)
    RMSE = np.sqrt(mean_squared_error(y_test,preds))
    res = f'MAE:{MAE} . RMSE:{RMSE}'

    #endregion
#endregion
#region Создание через Pipeline и GridSearch
if((usePipeline == True) and (usePolynomial == False)):
    #region Создаем стоковые элементы для Pipeline

    p_scaler = StandardScaler()
    p_est = ElasticNet()

    #endregion
    #region Создаем пайплайн

    operations = [('scaler',p_scaler)
                  ,('elastic_est',p_est)]

    #endregion
    #region Создание пайплайн

    pipe = Pipeline(operations)


    #endregion
    #region Создание GridSearch

    param_grid = {'elastic_est__alpha':[100,150,200]
                  ,'elastic_est__l1_ratio':[1]}
    p_grid_model = GridSearchCV(estimator=pipe
                                ,param_grid=param_grid
                                , scoring='neg_mean_squared_error'
                                , verbose=2
                                , cv=10)

    #endregion
    #region Обучение
    p_grid_model.fit(X_train,y_train)
    res = p_grid_model.best_estimator_.get_params()

    #endregion
    #region Метрики - # MAE: 14195.35490056217.RMSE:20558.50856689317
    preds = p_grid_model.predict(X_test)
    MAE = mean_absolute_error(y_test,preds)
    RMSE = np.sqrt(mean_squared_error(y_test,preds))
    res = f'MAE:{MAE} . RMSE:{RMSE}'

    #endregion
#endregion
#region График остатков
if(showResidualsPlot):
    #График остатков
    residuals = y_test - preds
    sns.scatterplot(x=y_test,y=residuals)
    plt.axhline(y=0,color='r',ls='--')
    plt.cla()

    #График распределения
    sns.displot(residuals,bins=25,kde=True)

#endregion

#region Создание полиномиальной модели без Pipeline
if((usePipeline == False) and (usePolynomial == True)):
    # region Проверка важности признаков
    est = ElasticNet()
    param_grid = {'alpha': [150]
        , 'l1_ratio': [1]}
    grid_model = GridSearchCV(param_grid=param_grid
                              , estimator=est
                              , scoring='neg_mean_squared_error'
                              , verbose=2
                              , cv=10)
    grid_model.fit(X_train,y_train)
    feature_importance = (pd.DataFrame(data=np.abs(grid_model.best_estimator_.coef_)
                                       ,index=X.columns
                                       ,columns=['Важность признаков'])
                          .sort_values('Важность признаков',ascending=False)).reset_index()
    # sns.barplot(data=feature_importance
    #              ,y='Важность признаков'
    #              ,x='index'
    #              ,palette='Set2')
    # res = feature_importance

    # endregion
    #region Создание полиномиальных признаков - берем 15 самых важных признаков (для примера)

    feature_importance = feature_importance.nlargest(n=15, columns='Важность признаков')
    polynomial_converter = PolynomialFeatures(degree=2, include_bias=False)
    polynomial_converter.fit(X)
    poly_features = polynomial_converter.transform(X)

    #endregion
    #region Делим данные

    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.1, random_state=101)

    #endregion
    #region Создание и обучение модели через GridSearchCV

    est = ElasticNet()
    param_grid = {'alpha':[150]
                  ,'l1_ratio':[1]}
    poly_grid_model = GridSearchCV(param_grid=param_grid
                              ,estimator=est
                              ,scoring='neg_mean_squared_error'
                              ,verbose=2
                              ,cv=10)

    #endregion
    #region Обучение

    poly_grid_model.fit(X_train,y_train)
    res = poly_grid_model.best_params_

    #endregion
    #region Метрики - MAE:24783.769052338554 . RMSE:43228.938056004015

    preds = poly_grid_model.predict(X_test)
    MAE = mean_absolute_error(y_test,preds)
    RMSE = np.sqrt(mean_squared_error(y_test,preds))
    res = f'MAE:{MAE} . RMSE:{RMSE}'

    #endregion

#endregion


print(res)
plt.show()