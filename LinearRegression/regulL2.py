import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv(r'D:\Khabarov\Курс ML\08-Linear-Regression-Models\Advertising.csv')
res = df.head()

#Выделяем признаки и целевую функцию
X = df.drop('sales',axis=1)
y = df['sales']

