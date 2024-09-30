import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from svm_margin_plot import plot_svm_boundary

res = ''

df = pd.read_csv(r'D:\Khabarov\Курс ML\DATA\mouse_viral_study.csv')
#      Med_1_mL  Med_2_mL  Virus Present
# 0    6.508231  8.582531              0
# 1    4.126116  3.073459              1
# 2    6.427870  6.369758              0
res = df

#Построение scatterplot
# sns.scatterplot(x='Med_1_mL',y='Med_2_mL',hue='Virus Present',data=df)



#Построение разделяющей линии
x = np.linspace(0,10,100)
m = -1
b = 11
y = m*x + b
# plt.plot(x,y,'black')



#Создаем модель (здесь сделаем это для визуализации)
y = df['Virus Present']
X = df.drop('Virus Present',axis=1)

y = pd.DataFrame(y)
X = pd.DataFrame(X)


model = SVC(kernel='poly',degree=4)
model.fit(X,y)

plot_svm_boundary(model,X,y)


print(res)
plt.show()