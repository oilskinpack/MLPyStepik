import matplotlib.pyplot as plt
import numpy as np

#Создание данных
a = np.linspace(0,10,11)
b = a**4

x = np.arange(0,10)
y = 2*x


#Создание графиков - распаковка кортежа
fig,axes= plt.subplots(nrows=2,ncols=2)

#Размещение данных - указываем n-строки, n-колонки
axes[0,0].plot(x,y)
axes[0,1].plot(a,b)

#Расстояние между графиками - кастомное
#fig.subplots_adjust(wspace=1,hspace=0.5)
#Расстояние между графиками - автоматом
plt.tight_layout()

plt.show()