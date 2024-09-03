import matplotlib.pyplot as plt
import numpy as np

#Создание данных
a = np.linspace(0,10,11)
b = a**4
x = np.arange(0,10)
y = 2*x

#Создание графика
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

#Первый график на оси
#ax.plot(a,b,label='a & b')
#Второй график на той же оси
#ax.plot(x,y,label='x & y')

#1.1 Вывод легенды - верхний левый угол
ax.legend(loc=0)
#1.2 Вывод легенды - кастомный за графиком
#ax.legend(loc=(1.1,0.5))


#2.1 Кастомизация линии
#label - легенда
#color - цвет hex
#lw - толщина линии где 1 по умолчанию [int]
#ls - стиль линии [str]
#ms -  толщина маркера [int]
#marker - тип маркера [str]
#markerfacecolor - цвет маркера внутри [str]
#markeredgecolor - цвет маркера снаружи [str]
#markeredgewidth - толщина маркера снаружи [int]

ax.plot(x,y,label='a & b',color='#b4bd3c',ls='--',ms=30,marker='o',markerfacecolor='red',markeredgewidth=4,markeredgecolor='orange')




plt.show()