import matplotlib.pyplot as plt
import numpy as np

#Создаем данные
a = np.linspace(0,10,11)
b = a**4
x = np.arange(0,10)
y=2*x

#1.Создаем объект figure - пустое пространство для осей
fig = plt.figure()

#2.1.Создание осей
axesOne = fig.add_axes([0,0,1,1])
#2.2.Перенос данных на график
axesOne.plot(a,b)
#2.3.Лимиты
axesOne.set_xlim(0,8)
axesOne.set_ylim(0,8000)
axesOne.set_xlabel('A')
axesOne.set_ylabel('B')
axesOne.set_title('Возведение в степень 4')

#3.2.Создание еще маленьких осей
axesSmall = fig.add_axes([0.2,0.5,0.25,0.25])
#3.2.Перенос данных на график
axesSmall.plot(a,b)
#3.3.Лимиты
axesSmall.set_xlim(1,2)
axesSmall.set_ylim(0,50)
axesSmall.set_xlabel('A')
axesSmall.set_ylabel('B')
axesSmall.set_title('Увеличенный фрагмент')


#4.1 Настройка качества и размера figure
#figsize - размер в дюймах
#dpi - пикселей на дюйм (качество)
fig = plt.figure(figsize=(2,2),dpi=200)
axesOne = fig.add_axes([0,0,1,1])
axesOne.plot(a,b)
#4.2 Сохрание
fig.savefig(r'D:\Khabarov\Скрипты\pythonScripts\Matplotlib\newFig.png',bbox_inches='tight')

plt.show()