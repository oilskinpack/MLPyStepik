import matplotlib.pyplot as plt
import numpy as np

x  = np.arange(0,10)
y = 2 * x

#Создание графика
plt.plot(x,y)
#Название
plt.title('Название графика')
#Ось x
plt.xlabel('Ось Х')
#Ось y
plt.ylabel('Ось Y')

#Мин значение на графике по X
plt.xlim(0,6)
#Мин значение на графике по Y
plt.ylim(0,15)

#Сохранение графика в формате jpg (или png)
#plt.savefig(r'D:\Khabarov\Скрипты\pythonScripts\Matplotlib\MyFirstPlot.jpg')


plt.show()