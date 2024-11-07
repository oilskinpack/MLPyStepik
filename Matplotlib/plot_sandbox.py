from random import random
import matplotlib.pyplot as plt
import numpy as np

#region Данные для графика
x_data = [0,2,4,6,10]
y_data = [0.5,2,6,8,10]

x_new_data = [10,6,4,2,0]
y_new_data = [10,8,6,2,0.5]
#endregion
#region Создание плота
figsize = (10,6)
#Для одиночного графика
# plot = plt.figure(figsize=figsize)

#Для двух линий на одном графике или двух графиков
fig,axes = plt.subplots(nrows=1,ncols=1,figsize=figsize)

#endregion
#region Создание axe
# axe_one = fig.add_axes((0.1,0.1,0.8,0.8))
axe_one = axes
axe_one.plot(x_data,y_data)

#endregion
#region Установка лимита для осей
axe_one.set_xlim(0,10)
axe_one.set_ylim(0,10)
#endregion
#region Настройка сетки
#visible - true/false
#axis - {'both', 'x', 'y'} К каким осям применяем
#linestyle - {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
#linewidth - 5
#color - (0.1, 0.2, 0.5, 0.3) для rgb или 'colorname'
axe_one.grid(visible=True
             ,axis='y'
             ,linewidth = 1
             ,markersize = 5
             ,color='#751501'
             ,linestyle='--')

#endregion
#region Названия осей и графика
#Словарь который передается для настройки текста
#fontsize(int) - размер
#fontweight - [ 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight']
#color - (0.1, 0.2, 0.5, 0.3) для rgb или 'colorname'
#verticalalignment - [ 'center' | 'top' | 'bottom' | 'baseline' ]
#horizontalalignment - [ 'center' | 'right' | 'left' ]
fontdict = {'fontsize': 14
    ,'fontweight': 'bold'
    , 'color': (0.1, 0.2, 0.5, 0.3)
    ,'verticalalignment': 'center'
    ,'horizontalalignment':'center'}
loc = 'center' #loc -  {'center', 'left', 'right'} Положение текста
fontname = 'Grtsk Exa' #Название шрифта
pad = 30 #Смещение z


labelpad = 16  #Смещение выше/ниже
axe_one.set_xlabel(xlabel = 'Название для оси X'
                   , fontdict=fontdict
                   , fontname=fontname
                   ,loc=loc
                   ,labelpad=labelpad)
axe_one.set_ylabel(ylabel = 'Название для оси Y'
                   , fontdict=fontdict
                   , fontname=fontname
                   ,loc=loc
                   ,labelpad=labelpad)

#Заголовок
axe_one.set_title(label='Название графика'
                  ,loc = loc
                  ,pad = pad
                  ,fontdict=fontdict
                  ,fontname=fontname)
#endregion
#region Шкала (риски) на осях

ticks = [1.5,3.33,6]  #Какие точки отображать на шкале
labels = ['Раз','Два','Три']  #Что подписать
minor = False     #True - показывает и дефолтные и твои, False - только твои
axe_one.set_xticks(ticks=ticks
                   ,labels=labels
                   ,minor=minor)

#endregion
#region Легенда

legend_loc = (0.5,0.5)  #Расположение легенды
axe_one.legend(loc=legend_loc
               ,labels=['Легенда1'])

#endregion
#region Цвет графика



#endregion
#region Создание еще одной линии на том же графике
axe_two = axe_one.twinx()
# axes[1] = axe_two
axe_two.plot(x_data,y_new_data)
axe_two.set_ylabel(ylabel = 'Другое название для Y')
#endregion

plt.show()