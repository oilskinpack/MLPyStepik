import numpy as np

#Создание одномерного списка из обычного списка
myList = [1,2,3]
#Тип - <class 'list'>
print(type(myList))
#Тип - <class 'numpy.ndarray'>
print(type(np.array(myList)))

#Создание двухмерного списка
my_matrix = [[1,2,3],[4,5,6],[7,8,9]]