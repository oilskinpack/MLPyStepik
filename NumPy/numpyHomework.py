import numpy as np
res = ''

#Создайте массив из 10 нулей - array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
res = np.zeros(10)

#Создайте массив из 10 единиц - [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
res = np.ones(10)

#Создайте массив из 10 пятерок - [5., 5., 5., 5., 5., 5., 5., 5., 5., 5.]
res = np.array([5,5,5,5,5,5,5,5,5,5])

#Создайте массив целых чисел от 10 до 50 вкл
res = np.arange(10,51,1)

#Создайте массив всех чётных чисел от 10 до 50 включительно
res = np.arange(10,52,2)

#Создайте матрицу 3x3 со значениями от 0 до 8 включительно
res = np.arange(0,9,1).reshape(3,3)

#Создайте единичную матрицу размером 3x3
res = np.eye(3)

#С помощью NumPy сгенерируйте случайное число в диапазоне от 0 до 1
res = np.random.random(1)

#С помощью NumPy сгенерируйте массив из 25 случайных чисел, выбранных из стандартного нормального распределения
res = np.random.randn(25)

#Создайте указанную ниже матрицу
res = np.linspace(0.01,1,100).reshape(10,10)

#Создайте массив из 20 равноудалённых друг от друга точек между 0 и 1
res = np.linspace(0,1,20)

#Создаем матрицу
mat = np.arange(1,26).reshape(5,5)

#Напишите код, который на основе матрицы mat выведет указанную ниже матрицу (подсказка - используйте индексацию)
res = mat[2:5,1:5]

#Напишите код, который на основе матрицы mat выведет указанную ниже матрицу.
res = mat[0:3,1:2]

#Напишите код, который на основе матрицы mat выведет указанную ниже матрицу.
res = mat[4:5,0:5]

#Напишите код, который на основе матрицы mat выведет указанную ниже матрицу.
res = mat[3:5,0:5]

#Найдите сумму всех чисел
res = mat.sum()

#Найдите среднеквадратичное отклонение значений в матрице mat
res = mat.std()

#Найдите сумму каждой из колонок в матрице mat
res = mat.sum(axis=0)

#Генерация случайных неслучайных чисел
np.random.seed(42)
res = np.random.rand(4)

print(res)