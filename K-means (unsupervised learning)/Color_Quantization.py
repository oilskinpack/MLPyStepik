import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

res = ''

#region Загрузка данных
image_arr =  mpimg.imread(r'D:\Khabarov\Репозиторий\MachineLearningCourse\DATA\palm_trees.jpg')

# (1401, 934, 3) - габариты изображения и 3 цвета
res = image_arr.shape

#Просмотр изображения
# plt.imshow(image_arr)

#endregion
#region Обработка

#Распаковка кортежа в 3 переменных (Высота, Ширина, Кол-во цветов)
(h,w,c) = image_arr.shape

#Изменяем размерность, чтобы она была 2хмерной
#Строки - пиксели, колонки(3шт) - цвета
image_arr_2d = image_arr.reshape(h*w,c)

# (1308534, 3)
res = image_arr_2d.shape

#endregion
#region Кластеризация цветов

#Создание модели
model = KMeans(n_clusters=6)

#Кластеризуем
labels = model.fit_predict(image_arr_2d)

#Центры кластеров - в нашем случае это новые цвета (+ округляем и превращаем в int)
# [[  3   3  4]
#  [138 144 143]
#  [219 135  46]
#  [ 72 110 138]
#  [193 155 108]
#  [ 67  62  62]]
rgb_codes = model.cluster_centers_.round(0).astype(int)
res = rgb_codes

#Меняем значения rgb на наши полученные центры кластеров
new_colors = rgb_codes[labels]

#Возвращаем размерность обратно - получаем такой же массив, как в начале
#Но уже с другими цветами
# [[[ 71 109 138]
#   [ 71 109 138]
#   [ 71 109 138]
#   ...
#   [ 68  62  62]
#   [ 71 109 138]
#   [ 71 109 138]]
quantized_image = np.reshape(new_colors,(h,w,c))
res = quantized_image

#endregion
#region Просмотр результата

plt.imshow(quantized_image)

#endregion




print(res)
plt.show()