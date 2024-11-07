from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

text = ['This is a line',
        'This is another line',
        'Completely different line']

#region Подсчет TF - по сути мешка слов
#Векторайзер слов в документах
#stop_words - стоп слова. Можно передать язык или написать их своим списком
# cv = CountVectorizer(stop_words='english')
cv = CountVectorizer()
#Создание разреженной матрицы
sparse_matrix = cv.fit_transform(text)

#Перевод её в обычную матрицу - 3 строки документа, 6 слов уникальных
# [[0 0 0 1 1 1]
#  [1 0 0 1 1 1]
#  [0 1 1 0 1 0]]
normal_matrix = sparse_matrix.todense()
res = normal_matrix

#Просмотр словаря
# {'line': 2, 'completely': 0, 'different': 1}
indexes = cv.vocabulary_
res = indexes
#endregion
#region Подсчет IDF - количество с поправкой на встречаемость во всех документах

#Просмотр в разреженном виде
# Coords Values
# (0, 3) 0.6198053799406072
# (0, 4) 0.48133416873660545
# ...
# (2, 4)	0.3853716274664007
tfidf = TfidfTransformer()
tfidf_sparce_res = tfidf.fit_transform(sparse_matrix)


#Просмотр в явном виде
# [[0.         0.         0.         0.61980538 0.48133417 0.61980538]
#  [0.63174505 0.         0.         0.4804584  0.37311881 0.4804584 ]
#  [0.         0.65249088 0.65249088 0.         0.38537163 0.        ]]
res = tfidf_sparce_res.todense()

#Или вообще можем сделать вот так - получаем то же самое
# [[0.         0.         0.         0.61980538 0.48133417 0.61980538]
#  [0.63174505 0.         0.         0.4804584  0.37311881 0.4804584 ]
#  [0.         0.65249088 0.65249088 0.         0.38537163 0.        ]]
tv = TfidfVectorizer()
tv_sparce_res = tv.fit_transform(text)

#endregion

print(res)