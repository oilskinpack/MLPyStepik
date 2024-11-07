import pandas as pd

#region Первый документ
with open(r'D:\Khabarov\Курс ML\18-Naive-Bayes-and-NLP\One.txt') as mytext:
    # Разбиваем текст из файла на отдельные слова
    words_one = mytext.read().lower().split()
    unique_words_one = set(words_one)
#endregion
#region Второй документ
with open(r'D:\Khabarov\Курс ML\18-Naive-Bayes-and-NLP\Two.txt') as mytext:
    # Разбиваем текст из файла на отдельные слова
    words_two = mytext.read().lower().split()
    unique_words_two = set(words_two)

#endregion
#region Получаем множество всех уникальных слов в документе
all_unique_words = set()
all_unique_words.update(unique_words_one)
all_unique_words.update(unique_words_two)

res = all_unique_words
#endregion
#region Создание словаря с порядковым индексом
full_vocab = dict()
i = 0

for word in all_unique_words:
    full_vocab[word] = i
    i = i + 1

# {'this': 0, 'dogs': 1, 'about': 2, 'furry': 3, 'popular': 4, 'are': 5,
#  'animals': 6, 'fun': 7, 'sport': 8, 'pets': 9, 'story': 10, 'is': 11,
#  'a': 12, 'surfing': 13, 'waves': 14, 'our': 15, 'canine': 16,
#  'water': 17, 'catching': 18}
res = full_vocab
#endregion
#region Частота вхождения в документ
#Заготовка под будущий счетчик - bag of words [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
one_freq = [0]*len(full_vocab)
two_freq = [0]*len(full_vocab)
all_words = ['']*len(full_vocab)

for word in words_one:
    word_index = full_vocab[word]
    one_freq[word_index] += 1

#Подсчет какие слова сколько раз встречаются в конкретном документе
#Смотрим по full_vocab, а слова считаем из words_one
#Например,dog встречается 2 раза
# [1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 2, 0, 0, 1, 0, 1]
res = one_freq

for word in words_two:
    word_index = full_vocab[word]
    two_freq[word_index] += 1
# [1, 1, 1, 0, 0, 1, 0, 0, 2, 1, 0, 0, 3, 1, 1, 0, 1, 1, 1]
res = two_freq


#endregion
#region Подготовка колонок

#Мы просто наполняем множество слов из словаря в список
for word in full_vocab:
    word_index = full_vocab[word]
    all_words[word_index] = word
res = all_words

#endregion
#region Финальный датафрейм Bag of words

#Строка - документ
#Колонка - слово
#Значения - количество повторений в тексте
bag_of_words = pd.DataFrame(data=[one_freq,two_freq],columns=all_words)

#endregion

print(res)