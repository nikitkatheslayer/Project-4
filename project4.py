#загрузка библиотек

#поддержка многомерных массивов 
import numpy
#разбивает исходных текст на подстроки, используя регулярное выражение, передаваемое ему в качестве параметра
from nltk.tokenize import RegexpTokenizer
#предустановленный список стоп-слов
from nltk.corpus import stopwords
#последовательное описание модели
from keras.models import Sequential
#
from keras.layers import Dense, Dropout, LSTM
#метод для предварительной обработки
from keras.utils import np_utils
#сохранение модели после прохождения всех эпох
from keras.callbacks import ModelCheckpoint
#пакет библиотек и программ для символьнйо и статической обработка естественного языка
import nltk

#загружается файл
file = open("Tonkosti_dizassemblirovanija.txt", encoding="utf8").read()
#загружается список стоп слов(слова, предлоги, которые не несут смысловой нагрузки)
nltk.download('stopwords')
#список стоп слов для русского языка
stopwords.words("russian")

#процедура токенизации(большое количество текста делится на более мелкие части(токены). К токенам относятся как слова, так и знаки пунктуации)
def tokenize_words(input):
    #сохранение в нижнем регистре, чтобы стандартизировать его
    input = input.lower()

    #создание экземпляра токенизатора
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    #если созданного токена нет в стоп-словах, сделать его частью "отфильтрованного"
    filtered = filter(lambda token: token not in stopwords.words('russian'), tokens)
    return " ".join(filtered)

#вызов функции в нашем файле
processed_inputs = tokenize_words(file)
#сортировка списка набора всех символов
chars = sorted(list(set(processed_inputs)))
#получаем числа, представляющие символы(функция enumerate)
char_to_num = dict((c, i) for i, c in enumerate(chars))
#общая длина наших входных данных
input_len = len(processed_inputs)
#общая длина нашего набора символов
vocab_len = len(chars)
#вывод input_lem
print ("Total number of characters:", input_len)
#вывод vocab_len
print ("Total vocab:", vocab_len)

#длина набор данных
seq_length = 100
#создание списков
x_data = []
y_data = []
#перебираем входные данные, начиная с самого начала до тех пор, пока не дойдем до последнего символа,
#из которого мы можем создать последовательность
for i in range(0, input_len - seq_length, 1):
    #определение входных и выходных последовательностей
    #текущий символ плюс желаемая длина последовательности
    in_seq = processed_inputs[i:i + seq_length]

    #исходная последовательность - это начальный символ плюс общая длина последовательности
    out_seq = processed_inputs[i + seq_length]

    #теперь мы преобразуем список символов в целые числа на основе предыдущего и добавляем значения в наши списки
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

#даем переменной n_patterns список x_data, в котором хранятся общие количество последовательностей
n_patterns = len(x_data)
#вывод n_patterns
print ("Total Patterns:", n_patterns)

#преобразование наших входных последовательностей в исходных массив numpy, для правильной работы нашей сети
X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
#стандартизация массива
X = X/float(vocab_len)
#предварительная обработка y_data(преобразует вектор класса в двоичную матрицу классов)
y = np_utils.to_categorical(y_data)

#указываем тип модели, которую хотим сделать
model = Sequential()
#создание LSTM модели
#отсев, чтобы предотвратить переобучение модели
#первый слой (количество нейронов, количество временных шагов(shape индикаторы), return_sequences true, если мы добавляем еще один слой)
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
#слой отсева, для избежания переобучения
model.add(Dropout(0.2))
#второй слой
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
#третий слой
model.add(LSTM(128))
model.add(Dropout(0.2))
#создание выходного слоя
model.add(Dense(y.shape[1], activation='softmax'))
#компилирование модели
model.compile(loss='categorical_crossentropy', optimizer='adam')

#указываем имя файла
filepath = "model_weights_saved.hdf5"
#сохранение весов и его перезагрузка после того как тренировка закончится
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#установка контрольной точки для сохранения весов
desired_callbacks = [checkpoint]

#обучение модели (epochs - проход)
model.fit(X, y, epochs=4, batch_size=256, callbacks=desired_callbacks)

#указываем имя файла
filename = "model_weights_saved.hdf5"
#указываем нагрузку на весах
model.load_weights(filename)
#перекомпелирование модели с сохраненными весами
model.compile(loss='categorical_crossentropy', optimizer='adam')

#преобразование выходных данных модели в числа
num_to_char = dict((i, c) for i, c in enumerate(chars))
#чтобы генерировать символы, нам нужно представить нашей обученной модели случайный первоначальный символ,
#из которой она может генерировать последовательность символоа
start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
#вывод
print("Random Seed:")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")






