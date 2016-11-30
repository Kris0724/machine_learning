import numpy as np
from array import array
import sys
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU

max_features = 40001
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 16

x_train = list()
x_test = list()
y_train = list()
y_test = list()
for line in open('horseColicTraining.txt'):
    line = line.strip('\n').strip('\r')
    arr = line.split('\t')
    len_arr = len(arr)
    if len_arr != 22:
        continue
    l = []
    for i in range(0, len_arr - 1):
        l.append(float(arr[i]))
    x_train.append(l)
    label = int(float(arr[-1]))
    y_train.append(label)

for line in open('horseColicTest.txt'):
    line = line.strip('\n').strip('\r')
    arr = line.split('\t')
    len_arr = len(arr)
    if len_arr != 22:
        continue
    l = []
    for i in range(0, len_arr - 1):
        l.append(float(arr[i]))
    x_test.append(l)
    label = int(float(arr[-1]))
    y_test.append(label)

y_train = np.array(y_train)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
#print x_train.shape

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
#print x_test.shape

#exit(0)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.2))
#model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(LSTM(256, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
#model.add(SimpleRNN(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
#model.add(SimpleRNN(256, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
#model.add(Dense(1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=2)
classes = model.predict_classes(x_test, batch_size=128, verbose=1)
test_accuracy = np.mean(np.equal(y_test,classes))
print("accuarcy:",test_accuracy) 

