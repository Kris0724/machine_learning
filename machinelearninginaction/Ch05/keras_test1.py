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
        #print "i:", i
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
        #print "i:", i
        l.append(float(arr[i]))
    x_test.append(l)
    label = int(float(arr[-1]))
    y_test.append(label)

y_train = np.array(y_train)

#print(len(x_train), 'train sequences')
#print(len(x_test), 'test sequences')
#print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
#print (type(x_train))
#print('x_train shape:', x_train.shape)
#print('x_test shape:', x_test.shape)
#print (x_train)
#print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
#print(x_train.shape)
#print(y_train.shape)
#model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15)
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1)

#print (model.predict_classes(x_test, batch_size=128, verbose=1))
classes = model.predict_classes(x_test, batch_size=128, verbose=1)
test_accuracy = np.mean(np.equal(y_test,classes))
print("accuarcy:",test_accuracy) 

#print("final result:")
#for item in classes:
#    print(item)



#y_train = sequence.pad_sequences(y_train, maxlen=100)
#print y_train.shape

#from keras.models import Sequential
#model = Sequential()
#from keras.layers import Dense, Activation
#model.add(Dense(output_dim=64, input_dim=100))
#model.add(Activation("relu"))
#model.add(Dense(output_dim=10))
#model.add(Activation("softmax"))

#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#model.fit(x_train, y_train, nb_epoch=5, batch_size=32)


