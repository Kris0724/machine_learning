from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from array import array
import sys
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
import pickle
max_features = 40001
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 16

print('Loading data...')
f_train=open('train_vector')
datas_train=f_train.readlines()
X_train=list()
X_test=list()
y_train=list()
#i = 0
for line in datas_train:
    #print(line)
    try:
        #print line
        id_list=list()
        line=line.strip().split('\t')
        #print(len(line))
        class_tag=int(line[0])
        ids=line[2]
        ids=ids.split(' ')
        for item in ids:
            id_list.append(int(item))
        X_train.append(id_list)
        y_train.append(class_tag)
        #i += 1
        #print (i)
    except Exception as e:
        print (e)
        pass

for line in sys.stdin:
    try:
        line=line.strip().split('\t')
        ids=line[1].split(' ')
        for item in ids:
            id_list.append(int(item))
        X_test.append(id_list)
    except:pass
#del datas_train
y_train=np.array(y_train)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print (type(X_train))
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print (X_train)
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                                    metrics=['accuracy'])

print('Train...')
print(X_train.shape)
print(y_train.shape)
#model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1)

#print (model.predict_classes(X_test, batch_size=128, verbose=1))
classes = model.predict_classes(X_test, batch_size=128, verbose=1)
print("final result:")
for item in classes:
    print(item)



