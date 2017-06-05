import pandas as pd
import numpy as np
import os
import sys
import gzip

from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential
from keras.utils import np_utils

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path)


EPOCH = 400
BATCH = 50
nb_classes = 2

PL     = 60484   # 38 + 60483
PS     = 60483   # 60483
F_MAX = 34.0
DR    = 0.1      # Dropout rate

def load_data():
        train_path = 'nt_train2.csv'
        test_path = 'nt_test2.csv'

        df_train = (pd.read_csv(train_path,header=None).values).astype('float32')
        df_test = (pd.read_csv(test_path,header=None).values).astype('float32')

	print('df_train shape:', df_train.shape)
	print('df_test shape:', df_test.shape)

        df_y_train = df_train[:,0].astype('int')
        df_y_test = df_test[:,0].astype('int')

        Y_train = np_utils.to_categorical(df_y_train,nb_classes)
        Y_test = np_utils.to_categorical(df_y_test,nb_classes)
              
        df_x_train = df_train[:, 1:PL].astype(np.float32)
        df_x_test = df_test[:, 1:PL].astype(np.float32)
            
#        X_train = df_x_train.as_matrix()
#        X_test = df_x_test.as_matrix()
            
        X_train = df_x_train
        X_test = df_x_test
            
        scaler = MaxAbsScaler()
        mat = np.concatenate((X_train, X_test), axis=0)
        mat = scaler.fit_transform(mat)
        
        X_train = mat[:X_train.shape[0], :]
        X_test = mat[X_train.shape[0]:, :]
        
        return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = load_data()

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)


model = Sequential()
model.add(Dense(1800, input_shape=(PS,)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(180))
model.add(Activation('sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=BATCH, nb_epoch=EPOCH,
                    verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])


