import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, CuDNNLSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import poetfuncs
import sys



def first_model(X, Y):
    model = Sequential()
    model.add(CuDNNLSTM(750, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(750))
    model.add(Dropout(0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))
    return model

def gigantic_model(X, Y):
    model = Sequential()
    model.add(CuDNNLSTM(450, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(450, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(450))
    model.add(Dropout(0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))
    return model


def generator(X, Y, batch_size, count):


    while True:
        if count >= 2250:
            count=0
        elif count % 20 == 0:
            print("{}/2250".format(count))

        count +=1

        index = np.random.choice(X.shape[0], batch_size)
        X_not_hot = X[index]
        Y_one_hot = map1.one_hot_label(Y[index])
        yield X_not_hot, Y_one_hot


map1 = poetfuncs.TextMapping(file="data/pessoa.txt")
X, Y = map1.word_map(seq_size=30)
print(X.shape)
print(Y.shape)
print("got X and Y")


index = np.random.choice(X.shape[0], 125)
dim_X_one_hot = X[index] #map1.faster_one_hot_feat(X[index])
dim_Y_one_hot = map1.one_hot_label(Y[index])
#print(dim_Y_one_hot.shape)


# model = first_model(dim_X_one_hot, dim_Y_one_hot)
model = gigantic_model(dim_X_one_hot, dim_Y_one_hot)

model.compile(loss='categorical_crossentropy', optimizer='adam')
filepath = "pessoa5-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
print(X.shape)
print(Y.shape)
# model.fit(X, Y, epochs=100, batch_size=125, callbacks=callbacks_list)
count = 0
model.fit_generator(generator(X, Y, batch_size=125, count=count), steps_per_epoch=2250, nb_epoch=200, verbose=10, callbacks=callbacks_list)

model.save_weights('final_weights_pessoa5.h5')

# model.load_weights('pessoa4/pessoa4-150-0.80.hdf5')

seed_idx = 3
print('seed: {}'.format(map1.seed_word_to_text(X[seed_idx])))
seed = np.reshape(X[seed_idx], (1, X.shape[1], X.shape[2]))
out = map1.generate_text_word(model=model, seed=seed, out_size=90)
print('outp: {}'.format(out))

