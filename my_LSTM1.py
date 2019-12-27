import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, CuDNNLSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import poetfuncs

def first_model(X_one_hot, Y_one_hot):
    model = Sequential()
    model.add(CuDNNLSTM(750, input_shape=(X_one_hot.shape[1], X_one_hot.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(750))
    model.add(Dropout(0.2))
    model.add(Dense(Y_one_hot.shape[1], activation='softmax'))
    return model

def gigantic_model(X_one_hot, Y_one_hot):
    model = Sequential()
    model.add(CuDNNLSTM(450, input_shape=(X_one_hot.shape[1], X_one_hot.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(450, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(450))
    model.add(Dropout(0.2))
    model.add(Dense(Y_one_hot.shape[1], activation='softmax'))
    return model


map1 = poetfuncs.TextMapping(file="data/pessoa_clean.txt")
X, Y = map1.character_map(seq_size=100)
print(X.shape)
print(Y.shape)

print("got X and Y")
print(X.shape)
X_one_hot = map1.faster_one_hot_feat(X)
print("got X hot encoded")
Y_one_hot = map1.one_hot_label(Y)
print("got Y hot encoded")

print(X_one_hot.shape)
print(Y_one_hot.shape)
# print(map1.letters_to_nr)

model = first_model(X_one_hot, Y_one_hot)
# model = gigantic_model(X_one_hot, Y_one_hot)

model.compile(loss='categorical_crossentropy', optimizer='adam')
filepath = "pessoa10-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(X_one_hot, Y_one_hot, epochs=200, batch_size=125, callbacks=callbacks_list)

model.save_weights('final_weights_pessoa10.h5')

#model.load_weights('camoes-07-1.14.hdf5')

seed_idx = 3
print('seed: {}'.format(map1.seed_char_to_text(X_one_hot[seed_idx])))
seed = np.reshape(X_one_hot[seed_idx], (1, X_one_hot.shape[1], X_one_hot.shape[2]))
out = map1.generate_text_character(model=model, seed=seed, out_size=180)
print('outp: {}'.format(out))

