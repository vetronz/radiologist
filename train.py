import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import pickle

def construct_data(index_list):
    X = []
    y = []
    for i in index_list:
        ct_id = 'ct_' + str(i)
        img = np.load(ct_id + '.npy').reshape(1, new_size, new_size, 1)
        X.append(img)
        y.append(label_dict[ct_id])
        time.sleep(0.01)
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(-1, new_size, new_size, 1)
    time.sleep(0.01)
    print(f'\n X shape:{X.shape}')
    print(f'\n y shape:{y.shape}\n')
    return X, y

pts_dynamic_abs = '/root/disp/pts_dynamic'
rad_abs = '/root/radiologist'
label_dict = np.load('label_dict.npy', allow_pickle='TRUE').item()
os.chdir(pts_dynamic_abs)

new_size = 350

# model
num_epochs = 1
reg_alpha = 0.1
drop_perc = 0.2
NAME = "ct-CNN"
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', activity_regularizer=regularizers.l2(reg_alpha), input_shape=(new_size, new_size, 1)))
model.add(Dropout(drop_perc))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', activity_regularizer=regularizers.l2(reg_alpha)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(drop_perc))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint('/root/radiologist/dnn/weights_best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='max')

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-1, patience=2, verbose=1, mode='auto', restore_best_weights=True)

callback_l = [checkpoint, monitor]

# data partitioning
# t_prop = 0.8
l = np.arange(1,20)
t_prop_l = [round(i*0.05, 2) for i in l]
# t_prop_l = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# construct array 0 to length of num imgs
len_data = 2500
idx = np.arange(0, len_data)

# inplace shuffle
np.random.seed(42)
np.random.shuffle(idx)

mod_l = []
for count, prop in enumerate(t_prop_l):
    train_idx = idx[0:int(round(len_data*prop))]
    # val_idx = idx[round(len_data*prop):]

    # calling constructor
    X_train, y_train = construct_data(train_idx)
    # X_test, y_test = construct_data(val_idx)

    # start gun
    t_start = time.time()
    time.sleep(0.01)

    # fit model
    print('fitting model')
    
    # mod = model.fit(X_train, y_train, batch_size=32, verbose=1, callbacks=callback_l, epochs=num_epochs, validation_data=(X_test, y_test))
    mod = model.fit(X_train, y_train, batch_size=32, verbose=1, callbacks=callback_l, epochs=num_epochs, validation_split = 0.5)

    mod_l.append(mod.history)

    # calling time
    time.sleep(1)
    t_end = time.time()
    t_tot = t_end - t_start
    print(f'\ntotal time: {t_tot}')

dnn_dir = '/root/radiologist/dnn'
os.chdir(dnn_dir)

 # creates a HDF5 file 'my_model.h5'
model.save('batch_mod.h5')
# model = load_model('my_model.h5')

os.chdir(rad_abs)
with open('hist_dict.pkl', 'wb') as f:
    pickle.dump(mod_l, f)
# with open('hist_dict.pkl', 'rb') as f:
#     pickle.load(f)