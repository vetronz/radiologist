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

def construct_train(index_list):
    X = []
    y = []
    for i in index_list:
        ct_id = 'ct_' + str(i)
        img = np.load(ct_id + '.npy').reshape(1, new_size, new_size, 1)
        aug_img = datagen.flow(img)
        aug_img_l = [next(aug_img)[0].astype(np.uint8) for i in range(aug_perm)]
        for k in aug_img_l:
            k = k.reshape(new_size, new_size)
            X.append(k)
            y.append(label_dict[ct_id])
            time.sleep(0.01)
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(-1, new_size, new_size, 1)
    time.sleep(0.01)
    print(f'\n X shape:{X.shape}')
    print(f'\n y shape:{y.shape}\n')
    return X, y

def construct_val(index_list):
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

def define_model(reg_alpha, drop_perc):
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
    return model


checkpoint = ModelCheckpoint('/root/radiologist/dnn/weights_best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='max')

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-1, patience=2, verbose=1, mode='auto', restore_best_weights=True)

callback_l = [checkpoint, monitor]

new_size = 350
aug_perm=10
bs = 64
# construct array 0 to length of num imgs
len_data = 2500
idx = np.arange(0, len_data)

# inplace shuffle
np.random.seed(42)
np.random.shuffle(idx)

# data partitioning
t_prop = 0.8

train_idx = idx[0:round(len_data*t_prop)]
val_idx = idx[round(len_data*t_prop):]

X_train, y_train = construct_train(train_idx)
X_test, y_test = construct_val(val_idx)

model = define_model(0, 0)
mod = model.fit(X_train, y_train, batch_size=bs, verbose=1, callbacks=callback_l, epochs=num_epochs, validation_data=(X_test, y_test))


dnn_dir = '/root/radiologist/dnn'
os.chdir(dnn_dir)
model.save('batch_mod.h5')
# model = load_model('my_model.h5')








