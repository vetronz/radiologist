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

pts_dynamic_abs = '/home/patrick/disp/data/pts_dynamic'
label_dict = np.load('label_dict.npy', allow_pickle='TRUE').item()
os.chdir(pts_dynamic_abs)

new_size = 350

aug_perm = 10

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.1,
    horizontal_flip=True)

# model
reg_alpha = 0.18
drop_perc = 0.3
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

checkpoint = ModelCheckpoint('/home/patrick/disp/dnn/weights_best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='max')

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-1, patience=2, verbose=1, mode='auto', restore_best_weights=True)

callback_l = [checkpoint, monitor]

class_weight = {0: 1,
                1: 2}

# construct array 0 to length of num imgs
len_X = len(os.listdir(pts_dynamic_abs))
# hard set len to limit dataset for debug
len_X = 3000
idx = np.arange(0, len_X)

# inplace shuffle
np.random.seed(42)
np.random.shuffle(idx)

train_idx = idx[0:round(len_X)]

# split 80 20 %
t_prop = 0.6
# train_idx = idx[0:round(len_X*t_prop)]
# val_idx = idx[round(len_X*t_prop):]

# # # def the val set
# X_test = []
# y_test = []
# for i in val_idx:
#     ct_id = 'ct_' + str(i)
#     img = np.load(ct_id + '.npy')
#     label = label_dict[ct_id]
#     X_test.append(img)
#     y_test.append(label)

# define the batch params for the train set
bs = 32
num_train_img = len(train_idx)
# num_val_img = len(val_idx)
num_batches = int(np.floor(num_train_img/bs))
print('\nnum batches: '+str(num_batches))
num_epochs = 1

indexes = np.arange(num_batches)

t_start = time.time()
for i in indexes:
    print(f'\nbatch: {i} of: {len(indexes)}\n')
    batch_l = train_idx[(i*bs):(i+1)*bs]
    train_batch_idx = batch_l[0:round(len(batch_l)*t_prop)]
    val_batch_idx = batch_l[round(len(batch_l)*t_prop):]
    # print(batch_l)
    time.sleep(0.02)

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for j in train_batch_idx:
        ct_id = 'ct_' + str(j)
        if label_dict[ct_id] == 1:
            # augment
            print(f'augmenting img: {ct_id}')
            img = np.load(ct_id + '.npy').reshape(1, new_size, new_size, 1)
            aug_img = datagen.flow(img)
            aug_img_l = [next(aug_img)[0].astype(np.uint8) for i in range(aug_perm)]
            for k in aug_img_l:
                print(k.shape)
                X_train.append(k)
                y_train.append(label_dict[ct_id])
        else:
            # unaugmented
            print(f'adding non aug img: {ct_id}')
            img = np.load(ct_id + '.npy')
            print(img.shape)
            X_train.append(img)
            y_train.append(label_dict[ct_id])
        time.sleep(1)
        
    for j in val_batch_idx:
        ct_id = 'ct_' + str(j)
        # unaugmented
        X_test.append(np.load(ct_id + '.npy'))
        y_test.append(label_dict[ct_id])
        time.sleep(0.02)

    # convert from list to array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = X_train.reshape(-1, new_size, new_size, 1)

    print(f'\nX_train shape: {X_train.shape}')
    print(f'\ny_train shape: {y_train.shape}')

    time.sleep(1)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test = X_test.reshape(-1, new_size, new_size, 1)

    print(f'\n X_test shape:{X_test.shape}\n')
    print(f'\n y_test shape:{y_test.shape}\n')
    time.sleep(0.01)

    # fit model
    print('fitting model')
    model.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=callback_l, epochs=num_epochs, class_weight=class_weight)

time.sleep(1)

t_end = time.time()
t_tot = t_end - t_start
print(f'\ntotal time: {t_tot}')

dnn_dir = '/home/patrick/disp/dnn'
os.chdir(dnn_dir)

model.save('batch_mod.h5')  # creates a HDF5 file 'my_model.h5'
# returns a compiled model
# identical to the previous one
# model = load_model('my_model.h5')

