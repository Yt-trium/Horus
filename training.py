from keras import Sequential, Input
from keras.layers import Dense, MaxPooling3D, Conv3D, Flatten, Dropout, Activation, BatchNormalization, UpSampling3D
from keras.optimizers import SGD
from keras.utils import np_utils

from Horus.configuration import *
import os
import nibabel as nib
import numpy as np

from keras import backend as K

K.set_image_dim_ordering("tf")
print(K.image_dim_ordering())

np.random.seed(42)

gd = np.empty((dataset_training_size, dataset_image_size_x, dataset_image_size_y, dataset_image_size_z))
mra = np.empty((dataset_training_size, dataset_image_size_x, dataset_image_size_y, dataset_image_size_z))

# Load training dataset
for i in range(1, dataset_training_size):
    gd[i, :, :, :] = nib.load(os.path.join(dataset_gd_path, str(i).zfill(3) + '.nii.gz')).get_data()
    mra[i, :, :, :] = nib.load(os.path.join(dataset_mra_path, str(i).zfill(3) + '.nii.gz')).get_data()

# gd_with_chan = gd.reshape(dataset_training_size, 1, dataset_image_size_x, dataset_image_size_y, dataset_image_size_z)
# mra_with_chan = mra.reshape(dataset_training_size, 1, dataset_image_size_x, dataset_image_size_y, dataset_image_size_z)

gd_train = gd.reshape(gd.shape[0], dataset_image_size_x, dataset_image_size_y, dataset_image_size_z, 1)
mra_train = mra.reshape(mra.shape[0], dataset_image_size_x, dataset_image_size_y, dataset_image_size_z, 1)

input_shape = (dataset_image_size_x, dataset_image_size_y, dataset_image_size_z, 1)

model = Sequential()

model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=(32, 32, 32, 1)))
model.add(BatchNormalization())
model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling3D((2, 2, 2)))

model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(UpSampling3D((2, 2, 2)))
model.add(MaxPooling3D((2, 2, 2)))

model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(UpSampling3D((2, 2, 2)))
model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling3D((2, 2, 2)))

# model.add(Flatten())
# model.add(Dense(128, activation="relu", kernel_initializer="normal"))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(2, kernel_initializer="normal"))
# model.add(Activation('softmax'))

model.add(UpSampling3D((2, 2, 2)))
model.add(Conv3D(1, (1, 1, 1), activation='relu', padding='same'))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer=SGD())
# model.compile(loss='binary_crossentropy',optimizer=SGD())

model.summary()
print(model.input_shape)
print(model.output_shape)

model.fit(gd_train, mra_train, epochs=1, verbose=1)

mra_test = np.empty((dataset_training_size, dataset_image_size_x, dataset_image_size_y, dataset_image_size_z))
for i in range(dataset_training_size, dataset_training_size + dataset_testing_size):
    mra_test[i-dataset_training_size, :, :, :] = nib.load(os.path.join(dataset_mra_path, str(i).zfill(3) + '.nii.gz')).get_data()

mra_test = mra_test.reshape(mra_test.shape[0], dataset_image_size_x, dataset_image_size_y, dataset_image_size_z, 1)

predictions = model.predict(mra_test)

print(predictions.shape)

import pickle

pickle.dump( predictions, open( "predictions.p", "wb" ) )