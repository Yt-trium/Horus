from keras import Sequential
from keras.layers import Dense, MaxPooling3D, Conv3D

from Horus.configuration import *
import os
import nibabel as nib
import numpy as np
np.random.seed(42)

gd  = np.empty((dataset_training_size, dataset_image_size_x, dataset_image_size_y, dataset_image_size_z))
mra = np.empty((dataset_training_size, dataset_image_size_x, dataset_image_size_y, dataset_image_size_z))

# Load training dataset
for i in range(0,dataset_training_size):
    gd[i, :, :, :] = nib.load(os.path.join(dataset_gd_path, str(i).zfill(2)+'.nii.gz')).get_data()
    mra[i, :, :, :] = nib.load(os.path.join(dataset_mra_path, str(i).zfill(2)+'.nii.gz')).get_data()

gd_with_chan = gd.reshape(dataset_training_size, 1, dataset_image_size_x, dataset_image_size_y, dataset_image_size_z)
mra_with_chan = mra.reshape(dataset_training_size, 1, dataset_image_size_x, dataset_image_size_y, dataset_image_size_z)

gd_flat = gd.reshape((dataset_training_size,dataset_image_size_x*dataset_image_size_y*dataset_image_size_z))
gd_unflat = gd_flat.reshape((dataset_training_size,dataset_image_size_x,dataset_image_size_y,dataset_image_size_z))

model = Sequential()
model.add(Conv3D(16, 16, activation='relu',input_shape=(1,32,32,32)))
model.add(MaxPooling3D(4))
model.add(Conv3D(4, 4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='softmax'))

print(model.input_shape)
print(model.output_shape)