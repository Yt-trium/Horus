from keras import Sequential
from keras.utils import plot_model

from Horus.configuration import *
from Horus.models import *
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

model = model1()

plot_model(model, to_file='model.png')

model.summary()
print(model.input_shape)
print(model.output_shape)

model.fit(gd_train, mra_train, epochs=10, verbose=1)

mra_test = np.empty((dataset_training_size, dataset_image_size_x, dataset_image_size_y, dataset_image_size_z))
for i in range(dataset_training_size, dataset_training_size + dataset_testing_size):
    mra_test[i-dataset_training_size, :, :, :] = nib.load(os.path.join(dataset_mra_path, str(i).zfill(3) + '.nii.gz')).get_data()

mra_test = mra_test.reshape(mra_test.shape[0], dataset_image_size_x, dataset_image_size_y, dataset_image_size_z, 1)

predictions = model.predict(mra_test)

print(predictions.shape)

import pickle

pickle.dump( predictions, open( "predictions.p", "wb" ) )
