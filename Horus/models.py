from keras import Sequential, Input, Model
from keras.layers import Dense, MaxPooling3D, Conv3D, Flatten, Dropout, Activation, BatchNormalization, UpSampling3D, \
    merge, Concatenate, concatenate
from keras.optimizers import SGD, Adam

from Horus.configuration import *

def model1():
    # Input layer
    input = Input(shape=(dataset_image_size_x, dataset_image_size_y, dataset_image_size_z, 1))

    # convolutional layer 1
    conv_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=(32, 32, 32, 1), name="conv_1_c1")(input)
    # conv_1 = Dropout(0.2, name="conv_1_d1")(conv_1)
    conv_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', name="conv_1_c2")(conv_1)
    # conv_1 = BatchNormalization(name="conv_1_n1")(conv_1)
    pool_1 = MaxPooling3D((2, 2, 2), name="pool_1_p1")(conv_1)

    # convolutional layer 2
    conv_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name="conv_2_c1")(pool_1)
    # conv_2 = Dropout(0.2, name="conv_2_d1")(conv_2)
    conv_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name="conv_2_c2")(conv_2)
    pool_2 = MaxPooling3D((2, 2, 2), name="pool_2_p1")(conv_2)

    # convolutional layer 3
    conv_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name="conv_3_c1")(pool_2)
    # conv_3 = Dropout(0.2, name="conv_3_d1")(conv_3)
    conv_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name="conv_3_c2")(conv_3)
    pool_3 = MaxPooling3D((2, 2, 2), name="pool_3_p1")(conv_3)

    # convolutional layer 4
    conv_4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name="conv_4_c1")(pool_3)
    conv_4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name="conv_4_c2")(conv_4)
    conv_4 = Dropout(0.4, name="conv_4_d1")(conv_4)
    pool_4 = MaxPooling3D((2, 2, 2), name="pool_4_p1")(conv_4)

    # convolutional layer 5
    conv_5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name="conv_5_c1")(pool_4)
    conv_5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name="conv_5_c2")(conv_5)
    conv_5 = Dropout(0.4, name="conv_5_d1")(conv_5)

    # Up sampling 5
    up_5 = UpSampling3D((2, 2, 2), name="up_5_u1")(conv_5)
    up_5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name="up_5_c1")(up_5)
    # Merge layer 4 and up 5
    merge_4 = concatenate([conv_4, up_5], axis=4, name="merge_4")

    # convolutional layer 6
    conv_6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name="conv_6_c1")(merge_4)
    conv_6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name="conv_6_c2")(conv_6)

    # Up sampling 4
    up_4 = UpSampling3D((2, 2, 2), name="up_4_u1")(conv_6)
    up_4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name="up_4_c1")(up_4)
    # Merge layer 3 and up 4
    merge_3 = concatenate([conv_3, up_4], axis=4, name="merge_3")

    # convolutional layer 7
    conv_7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name="conv_7_c1")(merge_3)
    conv_7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name="conv_7_c2")(conv_7)

    # Up sampling 3
    up_3 = UpSampling3D((2, 2, 2), name="up_3_u1")(conv_7)
    up_3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name="up_3_c1")(up_3)
    # Merge layer 2 and up 3
    merge_2 = concatenate([conv_2, up_3], axis=4, name="merge_2")

    # convolutional layer 8
    conv_8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name="conv_8_c1")(merge_2)
    conv_8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name="conv_8_c2")(conv_8)

    # Up sampling 2
    up_2 = UpSampling3D((2, 2, 2), name="up_2_u1")(conv_8)
    up_2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', name="up_2_c1")(up_2)
    # Merge layer 1 and up 2
    merge_1 = concatenate([conv_1, up_2], axis=4, name="merge_1")

    # convolutional layer 9
    conv_9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', name="conv_9_c1")(merge_1)
    conv_9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', name="conv_9_c2")(conv_9)
    conv_9 = Conv3D(2, (3, 3, 3), activation='relu', padding='same', name="conv_9_c3")(conv_9)

    # convolutional layer 10
    conv_10 = Conv3D(1, 1, activation = 'sigmoid', name="conv_10_c1")(conv_9)

    model = Model(input=input, output=conv_10)

    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model