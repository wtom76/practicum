from tensorflow import keras
import pandas as pd
import numpy as np

def generator():
    return keras.preprocessing.image.ImageDataGenerator(
            validation_split=0.25,
            rescale=1./255.,
#            horizontal_flip=True,
#            vertical_flip=True,
#            width_shift_range=0.2,
#            height_shift_range=0.2,
#            rotation_range=5,
    )

def load_subset(path, subset):
    return generator().flow_from_dataframe(
        pd.read_csv(path + '/labels.csv'),
        directory=path + '/final_files',
        x_col='file_name',
        y_col='real_age',
        target_size=(150, 150),
        class_mode='raw',
        subset=subset,
        batch_size=16,
        seed=42
)

def load_train(path):
	return load_subset(path, subset='training')

def load_test(path):
	return load_subset(path, subset='validation')

def create_model(input_shape, weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'):
    backbone = keras.applications.resnet.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )
    backbone.trainable = True
    model = keras.models.Sequential()
    model.add(backbone)
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='mean_squared_error', metrics=['mae'])
    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=5, steps_per_epoch=None, validation_steps=None):
    model.fit(train_data, 
              validation_data=test_data,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)
    return model