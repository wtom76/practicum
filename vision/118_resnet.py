from tensorflow import keras
import numpy as np

RND_SEED = 12345

def train_generator():
    return keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255.,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
#        rotation_range=90,
    )

def load_train(path):
    return train_generator().flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='training',
        seed=RND_SEED)

def create_model(input_shape, weights=None):
    backbone = keras.applications.resnet.ResNet50(
        input_shape=input_shape,
        classes=12,
        include_top=False,
        weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' if not weights else weights
    )
    backbone.trainable = True
    model = keras.models.Sequential()
    model.add(backbone)
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(12, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy', metrics=['acc'])
    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=5,
               steps_per_epoch=None, validation_steps=None):
    model.fit(train_data, 
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)
    return model