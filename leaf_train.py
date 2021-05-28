import sys
import os
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import ResizeMethod
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
# from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, SeparableConv2D
from tensorflow.python.keras import backend as K
from tensorflow import keras, image
#import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from numpy.testing import assert_allclose

K.clear_session()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'temp')

data_entrenamiento = './dataset/train'
data_validacion = './dataset/test'

# parametros
epocas = 600
tamaño_imagen = (64, 64)
batch_size = 128
pasos = int(100 / batch_size)
pasos_validacion = int(200 / batch_size)
filtrosCov1 = 32
filtrosCov2 = 64
tamaño_filtro1 = (3, 3)
tamaño_filtro2 = (2, 2)
tamaño_pool = (2, 2)
clases = 3
lr = 0.0005


# pre procesamiento de imagenes
def procesamiento_imagenes():
    entrenamiento_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True
    )

    validacion_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, vertical_flip=True)

    imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
        data_entrenamiento,
        target_size=tamaño_imagen,
        batch_size=batch_size,
        classes=listdir(data_entrenamiento),
        class_mode='categorical',
        interpolation='bicubic'
    )
    imagen_validacion = validacion_datagen.flow_from_directory(
        data_validacion,
        target_size=tamaño_imagen,
        batch_size=batch_size,
        classes=listdir(data_validacion),
        class_mode='categorical',
        interpolation='bicubic'
    )
    return imagen_entrenamiento, imagen_validacion


# crear la red CNN
def make_model(nro_classes):
    classifier = Sequential()

    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.2))

    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.2))

    classifier.add(Conv2D(128, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.4))

    classifier.add(Flatten())

    classifier.add(Dense(activation='relu', units=64))
    classifier.add(Dense(activation='relu', units=128))
    classifier.add(Dense(activation='relu', units=64))
    classifier.add(Dense(activation='softmax', units=nro_classes))
    return classifier


# define the checkpoint
filepath = "train_model_checkpoint_loss_best.h5"
filepath_weights = "weights.best-loss.h5"
filepath_accuracy = "train_accuracy_model_checkpoint-best.h5"
callbacks = [
    # keras.callbacks.EarlyStopping(patience=4),
    keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='auto', monitor='val_loss'),
    keras.callbacks.ModelCheckpoint(filepath_weights, save_weights_only=True, verbose=1, save_best_only=True,
                                    mode='auto', monitor='val_loss'),
    keras.callbacks.ModelCheckpoint(filepath_accuracy, verbose=1, save_best_only=True, mode='auto',
                                    monitor='val_accuracy')
]


def train():
    if os.path.isfile(filepath):
        print("continua entrenamiento")
        continue_train()
    else:
        print("nuevo entrenamiento")
        new_train()


def new_train():
    imagen_entrenamiento, imagen_validacion = procesamiento_imagenes()
    model = make_model(len(os.listdir(data_entrenamiento)))
    keras.utils.plot_model(model, show_shapes=True)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])

    history = model.fit(imagen_entrenamiento, batch_size=batch_size, epochs=epocas, validation_data=imagen_validacion,
                        callbacks=callbacks)
    #plot_train(history)

    dir = './modelo'

    if not os.path.exists(dir):
        os.mkdir(dir)
    model.save('./modelo/modelo.h5')
    model.save_weights('./modelo/pesos.h5')


def continue_train():
    imagen_entrenamiento, imagen_validacion = procesamiento_imagenes()
    new_model = keras.models.load_model(filepath)
    '''assert_allclose(model.predict(imagen_entrenamiento),
                    new_model.predict(imagen_entrenamiento),
                    lr)'''
    history = new_model.fit(
        imagen_entrenamiento, epochs=epocas, callbacks=callbacks, validation_data=imagen_validacion,
    )
    #plot_train(history)

'''
def plot_train(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val', 'loss'], loc='upper left')
    plt.show()'''


def predict(file):
    if os.path.isfile(filepath):
        model = load_model("./model_resnet.h5")
        model.load_weights('./model_weights.h5')
        x = load_img(os.path.join(UPLOAD_FOLDER, file), target_size=tamaño_imagen)
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        arreglo = model.predict(x)
        os.remove(os.path.join(UPLOAD_FOLDER, file))  # delete temporary file
        return arreglo.tolist()
        '''return (
                "La imagen es %.2f por ciento boro, %.2f por ciento fosforo, %.2f por ciento magnesio, %.2f por ciento nitrogeno and  %.2f por ciento potasio."
                % (100 * score[0], 100 * score[1], 100 * score[2], 100 * score[3], 100 * score[4]))'''
    else:
        return ("Aun no se ha entrenado ningun modelo")


def predict_image(img):
    if os.path.isfile(filepath):
        model = load_model(filepath)
        x = image.resize(
            img, tamaño_imagen, method=ResizeMethod.BICUBIC, preserve_aspect_ratio=False,
            antialias=False, name=None
        )
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        arreglo = model.predict(x)
        score = arreglo[0]
        return (
                "La imagen es %.2f por ciento boro, %.2f por ciento fosforo, %.2f por ciento magnesio, %.2f percent nitrogeno and  %.2f por ciento potasio."
                % (100 * score[0], 100 * score[1], 100 * score[2], 100 * score[3], 100 * score[4]))
    else:
        return ("Aun no se ha entrenado ningun modelo")


# continue_train()
#new_train()
# predict("./1586_fosforo.jpg")
# predict("./nitrogeno.21.jpg")
