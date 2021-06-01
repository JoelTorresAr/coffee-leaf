import os
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from tensorflow.python.keras import backend as K
import tensorflow.lite as lt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH  = './models/keras_model.h5'
LABELS_PATH = './models/labels.txt'
LABELS_LINE = open(LABELS_PATH, "r")
LABELS      = []
K.clear_session()

for line in LABELS_LINE:
  stripped_line = line.strip()
  LABELS.append(stripped_line)

training_data = './dataset/train'
validation_data = './dataset/test'
size = (224, 224)
batch_size = 32


def predecir(file_path):
    np.set_printoptions(suppress=True)
    model = tensorflow.keras.models.load_model(MODEL_PATH)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(file_path).convert('RGB')
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    #prediction[0] = np.rint(prediction[0])
    prediction[0] = np.dot(prediction[0], 100)
    final_pred = {}
    for i in range(len(LABELS)):
        final_pred[LABELS[i]] = prediction[0][i]
    os.remove(file_path)  # delete temporary file
    return final_pred


def train():
    K.clear_session()
    callbacks = [
        tensorflow.keras.callbacks.ModelCheckpoint(MODEL_PATH, verbose=2, save_best_only=True, mode='max',
                                        monitor='val_accuracy')
    ]
    training_images, validation_images = procesamiento_imagenes()
    model = tensorflow.keras.models.load_model(MODEL_PATH)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(
        training_images, epochs=50, callbacks=callbacks, validation_data=validation_images,
    )
    #model.save(MODEL_PATH)


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
        training_data,
        target_size=size,
        batch_size=batch_size,
        classes=os.listdir(training_data),
        class_mode='categorical',
        interpolation='bicubic'
    )
    imagen_validacion = validacion_datagen.flow_from_directory(
        validation_data,
        target_size=size,
        batch_size=batch_size,
        classes=os.listdir(validation_data),
        class_mode='categorical',
        interpolation='bicubic'
    )
    return imagen_entrenamiento, imagen_validacion


def convert_to_lite():
    model = tensorflow.keras.models.load_model(MODEL_PATH)
    converter = lt.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('models/model_unquant.tflite', 'wb') as f:
        f.write(tflite_model)




