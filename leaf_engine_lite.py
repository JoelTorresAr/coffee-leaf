import os
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from tensorflow.python.keras import backend as K
import tensorflow.lite as lt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

'''Configuramos el dorectorio raiz y los recursos a utilizar'''
APP_ROOT    = os.path.dirname(os.path.abspath(__file__))
#Modelo entrenado
MODEL_PATH  = os.path.join(APP_ROOT, 'models/keras_model.h5')
#Los nombres de las clases clasificadas
LABELS_PATH = os.path.join(APP_ROOT, 'models/labels.txt')
LABELS_LINE = open(LABELS_PATH, "r")
LABELS      = []
#Clear sesion nos permite iniciar con la memoria limpia
K.clear_session()
#leemos las clases del txt y lo guardamos en un array
for line in LABELS_LINE:
  stripped_line = line.strip()
  LABELS.append(stripped_line)
'''configuramos las rutas de las carpetas que contienen las clases con las imagenes a clasificar,
   las cuales eran utilizadas en el entrenamiento'''
training_data = os.path.join(APP_ROOT, 'dataset/train')
validation_data = os.path.join(APP_ROOT, 'dataset/test')
#tamaño de imagenes con el cual trabajaremos durante el entrenamiento y prediccion de imagenes
size = (224, 224)
#numero de muestras que tomara en cada entrenamiento
batch_size = 32


def predecir(file_path):
    #suprimimos los pequeños resultados
    np.set_printoptions(suppress=True)
    #cargamos el modelo que ya ha sido entrenado
    model = tensorflow.keras.models.load_model(MODEL_PATH)
    #especificamos que recibiremos una imagen con tamaño de 224 x 224 la cual usa 3 canales RGB
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    #cargamos la imagen y la convertimos a RGB
    image = Image.open(file_path).convert('RGB')
    #cambiamos el tamaño de la imagen utilizando un metodo de conversion ANTIALIAS
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    #convertimos la imagen a array y la normalizamos
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    #utilizamos la funcion predict con la imagen que hemos convertido en array
    prediction = model.predict(data)
    #redondeamos los resultados
    prediction[0] = np.round(prediction[0], 6)
    #multiplicamos los resultados por 100
    prediction[0] = np.dot(prediction[0], 100)
    #creamos un diccionario en el cual guardaremos asignaremos a cada clase su porcentaje de similitud
    final_pred = {}
    for i in range(len(LABELS)):
        final_pred[LABELS[i]] = prediction[0][i]
    os.remove(file_path)  # delete temporary file
    return final_pred


def train():
    K.clear_session()
    '''Crearemos un callbacks en la cual especificaremos puntos 
       de guardado por cada ciclo de entrenamiento donde se guardara los ciclos con mejor valor de presición'''
    callbacks = [
        tensorflow.keras.callbacks.ModelCheckpoint(MODEL_PATH, verbose=2, save_best_only=True, mode='max',
                                        monitor='val_accuracy')
    ]
    #optenemos las imagenes a utilizar
    training_images, validation_images = procesamiento_imagenes()
    #cargamos el modelo del cual haremos transfer learning, en este caso el modelo de
    # clasificación de iris la cual usa dos capas ocultas
    model = tensorflow.keras.models.load_model(MODEL_PATH)
    model.summary()
    #congelamos las capas que ya han sido entrenadas para aprender a reconocer formas y patrones
    for layer in model.layers:
        layer.trainable = False
    # configuramos nuestras capa de clasificacion la cual seras las capas a entrenar
    model.add(tensorflow.keras.layers.Dense(100))
    #len(os.listdir(training_data)) -> me devuelve el numero de clases que estoy clasificando
    model.add(tensorflow.keras.layers.Dense(len(os.listdir(training_data))))
    #compilamos el modelo con una funcion de perdida categorical_crossentropy y un optimizador
    # adam, especificando que la metrica sea la precision
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #iniciamos el entrenamiento y espcificamos que entrene por 50 ciclos
    history = model.fit(
        training_images, epochs=50, callbacks=callbacks, validation_data=validation_images,
    )
    #model.save(MODEL_PATH)


def procesamiento_imagenes():
    #creamos un data generator el cual aumentara la data en base a los parametros que configuraremos
    entrenamiento_datagen = ImageDataGenerator(
        rescale=1. / 255, # escalada de imagen
        shear_range=0.3, #cortes de la imagen
        zoom_range=0.3,  # zoom en la imagen
        horizontal_flip=True, # girar imagen horizontalmente
        vertical_flip=True #girar imagen verticalmente
    )

    validacion_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, vertical_flip=True)

    imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
        training_data,
        target_size=size, #tamaño de la imagen con la cual entrenaremos a la red neuronal
        batch_size=batch_size,
        classes=os.listdir(training_data),
        class_mode='categorical',
        interpolation='bicubic' #metodo de conversion de la imagen
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



train()


