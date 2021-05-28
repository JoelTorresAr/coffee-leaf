from decouple import config as config_decouple
from flask import Flask, flash, render_template, redirect, make_response, jsonify, request, url_for
from flask_session.__init__ import Session
import os
from config import config
import pathlib
from werkzeug.utils import secure_filename
import wget
from tensorflow.python.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from numpy import expand_dims

# ENVIROMENT VAR
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
enviroment = config['development']
# Model saved with Keras model.save()
MODEL_PATH = './models/checkpoint_best_model_accuracy.h5'
MODEL_URL = 'https://github.com/DARK-art108/Cotton-Leaf-Disease-Prediction/releases/download/v1.0/model_resnet.hdf5'
# Download model if not present
while not pathlib.Path(MODEL_PATH).is_file():
    print(f'Model {MODEL_PATH} not found. Downloading...')
    wget.download(MODEL_URL)

if config_decouple('PRODUCTION', default=False):
    enviroment = config['production']


def create_app(enviroment):
    app = Flask(__name__)
    app.config.from_object(enviroment)
    with app.app_context():
        sess = Session()
        sess.init_app(app)
    return app


def predecir(filename):
    if os.path.isfile(MODEL_PATH):
        model = load_model(MODEL_PATH)
        x = load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(64, 64))
        x = img_to_array(x)
        x = expand_dims(x, axis=0)
        arreglo = model.predict(x)
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # delete temporary file
        return arreglo.tolist()


app = create_app(enviroment)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# GET METHOD
@app.route("/", methods=['GET', 'POST'])
def template():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.mkdir(app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            x = predecir(filename)  # imported from process file
            res = make_response(jsonify(x), 200)
            return res
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
