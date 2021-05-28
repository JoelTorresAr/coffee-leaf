from decouple import config as config_decouple
from flask import Flask, flash, render_template, redirect, make_response, jsonify, request
from flask_session.__init__ import Session
import os
from config import config
from werkzeug.utils import secure_filename
import lite as tflite
import numpy as np
from PIL import Image

# ENVIROMENT VAR
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
enviroment = config['development']
# Model saved with Keras model.save()
MODEL_PATH = './models/checkpoint_best_model_accuracy.h5'

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
        i = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)).convert('RGB').resize((64, 64),
                                                 Image.ANTIALIAS)
        x = np.array(i)
        x = np.expand_dims(x, axis=0)
        f = tflite.Interpreter('lite_model.tflite')
        f.allocate_tensors()
        '''_, height, width, _ = f.get_input_details()[0]['shape']
        results = classify_image(f, i)'''
        ######
        i = f.get_input_details()[0]
        o = f.get_output_details()[0]
        f.set_tensor(i['index'], x)
        f.invoke()
        y = f.get_tensor(o['index'])
        print("TensorFlow Lite:", y[0])
        return y.tolist()


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
