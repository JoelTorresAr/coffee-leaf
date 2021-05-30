#from decouple import config as config_decouple
from flask import Flask, flash, render_template, redirect, make_response, jsonify, request, url_for
import json
from flask_session.__init__ import Session
import os
from config import config
import base64
import werkzeug
import leaf_engine_lite
import numpy as np

# ENVIROMENT VAR
werkzeug.cached_property = werkzeug.utils.cached_property
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
enviroment = config['development']

'''if config_decouple('PRODUCTION', default=False):
    enviroment = config['production']'''


def create_app(enviroment):
    app = Flask(__name__)
    app.config.from_object(enviroment)
    with app.app_context():
        sess = Session()
        sess.init_app(app)
    return app


app = create_app(enviroment)

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


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
            filename = werkzeug.utils.secure_filename(file.filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.mkdir(app.config['UPLOAD_FOLDER'])
            path_save = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path_save)
            x = leaf_engine_lite.predecir(path_save)  # imported from process file
            res = make_response(json.dumps(x, cls=NumpyFloatValuesEncoder), 200)
            return res
    return render_template('index.html')

# GET METHOD


@app.route("/api_predict", methods=['POST'])
def api_predict():
    path_save = os.path.join(app.config['UPLOAD_FOLDER'], "imageToSave.png")
    if request.method == 'POST':
        image_64_encode = request.json['image_base64']
        image_64_decode = image_64_encode.replace("data:image/jpeg;base64,", "")
        with open(path_save, "wb") as fh:
            fh.write(base64.b64decode(image_64_decode))
            fh.close()
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        x = leaf_engine_lite.predecir(path_save)  # imported from process file
        res = make_response(json.dumps(x, cls=NumpyFloatValuesEncoder), 200)
        return res

    return make_response(jsonify({"error": "Utiliza metodo POST; application/jso;"}), 400)


def list_routes():
    output = []
    for rule in app.url_map.iter_rules():

        options = {}
        for arg in rule.arguments:
            options[arg] = "[{0}]".format(arg)

        methods = ','.join(rule.methods)
        url = url_for(rule.endpoint, **options)
        line = rule.endpoint + ' ' + methods + ' ' + url
        output.append(line)

    for line in sorted(output):
        print(line)


if __name__ == '__main__':
    PORT = 5000
    HOST = '0.0.0.0'
    app.run(host=HOST, port=PORT)
