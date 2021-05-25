import os
from base64 import b64encode
from datetime import timedelta
from typing import Tuple, List

from flask import Flask, request, redirect, render_template, session, jsonify
from flask_session import Session
import csv
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import io
import sys

from nn.utils import get_param_from_config, object_from_dict
from nn.data import get_valid_aug_preproc

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = Flask(__name__)

app.config['SESSION_PERMANENT'] = True
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=5)
app.config['SESSION_FILE_THRESHOLD'] = 20

app.secret_key = os.urandom(10)
Session().init_app(app)

basedir = os.path.abspath(os.path.dirname(__file__))

config_path = os.path.join(basedir, 'configs/application_config.yaml')
config = get_param_from_config(config_path)

device = torch.device(config.device)
model_checkpoint_path = config.model_checkpoint_path
UPLOADED_PATH = os.path.join(basedir, config.UPLOADED_PATH)

DATA_PATH = config.DATA_PATH
print("Loading model")
model_checkpoint = torch.load(model_checkpoint_path, map_location=device)

model = object_from_dict(model_checkpoint['config']['model'])
model.load_state_dict(model_checkpoint['model'])
model.to(device)
model.eval()
preprocessing_for_model = get_valid_aug_preproc(model.get_preprocess_fn())

label_to_class = dict()
dirs = os.listdir(config.PATH_TO_DIRS)
for i, d in enumerate(dirs):
    label_to_class[i] = d

print(label_to_class)

def get_class():
    """
    retrieve image file or url parameter from request information and search for similar images.
    Returns:
        Tuple of original image base64 encoded and list of paths to similar images, ordered by similarity.
    """
    if "image" in request.files and request.files["image"].filename != "":
        np_file = np.fromfile(request.files["image"], np.uint8)
        target_image = cv2.imdecode(np_file, cv2.IMREAD_COLOR)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    else:
        url = request.form["url"]
        r = requests.get(url, allow_redirects=True)
        stream = BytesIO(r.content)
        image = Image.open(stream).convert("RGB")
        stream.close()
        target_image = np.array(image)

    target_image_after_preproc = preprocessing_for_model(image=target_image)['image']
    target_image_after_preproc = target_image_after_preproc.unsqueeze(0)

    classes = model(target_image_after_preproc.to(device)).tolist()[0]

    classes = [[label_to_class[i], c] for i, c in enumerate(classes)]
    classes.sort(key=lambda x: x[1], reverse=True)

    classes = [[i, '{0:.8f}'.format(c)] for [i, c] in classes]

    file_object = io.BytesIO()
    img = Image.fromarray(target_image.astype('uint8'))
    img.save(file_object, 'PNG')
    target_image = "data:image/png;base64," + b64encode(file_object.getvalue()).decode('ascii')

    return target_image, classes


@app.route('/process', methods=['POST'])
def process():
    if request.files or request.form:
        if "image" in request.files and request.files["image"].filename == "" and request.form["url"] == "":
            return jsonify({})
        target_image, classes = get_class()

        return jsonify({"target_image": target_image, "classes": classes})
    else:
        return jsonify({})


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        if request.files or request.form:
            if request.files["image"].filename == "" and request.form["url"] == "":
                return redirect('/')
            target_image, classes = get_class()
            session['classes'] = classes
            session['target_image'] = target_image

            return redirect('/')
    else:
        classes = session.get('classes')
        target_image = session.get('target_image')
        if classes and target_image:

            delete_sessions()

            return render_template('index.html', target_image=target_image, classes=classes)
        else:
            return render_template('index.html')


def delete_sessions():
    session.pop('classes', None)
    session.pop('target_image', None)


print("Service-started")

if __name__ == "__main__":
    app.run(debug=False, port=5000, host='0.0.0.0')
