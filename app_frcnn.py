from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

import warnings
warnings.filterwarnings("ignore")

import os
from base64 import b64encode
from datetime import timedelta

from flask import Flask, request, redirect, render_template, session, jsonify
from flask_session import Session
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import io


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = Flask(__name__)

app.config['SESSION_PERMANENT'] = True
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=5)
app.config['SESSION_FILE_THRESHOLD'] = 20

app.secret_key = os.urandom(10)
Session().init_app(app)

cfg = get_cfg()
cfg.merge_from_file("configs/faster_rcnn.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "/home/juravlik/Downloads/faster_rcnn.pth"

predictor = DefaultPredictor(cfg)

def get_class():
    """
    retrieve image file or url parameter from request information and search for similar images.
    Returns:
        Tuple of original image base64 encoded and list of paths to similar images, ordered by similarity.
    """
    if "image" in request.files and request.files["image"].filename != "":
        np_file = np.fromfile(request.files["image"], np.uint8)
        target_image = cv2.imdecode(np_file, cv2.IMREAD_COLOR)
    else:
        url = request.form["url"]
        r = requests.get(url, allow_redirects=True)
        stream = BytesIO(r.content)
        image = Image.open(stream).convert("RGB")
        stream.close()
        target_image = np.array(image)

    outputs = predictor(target_image)

    v = Visualizer(target_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

    for box, score in zip(outputs["instances"].pred_boxes.to('cpu'), outputs["instances"].scores.to('cpu')):
        v.draw_box(box, edge_color='blue')
        # v.draw_text(
        #     'airplane ' + str(np.around(score.numpy(), 5) * 100) + '%',
        #     tuple(box[:2].numpy()),
        #     color='green',
        #     font_size=12,
        # )

    v = v.get_output()

    target_image = v.get_image()

    file_object = io.BytesIO()
    img = Image.fromarray(target_image.astype('uint8'))
    img.save(file_object, 'PNG')
    target_image = "data:image/png;base64," + b64encode(file_object.getvalue()).decode('ascii')

    return target_image


@app.route('/process', methods=['POST'])
def process():
    if request.files or request.form:
        if "image" in request.files and request.files["image"].filename == "" and request.form["url"] == "":
            return jsonify({})
        target_image = get_class()

        return jsonify({"target_image": target_image})
    else:
        return jsonify({})


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        if request.files or request.form:
            if request.files["image"].filename == "" and request.form["url"] == "":
                return redirect('/')
            target_image = get_class()
            session['target_image'] = target_image

            return redirect('/')
    else:
        target_image = session.get('target_image')
        if target_image:

            delete_sessions()

            return render_template('index_frcnn.html', target_image=target_image)
        else:
            return render_template('index_frcnn.html')


def delete_sessions():
    session.pop('target_image', None)


print("Service-started")

if __name__ == "__main__":
    app.run(debug=False, port=5000, host='0.0.0.0')


