# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: kristine
# mail:   kristine@mail.sim.ac.cn
# Date:   2020/3/21

import io
import os
import json
import time
import flask
import exifutil
import logging
import urllib
import base64
from io import BytesIO
import datetime
import torch
from ssd.config import cfg
import numpy
import werkzeug
from PIL import Image, ImageDraw
from ssd.modeling.detector import build_detection_model
from ssd.utils.checkpoint import CheckPointer
import cv2
from torch.autograd import Variable
import numpy as np
from vizer.draw import draw_boxes

# Initialize our Flask application and the PyTorch model.
app = flask.Flask(__name__)
model = None
use_gpu = False


from ssd.data.datasets import COCODataset, VOCDataset
class_names = VOCDataset.class_names

def load_model():
    """Load the pre-trained model, you can use your model just as easily.
    """
    global model

    ckpt = 'model/vgg_ssd300_voc0712.pth'
    model = build_detection_model(cfg)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    model.eval()
    if use_gpu:
        model.cuda()


def transforms(image):
    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    # plt.imshow(x)
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
    return xx



def default(obj):
    if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
        numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
        numpy.uint16,numpy.uint32, numpy.uint64)):
        return int(obj)
    elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
        numpy.float64)):
        return float(obj)
    elif isinstance(obj, (numpy.ndarray,)): # add this line
        return obj.tolist() # add this line
    return json.JSONEncoder.default(obj)





REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = 'demo/'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif', 'tif', 'tiff'])

def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


@app.route('/', methods=['POST', 'GET'])

def index():
    return flask.render_template('index.html', has_result=False)

# fyk
def load_img(img_buffer):
    # image = caffe.io.load_image(string_buffer)
    pass
def disp_wait_msg(imagesrc):
    flask.render_template(
        'index.html', has_result=True,
        result=(False, '处理图片中...'),
        imagesrc=imagesrc
    )

def draw_rectangles(image_pil,det_result):
    # draw rectangles
    draw = ImageDraw.Draw(image_pil)
    for idx, result in enumerate(det_result):
        xmin, ymin,xmax ,ymax = result['location']
        draw.rectangle((xmin,ymin,xmax,ymax),  width=2, outline='yellow')
        draw.text((xmax + 5, ymax + 5), result['predicted_class']+ str(idx), 'yellow')
    del draw

@app.route('/detection_url', methods=['GET'])
def detection_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        # download
        raw_data = urllib.request.urlopen(imageurl).read()
        string_buffer = BytesIO(raw_data)
        # image = load_img(string_buffer)
        image_pil = Image.open(string_buffer)
        filename = os.path.join(UPLOAD_FOLDER, 'tmp.jpg')
        with open(filename,'wb') as f:
            f.write(raw_data)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        print(err)
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    # img_base64 = embed_image_html(filename)
    # disp_wait_msg(img_base64)
    results = prediction(filename)
    draw_rectangles(image_pil, results[1:-1])
    new_img_base64 = embed_image_html(image_pil)
    return flask.render_template(
        'index.html', has_result=True, result=results, imagesrc=new_img_base64)



@app.route('/detection_upload', methods=['POST'])
def detection_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
                    werkzeug.utils.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image_pil = exifutil.open_oriented_pil(filename)

    except Exception as err:
        print(err)
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    results = prediction(filename)


    draw_rectangles(image_pil, results[1:-1])
    new_img_base64 = embed_image_html(image_pil)

    return flask.render_template(
        'index.html', has_result=True, result=results,
        imagesrc=new_img_base64
    )



def prediction(filename):
    data = [0]
    start = time.time()
    print("filename:",filename)


    threshold = 0.6 #flask.request.files["threshold"].read()


    image = np.array(Image.open(filename).convert("RGB"))
    height, width = image.shape[:2]
    images = transforms(image)[0].unsqueeze(0)
    load_time = time.time() - start

    start = time.time()
    result = model(images)[0]
    inference_time = time.time() - start

    result = result.resize((width, height)).to(torch.device("cpu")).detach().numpy()
    boxes, labels, scores = result['boxes'], result['labels'], result['scores']

    indices = scores > threshold
    boxes = boxes[indices]
    labels = labels[indices]
    scores = scores[indices]


    if len(labels) != 0:
        data[0] = 1

    for i in range(len(labels)):
        item = {
            'predicted_class': class_names[labels[i]],
            'location': default(boxes[i]),
            'score': str(round(scores[i],2))
        }
        data.append(item)



    T = {
        'load_time': round(load_time * 1000,3),
        'inference': round(inference_time * 1000,3),
        'FPS': round(1.0 / inference_time,3)

    }
    data.append(T)
    print(data)

    return data

def embed_image_html(image_pil):
    """Creates an image embedded in HTML base64 format."""
    size = (512, 512) # (256, 256)
    resized = image_pil.resize(size)
    string_buf = BytesIO()
    resized.save(string_buf, format='png')
    data = string_buf.getvalue()
    data = base64.b64encode(data).decode().replace('\n', '')

    return 'data:image/png;base64,' + data




'''
terminal serve
'''

@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            # Read the image in PIL format
            start = time.time()
            image = flask.request.files["image"].read()

            image = np.array(Image.open(io.BytesIO(image)).convert("RGB"))

            threshold = flask.request.files["threshold"].read()

            image_name = flask.request.files["image_name"].read()

            # image = np.array(Image.open(image_path).convert("RGB"))
            height, width = image.shape[:2]
            images = transforms(image)[0].unsqueeze(0)
            load_time = time.time() - start



            start = time.time()
            result = model(images)[0]

            inference_time = time.time() - start
            # print(result)

            result = result.resize((width, height)).to(torch.device("cpu")).detach().numpy()
            boxes, labels, scores = result['boxes'], result['labels'], result['scores']

            indices = scores > float(threshold.decode())
            boxes = boxes[indices]
            labels = labels[indices]
            scores = scores[indices]


            drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
            Image.fromarray(drawn_image).save(os.path.join('demo/result', image_name.decode()))


            data['location'] = default(boxes)
            data['label'] = default(labels)
            data['score'] = default(scores)
            data['load_time'] = load_time * 1000
            data['inference'] = round(inference_time * 1000)
            data['FPS'] = round(1.0 / inference_time)

            # Indicate that the request was a success.
            data["success"] = True
            print(data)

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)




if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")


    load_model()
    app.run(host='0.0.0.0', port=5000)

