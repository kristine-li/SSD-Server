# !/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: kristine
# mail:   kristine@mail.sim.ac.cn
# Date:   2020/3/1

import io
import os
import json

import time

import flask


import torch
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet50

from ssd.modeling.detector.ssd_detector import SSDDetector

from ssd.config import cfg
import numpy


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


# with open('imagenet_class.txt', 'r') as f:
#     idx2label = eval(f.read())
from ssd.data.datasets import COCODataset, VOCDataset
class_names = VOCDataset.class_names

def load_model():
    """Load the pre-trained model, you can use your model just as easily.
    """
    global model

    ckpt = '/Users/kristine/Downloads/SSD_new/vgg_ssd300_voc0712.pth'
    model = build_detection_model(cfg)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    # model = SSDDetector(cfg)
    # checkpoint = torch.load('/Users/kristine/Downloads/SSD_new/vgg_ssd300_voc0712.pth', map_location='cpu')
    # model.load_state_dict(checkpoint)
    #
    # model.load_state_dict(checkpoint.pop("model"))
    # if "optimizer" in checkpoint:
    #     logger.info("Loading optimizer from {}".format(f))
    #     self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
    # if "scheduler" in checkpoint and self.scheduler:
    #     self.logger.info("Loading scheduler from {}".format(f))
    #     self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
    #
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

            # image_name = '1.jpg'

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


            # for i in range(len(labels)):
            #     item = {
            #         'predicted_class': class_names[labels[i]],
            #         'box': default(boxes[i]),
            #         'score': str(scores[i])
            #     }
            #     data.append(item)
            #
            #
            # print(data)
            # #
            # data = flask.make_response(json.dumps(data))

            # rsp.headers['Connection'] = 'close'
            # return rsp





            # drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
            # Image.fromarray(drawn_image).save(os.path.join('demo/result', image_name))

            # # Preprocess the image and prepare it for classification.
            # image = prepare_image(image, target_size=(224, 224))
            #
            # # Classify the input image and then initialize the list of predictions to return to the client.
            # preds = F.softmax(model(image), dim=1)
            # results = torch.topk(preds.cpu().data, k=3, dim=1)
            #
            # data['location'] = list()
            # data['label'] = list()
            # data['score'] = list()
            #
            # # Loop over the results and add them to the list of returned predictions
            # for prob, label in zip(results[0][0], results[1][0]):
            #     label_name = idx2label[label]


            data['location'] = default(boxes)
            data['label'] = default(labels)
            data['score'] = default(scores)
            data['load_time'] = load_time * 1000
            data['inference'] = round(inference_time * 1000)
            data['FPS'] = round(1.0 / inference_time)

            # Indicate that the request was a success.
            data["success"] = True
            print(data)


    # print(type(result))

    # tmp = default(boxes)


    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)




if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")


    load_model()
    app.run()
