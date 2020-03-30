# !/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: kristine
# mail:   kristine@mail.ustc.edu.cn
# Date:   2020/3/18


import requests
import argparse

from ssd.data.datasets import COCODataset, VOCDataset
class_names = VOCDataset.class_names

# Initialize the PyTorch REST API endpoint URL.
PyTorch_REST_API_URL = 'http://0.0.0.0:5000/predict'


def predict_result(image_file, threshold):
    # Initialize image path
    image = open(image_file, 'rb').read()
    # image_name = image[]
    payload = {'image': image, 'image_name': (image_file.split("/")[-1]).encode(),'threshold':str(threshold).encode()}

    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL, files=payload).json()

    # Ensure the request was successful.
    if r['success']:
        print('Hello')
        assert len(r['location']) == len(r['label']) == len(r['score']) ,'len(location) 不等于 len(label) 不等于 len(score)'

        meters = ' | '.join(
            [
                'load {:.2f}ms'.format(r['load_time']),
                'inference {:.2f}ms'.format(r['inference']),
                'FPS {:.2f}'.format(r['FPS']),
                '\n'
                ]
            )

        for i in range(len(r['location'])):
            meters = ' | '.join(
                [
                    meters,
                    'label {}'.format(class_names[r['label'][i]]),
                    'location {}'.format(r['location'][i]),
                    'score {:.2f}'.format(r['score'][i]),
                    '\n'
                ]
            )
        print(' result: {}'.format(meters))

    else:
        print('Request failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSD demo')
    parser.add_argument('--file', type=str, help='test image file')
    parser.add_argument("--score_threshold", type=float, default=0.7)

    args = parser.parse_args()
    predict_result(args.file, args.score_threshold)
