import keras
# import keras_retinanet
import tensorflow as tf
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
# import miscellaneous modules
import cv2
from io import BytesIO
import pybase64
import argparse
import sys
import os
import numpy as np
import time
import json
from flask import Flask, jsonify, request, abort

import time
import classify
import tflite_runtime.interpreter as tflite
import platform
from PIL import Image

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({'status': "server is running"}), 200


@app.route('/image', methods=['POST'])
def predict_image():
    if not request.json or not 'data' in request.json:
        abort(400)
    
    caption = run_detection_image(model, labels_to_names, request.json['data'])
    return caption, 200

def preprocess_image_tpu(image):
    image = preprocess_image(image)
    image, _ = resize_image(image, min_side=1500, max_side=2000)

def predict_image_tpu(image):
    labels = {0: 'background', 1: 'pedestrian'}
    #interpreter = make_interpreter(os.path.join('snapshots', 'output_tflite_graph_edgetpu.tflite'))
    #interpreter.allocate_tensors()
    classify.set_input(interpreter, image)
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    classes = classify.get_output(interpreter, 1, 0.3)
    print('%.1fms' % (inference_time * 1000))
    for klass in classes:
        print('%s: %.5f' % (labels.get(klass.id, klass.id), klass.score))
        return labels.get(klass.id, klass.id), klass.score

def get_crops(img):
    height, width, _ = img.shape
    crop_size=224
    crop_list=[]
    cnt=0
    for yi in range(width // 224):
        for xi in range(height // 224):
            cnt+=1
            crop = img[crop_size*yi:crop_size*(yi+1),crop_size*xi:(crop_size)*(xi+1)]
            if crop.shape[0]==crop_size and crop.shape[1]==crop_size:
                crop_list.append(crop)
    return crop_list

def run_detection_image(model, labels_to_names, data):
    print("start predict...")
    start_time = time.time()
    with sess.as_default():
        with graph.as_default():
            imgdata = pybase64.b64decode(data)
            file_bytes = np.asarray(bytearray(imgdata), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            start = time.perf_counter()
            image_tpu, _ = resize_image(image, min_side=800, max_side=1333)
            crop_list = get_crops(image_tpu)
            for crop in crop_list:
                l_id, score = predict_image_tpu(crop)
                if l_id == 'pedestrian':
                    print('condidate is found: {}'.format(score))
                    break
            inference_time = time.perf_counter() - start
            print('total tpu time - %.1fms' % (inference_time * 1000))

            # preprocess image for network
            image = preprocess_image(image)
            image, scale = resize_image(image)

            # process image
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

            # correct for image scale
            boxes /= scale

            objects = []
            reaponse = {
              'objects': objects
            }

            # visualize detections
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.5:
                    break
                b = np.array(box.astype(int)).astype(int)
                # x1 y1 x2 y2
                obj = {
                  'name': labels_to_names[label],
                  'score': str(score),
                  'xmin': str(b[0]),
                  'ymin': str(b[1]),
                  'xmax': str(b[2]),
                  'ymax': str(b[3])
                }
                objects.append(obj)
            reaponse_json = json.dumps(reaponse)
            print("done in {} s".format(time.time() - start_time))
            return reaponse_json

def load_model(args):
    global sess 
    global model
    global labels_to_names
    global graph
    global interpreter

    sess = tf.InteractiveSession()
    graph = tf.get_default_graph()
    setup_gpu(0)
    model_path = args.model
    model = models.load_model(model_path, backbone_name='resnet50')
    labels_to_names = {0: 'Pedestrian'}

    interpreter = make_interpreter(os.path.join('snapshots', 'output_tflite_graph_edgetpu.tflite'))
    interpreter.allocate_tensors()
    return model, labels_to_names, interpreter

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    parser.add_argument('--model', help='Path to RetinaNet model.', default=os.path.join('snapshots', 'resnet50_liza_alert_v1_interface.h5'))
    parser.add_argument('--gpu', help='Visile gpu device. Set to -1 if CPU', default=0)
    return parser.parse_args(args)

def main(args=None):
    args = parse_args(args)
    load_model(args)
    print('model loaded')
    app.run(debug=False, host='0.0.0.0', port=5000)    

if __name__ == '__main__':
    main()
