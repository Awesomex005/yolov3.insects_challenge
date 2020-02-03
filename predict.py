# -*- coding: utf-8 -*-

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable

from reader import single_image_data_loader
from multinms import multiclass_nms
from yolov3 import YOLOv3
from draw_results import draw_results

import argparse

def parse_args():
    parser = argparse.ArgumentParser("Evaluation Parameters")
    parser.add_argument(
        '--image_name',
        type=str,
        default='./insects/test/images/2599.jpeg',
        help='the directory of test images')
    parser.add_argument(
        '--weight_file',
        type=str,
        default='./yolo_epoch50',
        help='the path of model parameters')
    args = parser.parse_args()
    return args

args = parse_args()
IMAGE_NAME = args.image_name
WEIGHT_FILE = args.weight_file


ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

VALID_THRESH = 0.01

NMS_TOPK = 400
NMS_POSK = 100
NMS_THRESH = 0.45

NUM_CLASSES = 7


if __name__ == '__main__':
    with fluid.dygraph.guard():
        model = YOLOv3('yolov3', num_classes=NUM_CLASSES, is_train=False)
        model_state_dict, _ = fluid.load_dygraph(WEIGHT_FILE)
        model.load_dict(model_state_dict)
        model.eval()

        total_results = []
        test_loader = single_image_data_loader(IMAGE_NAME, mode='test')
        for i, data in enumerate(test_loader()):
            img_name, img_data, img_scale_data = data
            img = to_variable(img_data)
            img_scale = to_variable(img_scale_data)

            outputs = model.forward(img)
            bboxes, scores = model.get_pred(outputs,
                                     im_shape=img_scale,
                                     anchors=ANCHORS,
                                     anchor_masks=ANCHOR_MASKS,
                                     valid_thresh = VALID_THRESH)

            bboxes_data = bboxes.numpy()
            scores_data = scores.numpy()
            results = multiclass_nms(bboxes_data, scores_data,
                          score_thresh=VALID_THRESH, 
                          nms_thresh=NMS_THRESH, 
                          pre_nms_topk=NMS_TOPK, 
                          pos_nms_topk=NMS_POSK)

result = results[0]
draw_results(result, IMAGE_NAME, draw_thresh=0.5)
