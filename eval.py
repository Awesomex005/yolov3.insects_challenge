# -*- coding: utf-8 -*-

import numpy as np
import json

import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable

from reader import test_data_loader
from multinms import multiclass_nms
from yolov3 import YOLOv3

import argparse

def parse_args():
    parser = argparse.ArgumentParser("Evaluation Parameters")
    parser.add_argument(
        '--image_dir',
        type=str,
        default='./insects/test/images',
        help='the directory of test images')
    parser.add_argument(
        '--weight_file',
        type=str,
        default='./yolo_epoch50',
        help='the path of model parameters')
    args = parser.parse_args()
    return args


args = parse_args()
TESTDIR = args.image_dir
WEIGHT_FILE = args.weight_file

ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

VALID_THRESH = 0.01

NMS_TOPK = 400
NMS_POSK = 100
NMS_THRESH = 0.45

NUM_CLASSES = 7

#TESTDIR = './insects/test/images' #请将此目录修改成用户自己保存测试图片的路径
#WEIGHT_FILE = './yolo_epoch50' # 请将此文件名修改成用户自己训练好的权重参数存放路径


if __name__ == '__main__':
    with fluid.dygraph.guard():
        model = YOLOv3('yolov3', num_classes=NUM_CLASSES, is_train=False)
        params_file_path = WEIGHT_FILE
        model_state_dict, _ = fluid.load_dygraph(params_file_path)
        model.load_dict(model_state_dict)
        model.eval()

        total_results = []
        test_loader = test_data_loader(TESTDIR, batch_size= 1, mode='test')
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
            result = multiclass_nms(bboxes_data, scores_data,
                          score_thresh=VALID_THRESH, 
                          nms_thresh=NMS_THRESH, 
                          pre_nms_topk=NMS_TOPK, 
                          pos_nms_topk=NMS_POSK)
            for j in range(len(result)):
                result_j = result[j]
                img_name_j = img_name[j]
                total_results.append([img_name_j, result_j.tolist()])
            print('processed {} pictures'.format(len(total_results)))
        print('processed finished, total {} pictures'.format(len(total_results)))
        json.dump(total_results, open('pred_results.json', 'w'))

