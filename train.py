# -*- coding: utf-8 -*-

import time
import os
import numpy as np
import json
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable

from reader import data_loader, test_data_loader, multithread_loader
from yolov3 import YOLOv3

# train.py
# 提升点： 可以改变anchor的大小，注意训练和测试时要使用同样的anchor
ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

IGNORE_THRESH = .7
NUM_CLASSES = 7

TRAINDIR = '../insects/train'
VALIDDIR = '../insects/val'

# train.py
if __name__ == '__main__':
    with fluid.dygraph.guard():
        model = YOLOv3('yolov3', num_classes = NUM_CLASSES, is_train=True)
        opt = fluid.optimizer.Momentum(
                     learning_rate=0.001,  #提升点：可以调整学习率，或者设置学习率衰减
                     momentum=0.9)   # 提升点： 可以添加正则化项

        train_loader = multithread_loader(TRAINDIR, batch_size= 10, mode='train')
        valid_loader = multithread_loader(VALIDDIR, batch_size= 10, mode='valid')

        losses = []
        MAX_EPOCH = 50  # 提升点： 可以改变训练的轮数
        MAX_EPOCH = 20
        for epoch in range(MAX_EPOCH):
            for i, data in enumerate(train_loader()):
                img, gt_boxes, gt_labels, img_scale = data
                gt_scores = np.ones(gt_labels.shape).astype('float32')
                gt_scores = to_variable(gt_scores)
                img = to_variable(img)
                gt_boxes = to_variable(gt_boxes)
                gt_labels = to_variable(gt_labels)
                outputs = model(img)
                loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                      anchors = ANCHORS,
                                      anchor_masks = ANCHOR_MASKS,
                                      ignore_thresh=IGNORE_THRESH,
                                      use_label_smooth=False)

                loss.backward()
                opt.minimize(loss)
                model.clear_gradients()
                if i % 5 == 0:
                    timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                    print('{}[TRAIN]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))
                    losses.append(loss.numpy().item())

            # save params of model
            if (epoch % 5 == 0) or (epoch == MAX_EPOCH -1):
                fluid.save_dygraph(model.state_dict(), 'yolo_epoch{}'.format(epoch))
                # 保存loss数据
                l_data = {
                    'x_range' : [5, 5*(len(losses)+1), 5],
                    'losses' : losses,
                }
                with open('losses.json', 'w') as f:
                    json.dump(l_data, f)
                
            # 每个epoch结束之后在验证集上进行测试
            model.eval()
            for i, data in enumerate(valid_loader()):
                img, gt_boxes, gt_labels, img_scale = data
                gt_scores = np.ones(gt_labels.shape).astype('float32')
                gt_scores = to_variable(gt_scores)
                img = to_variable(img)
                gt_boxes = to_variable(gt_boxes)
                gt_labels = to_variable(gt_labels)
                outputs = model(img)
                loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                      anchors = ANCHORS,
                                      anchor_masks = ANCHOR_MASKS,
                                      ignore_thresh=IGNORE_THRESH,
                                      use_label_smooth=False)
                if i % 1 == 0:
                    timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                    print('{}[VALID]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))
            model.train()


