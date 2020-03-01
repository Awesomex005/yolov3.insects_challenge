import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from insects_reader import get_insect_names, get_annotations
import argparse

def x_histograms(data, num_bins, name):
    fig = plt.figure(figsize=(8,6))
    n, bins, patches = plt.hist(data, num_bins, density=1, facecolor='blue', alpha=0.5)
    plt.title(name)
    plt.savefig("%s.png" %name)
    print("Generate image: %s.png" %name)
    plt.show()
    return

def data_summary(records):
    valid_images = len(records)
    total_objects = 0
    max_objs_per_img = 0
    min_objs_per_img = float('inf')
    avg_objs_per_img = 0
    for _, record in enumerate(records):
        objs = len(record["gt_class"])
        total_objects += objs
        max_objs_per_img = objs if objs > max_objs_per_img else max_objs_per_img
        min_objs_per_img = objs if objs < min_objs_per_img else min_objs_per_img
    avg_objs_per_img = total_objects*1.0/valid_images
    print("valid Images: {} total objects: {}".format(valid_images, total_objects))
    print("objs per image [max, avg, min]: {}".format([max_objs_per_img, avg_objs_per_img, min_objs_per_img]))
    return

def inspect_gtbboxes(records):
    gtbbox_area_scales = []
    for record in records:
        img_w = record["w"]
        img_h = record["h"]
        for gtbbox in record["gt_bbox"]:
            box_w = gtbbox[2]
            box_h = gtbbox[3]
            area_scale = (box_w*box_h)*1.0/(img_w*img_h)
            gtbbox_area_scales.append(area_scale)

    x_histograms(gtbbox_area_scales, 200, "compare_gtbbox_area_to_image_area")
    return

def parse_args():
    parser = argparse.ArgumentParser("Data summary")
    parser.add_argument(
        '--data_path',
        type=str,
        default='./insects/test/images/2599.jpeg',
        help='the directory of test images')
    args = parser.parse_args()
    return args

args = parse_args()

if __name__ == '__main__':
    records = get_annotations(get_insect_names(), args.data_path)
    data_summary(records)
    inspect_gtbboxes(records)

