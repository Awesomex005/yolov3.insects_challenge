from insects_reader import get_insect_names, get_annotations
import argparse

def data_summary(datadir):
    records = get_annotations(get_insect_names(), datadir)
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
    data_summary(args.data_path)