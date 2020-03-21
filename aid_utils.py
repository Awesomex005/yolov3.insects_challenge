import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_loss_json(path2json):
    '''
    data = {
        'x_range' : [5, 45, 5],
        'losses' : [0.11, 0.22, 0.33, 0.9, 1., 2., 3., 0.88],
    }
    '''
    with open(path2json, 'r') as f:
        data = json.load(f)
        fig = plt.figure(figsize=(8,6))
        x = data['x_range']
        x = np.arange(x[0], x[1], x[2])
        y = data['losses']
        plt.plot(x, y);
        plt.ylim(0, np.mean(y)+1.0*np.std(y, ddof=1))
        fn = path2json.split("/")[-1].split('.')[0]
        plt.savefig("%s.png" %fn)
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        type=str,
        default='./data.json',
        metavar='path2losses',
        help='The losses data json file')
    args = parser.parse_args()
    return args

args = parse_args()

if __name__ == "__main__":

    plot_loss_json(args.p)
