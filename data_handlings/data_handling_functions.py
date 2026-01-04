import torch
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt

with open('dataset//ids.json', 'r') as f:
    filenames = json.load(f)

with open('dataset//categories.json', 'r') as f:
    categories = json.load(f)['categories']

def get_image(id):

    for name in filenames.keys():
        if id in filenames[name]:
            break

    file_path = 'dataset//' + name + '//images//' + id + '.jpg'
    return cv2.imread(file_path)

def get_labels(id):

    for name in filenames.keys():
        if id in filenames[name]:
            break
    
    file_path = 'dataset//' + name + '//labels//' + id + '.txt'
    with open(file_path, 'r') as f:
        line = f.readline()[:-1]

    line += ' '
    values = []
    i_start = 0
    for i, n in enumerate(line):
        if n == ' ':
            values.append(line[i_start:i])
            i_start = i

    values = [float(v) if i > 0 else int(v) for i, v in enumerate(values) ]
    return values[0], values[1:]

def generate_batch(ids, resize = (800, 800)):

    x_batch = []
    y_cl_batch = []
    y_re_batch = []

    for id in ids:
        image = cv2.resize(get_image(id), resize) / 255.
        image = image.transpose(2, 0, 1)
        cl, re = get_labels(id)

        x_batch.append(image)
        y_cl_batch.append(cl)
        y_re_batch.append(re)

    x_batch = torch.tensor(np.array(x_batch), dtype=torch.float32)
    y_cl_batch = torch.tensor(np.array(y_cl_batch), dtype=torch.long)
    y_re_batch = torch.tensor(np.array(y_re_batch), dtype=torch.float32)

    return x_batch, y_cl_batch, y_re_batch

def plot_bounding_box(id, resize = None):

    image = get_image(id)
    label = get_labels(id)

    if resize is not None:
        image = cv2.resize(image, resize)

    im_h, im_w, _ = image.shape
    categ = categories[label[0]]
    bb = label[1:]

    centre_y, centre_x = im_h * bb[1], im_w * bb[0]
    box_y, box_x = im_h * bb[3], im_w * bb[2]

    plt.imshow(image)
    plt.scatter(centre_x, centre_y, label = 'Centre point', marker='x', color='red')
    plt.vlines([centre_x + .5 * box_x, centre_x - .5 * box_x],
               [centre_y - .5 * box_y] * 2, 
               [centre_y + .5 * box_y] * 2, colors='red')
    plt.hlines([centre_y + .5 * box_y, centre_y - .5 * box_y],
               [centre_x - .5 * box_x] * 2, 
               [centre_x + .5 * box_x] * 2, color='red')
    
    plt.title('Animal: ' + categ if resize is None else 'Animal: ' + categ +'. Resized')
    plt.show()

