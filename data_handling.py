import os
import json
import cv2
import numpy as np

def get_labels(file_path):

    with open(file_path, 'r') as f:
        line = f.readline()[:-1]

    line += ' '
    values = []
    i_start = 0
    for i, n in enumerate(line):
        if n == ' ':
            values.append(line[i_start:i])
            i_start = i

    values = [float(v) for v in values]
    return values

def get_image(file_path):
    return cv2.imread(file_path)


folder_names = ['train', 'valid', 'test']

image_filenames = {}
label_filenames = {}

for name in folder_names:
    image_filenames[name] = os.listdir('dataset//' + name + '//images')
    label_filenames[name] = os.listdir('dataset//' + name + '//labels')

images = {}
labels = {}

for name in folder_names:

    images[name] = {}
    for image_name in image_filenames[name]:
        im = get_image('dataset//' + name + '//images//' + image_name)
        images[name][image_name[:-4]] = (im.tolist(), im.shape)

    labels[name] = {}
    for label_name in label_filenames[name]:
        la = get_labels('dataset//' + name + '//labels//' + label_name)
        labels[name][label_name[:-4]] = la

with open('data-handling-results//mapping.json','w') as f:
    json.dump({'images':images, 'labels':labels}, f)