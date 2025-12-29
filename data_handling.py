import os
import json
import cv2
import numpy as np

folder_names = ['train', 'valid', 'test']

image_filenames = {}
label_filenames = {}

print('Collecting file names...')
for name in folder_names:
    print(f'\t{name}')
    image_filenames[name] = os.listdir('dataset//' + name + '//images')
    label_filenames[name] = os.listdir('dataset//' + name + '//labels')
    print(f'\t\t# image names: {len(image_filenames[name])}')
    print(f'\t\t# label names: {len(label_filenames[name])}\n')

def get_labels(name, id):

    file_path = 'datset//' + name + '//labels//' + id + '.txt'
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

def get_image(name, id):
    file_path = 'dataset//' + name + '//images//' + id + '.jpg'
    return cv2.imread(file_path)

