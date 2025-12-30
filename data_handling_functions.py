import random
import json
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
    return values

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
    
    plt.title('Animal: ' + categ)
    plt.show()