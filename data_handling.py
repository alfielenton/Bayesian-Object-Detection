import os
import cv2
import numpy as np

folder_names = ['train', 'valid', 'test']
filenames = {}

print('Collecting file names...\n')
for name in folder_names:

    image_filenames = [file_name[:-4] for file_name in os.listdir('dataset//' + name + '//images')]
    label_filenames = [file_name[:-4] for file_name in os.listdir('dataset//' + name + '//labels')]

    if set(image_filenames) != set(label_filenames):
        raise Exception(f"Images and Labels for {name} folder do not correlate")
    else:
        print(f'\tImages and Labels for {name} folder correlate')

    filenames[name] = image_filenames

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

    values = [float(v) for v in values]
    return values

def get_image(id):

    for name in filenames.keys():
        if id in filenames[name]:
            break

    file_path = 'dataset//' + name + '//images//' + id + '.jpg'
    return cv2.imread(file_path)

print('\nCollecting data measurements...\n')
image_dimensions = {}

for name in folder_names:
    im_dims = []
    animal_classes = []

    print(f'\tCollecting {name} measurements...')
    for i, id in enumerate(filenames[name]):
        
        im_dims.append(get_image(id).shape)
        animal_classes.append(int(get_labels(id)[0]))
        if (i + 1) % 100 == 0:
            print(f'\t\t{i+1}/{len(filenames[name])} measured')

    animal_classes = np.array(animal_classes)
    im_dims = np.array(im_dims).mean(axis=0).tolist()

    measurement_file = f"dataset//{name}//measurements.txt"
    with open(measurement_file, "w") as f:
        f.write(f"Average Image Dimension: {im_dims}\n\n")

        unique, counts = np.unique(animal_classes, return_counts=True)
        f.write("Class counts:\n")
        for u, c in zip(unique, counts):
            f.write(f"{u} : {c}\n")

    print("\n\tWritten results to " + measurement_file)
    print('\n')