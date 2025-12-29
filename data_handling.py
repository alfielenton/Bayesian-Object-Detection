import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

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

_, axs = plt.subplots(3, 1, figsize=(10, 10))

for i, name in enumerate(folder_names):
    im_dims = []
    animal_classes = []

    print(f'\tCollecting {name} measurements...')
    for j, id in enumerate(filenames[name]):
        
        im_dims.append(get_image(id).shape)
        animal_classes.append(int(get_labels(id)[0]))
        if (j + 1) % 100 == 0:
            print(f'\t\t{j + 1}/{len(filenames[name])} measured')

    animal_classes = np.array(animal_classes)
    im_dims = np.array(im_dims).mean(axis=0).tolist()

    measurement_file = f"dataset//{name}//measurements.txt"
    with open(measurement_file, "w") as f:
        f.write(f"Average Image Dimension: {im_dims}\n\n")

        unique, count = np.unique(animal_classes, return_counts=True)
        f.write("Class counts:\n")
        for u, c in zip(unique, count):
            f.write(f"{u} : {c}\n")

    axs[i].bar(unique, count)
    axs[i].set_xlabel('Category')
    axs[i].set_ylabel('Number')
    axs[i].set_title(f'Counts in {name} data')

    print("\n\tWritten results to " + measurement_file)
    print('\n')

plt.savefig('dataset//category-counts.png')
print('Saved Bar Graph in dataset folder')
plt.close()