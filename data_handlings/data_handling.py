import os
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt
from data_handling_functions import get_image, get_labels

categories = []
start_noting_categories = False
with open("dataset//data.yaml", 'r') as f:

    for line in f.readlines():
        if line.startswith('name'):
            start_noting_categories = True
            continue

        if start_noting_categories:
            ind = line.find(":")
            categories.append(line[ind+2:-1])
categories[-1] += 'a'
print('Obtained categories\n')

with open('dataset//categories.json', 'w') as f:
    json.dump({'categories':categories}, f)

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

with open('dataset//ids.json', 'w') as f:
    json.dump(filenames, f)


print('\nCollecting data measurements...\n')

_, axs = plt.subplots(3, 1, figsize=(20, 10))

for i, name in enumerate(folder_names):
    im_dims = []
    animal_classes = []

    print(f'\tCollecting {name} measurements...')
    for j, id in enumerate(filenames[name]):
        
        im_dims.append(get_image(id).shape)
        animal_classes.append(int(get_labels(id)[0]))
        if (j + 1) % 100 == 0 or (j + 1) == len(filenames[name]):
            print(f'\t\t{j + 1}/{len(filenames[name])} measured')

    animal_classes = np.array(animal_classes)
    im_dims = np.array(im_dims).mean(axis=0).tolist()

    measurement_file = f"dataset//{name}//measurements.json"
    data = {'avg_im_dim':im_dims, 'category_count':{}}

    unique, count = np.unique(animal_classes, return_counts=True)
    for u, c in zip(unique, count):
            data['category_count'][categories[u]] = int(c)

    with open(measurement_file, "w") as f:
        json.dump(data, f)

    axs[i].bar(unique, count)
    axs[i].set_xlabel('Category')
    axs[i].set_xticks(unique)
    axs[i].set_xticklabels(categories, rotation=45, fontsize=7)
    axs[i].set_ylabel('Number')
    axs[i].set_title(f'Counts in {name} data')

    print("\n\tWritten results to " + measurement_file)

plt.tight_layout()
plt.savefig('dataset//category-counts.png')
print('Saved Bar Graph in dataset folder')
plt.close()