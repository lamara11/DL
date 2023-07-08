import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.cur_index = 0
        self.cur_epoch_num = 0
        self.images = []
        self.labels = []
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.load()

    def load(self):
        with open(self.label_path, 'r') as label_file:
            labels = json.load(label_file)

        for filename in os.listdir(self.file_path):
            image_path = os.path.join(self.file_path, filename)
            image = np.load(image_path)
            label = labels.get(filename.split('.')[0], -1)
            self.images.append(image)
            self.labels.append(label)

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def next(self):
        start_index = self.cur_index
        end_index = self.cur_index + self.batch_size
        if self.shuffle:
            indexs = np.random.permutation(len(self.images))
            self.images = self.images[indexs]
            self.labels = self.labels[indexs]

        if end_index > len(self.images):
            rest = end_index - len(self.images)
            images_batch = np.concatenate((self.images[start_index:], self.images[:rest]))
            labels_batch = np.concatenate((self.labels[start_index:], self.labels[:rest]))
            self.cur_index = rest
            self.cur_epoch_num += 1
        else:
            images_batch = self.images[start_index:end_index]
            labels_batch = self.labels[start_index:end_index]
            self.cur_index = end_index

        batched_images = []
        for image in images_batch:
            if self.mirroring:
                if np.random.random() < 1:
                    image = np.flip(image, axis=1)
            if self.rotation:
                num_rotations = np.random.randint(4)
                image = np.rot90(image, num_rotations)

            image = resize(image, self.image_size)
            batched_images.append(image)

        batched_images = np.array(batched_images)
        return batched_images, labels_batch

    def current_epoch(self):
        return self.cur_epoch_num

    def class_name(self, label):
        return self.class_dict[label]

    def show(self):
        images, labels = self.next()
        columns = 3
        rows=len(images)//3
        fig=plt.figure()
        for i in range(1, columns*rows+1):
            fig.add_subplot(rows,columns,i,frameon=False).set_title(self.class_name(labels[i-1]))
            plt.axis('off')
            plt.imshow(images[i-1])
        plt.show()

