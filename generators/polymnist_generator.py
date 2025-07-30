import numpy as np
import os
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class PolyMNISTGenerator(Sequence):
    def __init__(self, data_dir, batch_size=64, num_classes=10, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle

        self.image_paths, self.labels = self._load_paths_and_labels()
        self.on_epoch_end()

    def _load_paths_and_labels(self):
        image_paths = []
        labels = []
        for subdir in os.listdir(self.data_dir):
            full_path = os.path.join(self.data_dir, subdir)
            for fname in os.listdir(full_path):
                if fname.endswith('.png'):
                    label = int(fname.split('.')[-1][0])  # e.g., 7 from "00001.7.png"
                    path = os.path.join(full_path, fname)
                    image_paths.append(path)
                    labels.append(label)
        return np.array(image_paths), np.array(labels)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            self.image_paths = self.image_paths[indices]
            self.labels = self.labels[indices]

    def __getitem__(self, index):
        batch_paths = self.image_paths[index*self.batch_size:(index+1)*self.batch_size]
        batch_labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]

        batch_images = [img_to_array(load_img(path, target_size=(28, 28))) / 255.0 for path in batch_paths]
        X = np.array(batch_images)
        y = to_categorical(batch_labels, num_classes=self.num_classes)

        return X, y

print("ya termino")