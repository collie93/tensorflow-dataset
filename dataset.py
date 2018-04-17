import os
import numpy as np
import skimage.io

class MakeDataset(object):

    def __init__(self, root='./dataset', num_train=None):
        self.root = root
        self.label_names = os.listdir(root)
        self.num_train   = num_train

        self.images = []
        self.labels = []

    def __call__(self):
        return self.label_names
        
    def LoadingData(self):
        # get image and label
        for subdir in os.listdir(self.root):
            label = self._convertOneHot(subdir)
            path  = os.path.join(self.root, subdir)
            for fname in os.listdir(path):
                image = _imread(path, fname)
                self.images.append(image)
                self.labels.append(label)

        # list -> numpy.ndarray
        self.images = np.array(self.images, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int32)

        # shuffle
        randomize = np.arange(len(self.images))
        np.random.shuffle(randomize)
        self.images = self.images[randomize]
        self.labels = self.labels[randomize]

        # visualization
        print('# ==========================================')
        print('# debug :                                   ')
        print('#    train :                                ')
        print('#        images.shape : {}'.format(self.images[:self.num_train].shape))
        print('#        labels.shape : {}'.format(self.labels[:self.num_train].shape))
        print('#    valid :                                ')
        print('#        images.shape : {}'.format(self.images[self.num_train:].shape))
        print('#        labels.shape : {}'.format(self.labels[self.num_train:].shape))
        print('# Num all data   : {}'.format(len(self.labels)))
        print('# Num train data : {}'.format(self.num_train))
        print('# Num valid data : {}'.format(len(self.images) - self.num_train))
        print('# ==========================================')
        
        return self.images[:self.num_train], self.labels[:self.num_train], self.images[self.num_train:], self.labels[self.num_train:]

    def _convertOneHot(self, name):
        onehot = np.zeros(len(self.label_names), dtype=np.int32)
        onehot[self.label_names.index(name)] = 1

        return onehot

def _imread(path, img):
    isRead = os.path.join(path, img)
    return skimage.io.imread(isRead) / 255.0

def main():
    md = MakeDataset(num_train=500)
    print(md())
    train_img, train_label, test_img, test_label = md.LoadingData()

if __name__ == '__main__':
    main()