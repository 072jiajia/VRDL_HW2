import cv2
import os
import torch
import numpy as np
from torchvision.transforms import Normalize
from torch.utils.data import Dataset, DataLoader

'''Define Input Data Normalizer
Using the mean and std of Imagenet
'''
Normalizer = Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])


def get_data(Indices):
    ''' Get the training/testing data '''
    x = []
    y = []
    for idx in Indices:
        x.append('data/train/' + str(idx + 1) + '.png')
        y.append('data/train/' + str(idx + 1) + '.npy')

    return x, y


class TrainDataset(Dataset):
    ''' TrainDataset '''

    def __init__(self, Indices):
        x_train, y_train = get_data(Indices)
        self.x = x_train
        self.y = y_train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        # Define resized image size's shorter edge
        FullSize = np.random.randint(144, 176)
        # Crop size
        Size = 128

        annot = np.load(self.y[item])
        image = cv2.imread(self.x[item])

        # resize
        H, W, C = image.shape
        scale = FullSize / min(H, W)
        _H = int(H * scale)
        _W = int(W * scale)

        image = cv2.resize(image, (_W, _H))
        image = np.transpose(image, (2, 0, 1)) / 255.
        image = torch.tensor(image)

        # Normalize
        image = Normalizer(image)

        # Crop
        C, _H, _W = image.shape
        cropH = np.random.randint(_H - Size + 1)
        cropW = np.random.randint(_W - Size + 1)

        img = image[:, cropH: cropH + Size, cropW: cropW + Size]

        # resize and shift BBox
        annot[:, :4] = annot[:, :4] * scale
        annot[:, [0, 2]] = annot[:, [0, 2]] - cropW
        annot[:, [1, 3]] = annot[:, [1, 3]] - cropH
        annot = torch.tensor(annot)

        return {'img': img, 'annot': annot}


class ValDataset(Dataset):
    ''' Validation Dataset '''

    def __init__(self, Indices):
        x_test, y_test = get_data(Indices)
        self.x = x_test
        self.y = y_test

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        ''' Because the Validation data will not be changed
        Load it and save it in self.x and self.y
        '''
        if type(self.x[item]) is str:
            # Define resized image size's shorter edge
            FullSize = 160
            # load data
            image = cv2.imread(self.x[item])
            annot = np.load(self.y[item])

            # resize image
            H, W, C = image.shape
            if H > W:
                scale = FullSize/W
                _H = int(H * scale)
                _W = FullSize
                image = cv2.resize(image, (_W, _H))
                # pad zero because my model can only
                # input images with size of *16 * n, 16 * m)
                padding = 15 - (_H + 15) % 16
                image = cv2.copyMakeBorder(image, 0, padding, 0, 0,
                                           cv2.BORDER_CONSTANT, value=0)

            else:
                scale = FullSize / H
                _H = FullSize
                _W = int(W * scale)

                image = cv2.resize(image, (_W, _H))
                # pad zero because my model can only
                # input images with size of *16 * n, 16 * m)
                padding = 15 - (_W + 15) % 16
                image = cv2.copyMakeBorder(image, 0, 0, 0, padding,
                                           cv2.BORDER_CONSTANT, value=0)

            image = np.transpose(image, (2, 0, 1)) / 255.
            image = torch.tensor(image)
            self.x[item] = Normalizer(image)

            annot[:, :4] = annot[:, :4] * scale
            annot = torch.tensor(annot)

            self.y[item] = annot

        return {'img': self.x[item], 'annot': self.y[item]}


def collater(data):
    ''' collect data '''
    imgs = torch.stack([s['img'] for s in data])

    annots = [s['annot'] for s in data]
    max_num_annots = max(annot.shape[0] for annot in annots)

    # padding -1 to annotations to make each
    # annotations have the same size
    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        for idx, annot in enumerate(annots):
            annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    return {'img': imgs, 'annot': annot_padded}


class TestDataset(Dataset):
    def __init__(self):
        self.x = ['data/test/' + str(i + 1) + '.png' for i in range(13068)]
        self.scale = [None] * 13068

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        # Define resized image size's shorter edge
        FullSize = 160
        # load data
        image = cv2.imread(self.x[item])

        H, W, C = image.shape
        if H > W:
            scale = FullSize/W
            _H = int(H * scale)
            _W = FullSize
            image = cv2.resize(image, (_W, _H))
            # pad zero because my model can only
            # input images with size of *16 * n, 16 * m)
            padding = 15 - (_H + 15) % 16
            image = cv2.copyMakeBorder(image, 0, padding, 0, 0,
                                       cv2.BORDER_CONSTANT, value=0)

        else:
            scale = FullSize / H
            _H = FullSize
            _W = int(W * scale)

            image = cv2.resize(image, (_W, _H))
            # pad zero because my model can only
            # input images with size of *16 * n, 16 * m)
            padding = 15 - (_W + 15) % 16
            image = cv2.copyMakeBorder(image, 0, 0, 0, padding,
                                       cv2.BORDER_CONSTANT, value=0)

            image = np.transpose(image, (2, 0, 1)) / 255.
            image = torch.tensor(image)

        image = Normalizer(image)

        return {'img': image, 'scale': scale}
