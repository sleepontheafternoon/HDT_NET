import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)





class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        if random.random() < 0.5:
            image = np.flip(image, 0)

        if random.random() < 0.5:
            image = np.flip(image, 1)

        if random.random() < 0.5:
            image = np.flip(image, 2)
        return {'image': image}


class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']

        H = random.randint(0, 240 - 128)
        W = random.randint(0, 240 - 128)
        D = random.randint(0, 160 - 128)

        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        # image = image[61: 61 + 128, 61: 61 + 128, 11: 11 + 128, ...]


        return {'image': image}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']


        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, }


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)

        return {'image': image}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']

        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')

        return {'image': image}
    #(240,240,155)>(240,240,160)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))


        image = torch.from_numpy(image).float()

        return {'image': image}




def transform(sample):
    trans = transforms.Compose([
        Pad(),
        #Random_rotate(),  # time-consuming
        Random_Crop(),
        Random_Flip(),
        Random_intencity_shift(),
        # Augmentation(),
        ToTensor()
    ])
    return trans(sample)



def transform_valid(sample):
    trans = transforms.Compose([
        Pad(),
        # MaxMinNormalization(),
        Random_Crop(),
        ToTensor()
    ])

    return trans(sample)

def transform_test(sample):
    trans = transforms.Compose([
        Pad(),
        # MaxMinNormalization(),
        Random_Crop(),
        ToTensor()
    ])

    return trans(sample)

class HDT(Dataset):
    def __init__(self, list_file, root='', mode='train',csv_file='addition_data.csv'):
        self.lines = []
        paths, names, idhs = [], [],[]

        with open(list_file) as f:
            for line in f:
                line = line.strip()
                row = line.split('/')[-1].split(',')
                names.append(row[0])
                idhs.append(int(row[1]))
                path = os.path.join(root, row[0], row[0] + '_')
                paths.append(path)
                self.lines.append(line)
        self.mode = mode
        self.names = names
        self.paths = paths
        self.idhs = idhs


    def __getitem__(self, item):
        path = self.paths[item]
        name = self.names[item]
        idh = self.idhs[item]

        data = pkload(path + 'data_f32b0.pkl')

        if self.mode == 'train':

            image, label,grade = data[0], data[1], data[2]
            sample = {'image': image}
            sample = transform(sample)
            idh_target = grade[1]

            return sample['image'],torch.tensor(idh)

        elif self.mode == 'valid':

            image, label,grade = data[0], data[1], data[2]
            sample = {'image': image}
            sample = transform_valid(sample)

            return sample['image'],torch.tensor(idh)
        else:
            image, label = data[0], data[1]
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')


            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
            image = torch.from_numpy(image).float()
            return image,label[1]  # 本来有个label[0]


    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

