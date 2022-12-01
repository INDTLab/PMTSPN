import os
import re
import torch
import numpy as np
import pandas as pd

from PIL import Image


def data_loader(dataset_name, batch_size, transforms):
    if dataset_name == 'pertex':
        train_path = "/data1/pertex_train/"
        valid_path = "/data1/pertex_valid/"
        label_path = "/data1/ISOMap8 Similarity.csv"
        train_dataset = PertexSet(filepath=train_path,
                                  label_path=label_path,
                                  transform=transforms)
        valid_dataset = PertexSet(filepath=valid_path,
                                  label_path=label_path,
                                  transform=transforms)
    elif dataset_name == 'ptd':
        train_path = "/data1/PTD/image_train/"
        valid_path = "/data1/PTD/image_test/"
        label_path = "/data1/PTD/sim_gro_mer.csv"
        train_dataset = PTDSet(filepath=train_path,
                               label_path=label_path,
                               transform=transforms)
        valid_dataset = PTDSet(filepath=valid_path,
                               label_path=label_path,
                               transform=transforms)
    else:
        print('Dataset name not matched.')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=16)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=1)
    return train_loader, valid_loader


class PertexSet(torch.utils.data.Dataset):
    def __init__(self, filepath, label_path, transform):
        super(PertexSet).__init__()
        self.__load_data__(filepath, label_path, transform)

    def __getitem__(self, index):
        (x1, x2, y) = self.data[index]
        img1 = self.trans(Image.open(x1).convert('RGB'))
        img2 = self.trans(Image.open(x2).convert('RGB'))
        score = torch.tensor(y, dtype=torch.float32)
        return img1, img2, score

    def __len__(self):
        return len(self.data)

    def __load_data__(self, filepath, label_path, transform):
        files = os.listdir(filepath)
        self.scores = np.asarray(pd.read_csv(label_path))
        self.data = []
        for i in range(len(files)):
            for j in range(i, len(files)):
                m = int(re.findall(r"\d+", files[i])[0])
                n = int(re.findall(r"\d+", files[j])[0])
                self.data.append((os.path.join(filepath, files[i]),
                                  os.path.join(filepath, files[j]),
                                  self.scores[m - 1, n - 1]))
        self.trans = transform

class PTDSet(torch.utils.data.Dataset):
    def __init__(self, filepath, label_path, transform):
        super(PTDSet).__init__()
        self.__load_data__(filepath, label_path, transform)

    def __getitem__(self, index):
        (x1, x2, y) = self.data[index]
        img1 = self.trans(Image.open(x1).convert('RGB'))
        img2 = self.trans(Image.open(x2).convert('RGB'))
        score = torch.tensor(y, dtype=torch.float32)
        return img1, img2, score

    def __len__(self):
        return len(self.data)

    def __load_data__(self, filepath, label_path, transform):
        files = os.listdir(filepath)
        self.scores = np.asarray(pd.read_csv(label_path))
        self.data = []
        for i in range(len(files)):
            for j in range(i, len(files)):
                m = int(re.findall(r"\d+", files[i])[0])
                n = int(re.findall(r"\d+", files[j])[0])
                self.data.append((os.path.join(filepath, files[i]),
                                  os.path.join(filepath, files[j]),
                                  self.scores[m - 1, n - 1]))
        self.trans = transform