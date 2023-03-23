import os
import time
import torch
import random
import argparse
import matplotlib
import numpy as np

from torch import optim
from Models.PMTSPN import PMTSPN
from dataer import data_loader
from trainer import train
from torchvision import transforms


matplotlib.use('Agg')

# seeds
seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, default=0, metavar='cuda', help='cuda id(default: 0)')
parser.add_argument('-e', '--epochs', type=int, default=30, metavar='epochs', help='epoches (default: 40)')
parser.add_argument('-b', '--batchsize', type=int, default=16, metavar='batch size', help='batch size(default: 16)')
parser.add_argument('-d', '--dataset', type=str, default='pertex', metavar='data set', help='data set(default: pertex)')
parser.add_argument('-fp', '--filepath', type=str, default='test', metavar='filepath', help='filepath(default: test)')
parser.add_argument('-nn', '--modelname', type=str, default='pmtspn', metavar='models', help='models (default: resnet)')
parser.add_argument('-lr', '--learningrate', type=float, default=0.002, metavar='lr', help='learning rate (default: 2e-3)')
args = parser.parse_args()

if __name__ == '__main__':
    # device
    device = (f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # data set
    transforms = transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    train_loader, valid_loader = data_loader(dataset_name=args.dataset,
                                             batch_size=args.batchsize,
                                             transforms=transforms)
    # model                                        
    model = PMTSPN(device)
    model.to(device)

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learningrate, momentum=0.9)

    # training
    train(device=device,
          train_loader=train_loader,
          valid_loader=valid_loader,
          optimizer=optimizer,
          model=model,
          epochs=args.epochs,
          filepath=args.filepath,
          modelname=args.modelname)
