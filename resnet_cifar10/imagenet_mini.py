import os
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from resnet_cifar10.dataset_maker import DatasetMaker, get_class_i

# https://patrykchrabaszcz.github.io/Imagenet32/

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_databatch(data_file, img_size=32):

    d = unpickle(data_file)
    x = d['data']
    y = d['labels']

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    img_size2 = img_size * img_size
    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))

    return [x[i,:, :, :] for i in range(x.shape[0])], y

def get_imagenet_trainloader(classes):


    train_data = [[] for i in classes]
    for idx in range(1, 11):
        print(idx)
        data_file = "../../../../mnt/datasets/TinyImageNet/Imagenet32_train/train_data_batch_" + str(idx)
        x, y = load_databatch(data_file)
        for i in range(len(classes)):
            train_data[i].extend(get_class_i(x, y, classes[i]))

    print([len(d) for d in train_data])
    x = np.concatenate([np.concatenate(i) for i in train_data])

    # x ranges from 0 to 255. compute mean+std, then normalize
    # x gets normalized later when we convert to an image.
    mean = np.mean(x, (0, 1))/np.float32(255)
    std = np.std(x, (0, 1))/np.float32(225)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = DatasetMaker(train_data, transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    return trainloader, transform

def get_imagenet_testloader(classes, transform, batch_size=100):

    test_data = []

    print("valid")
    data_file = "../../../../mnt/datasets/TinyImageNet/Imagenet32_val/val_data"
    x, y, = load_databatch(data_file)
    for i in range(len(classes)):
        test_data.append(get_class_i(x, y, classes[i]))

    print([len(d) for d in test_data])

    testset = DatasetMaker(test_data, transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return testloader