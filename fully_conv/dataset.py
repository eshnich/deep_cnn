from PIL import Image
from torchvision import datasets, transforms
import torch
import os
import torchvision.transforms as transforms


def make_dataset(size, num_classes):
    
    #files = os.listdir('./../impainting/data')
    #files = ['./../impainting/data/' + fname for fname in files]
    frames = []
    targets = []
    
    transform = transforms.Compose(
        [#transforms.Grayscale(),
         transforms.Resize((size, size)),
         transforms.ToTensor()])

    dataset = datasets.CIFAR10(root='./data', train=True, transform = transform, download=True)

    if num_classes == 2:
        classes = [3, 5] # 3 is cat, 5 is dog
    elif num_classes == 3:
        classes = [3, 5, 7] #3=cat, 5=dog, 7=horse
    elif num_classes == 5:
        classes = [2, 3, 4, 5, 7] # bird, cat, deer, dog, horse
    else:
        classes = range(0, num_classes)

    idx = []
    dataset_size=None
    batch_size=256
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in classes:
            idx.append(i)
        if dataset_size is not None:
            if len(idx)==dataset_size:
                break
    print(batch_size)
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, sampler = torch.utils.data.sampler.SubsetRandomSampler(idx))

    test_dataset = datasets.CIFAR10(root = '.data', train=False, transform=transform, download=True)
    idx = []
    for i in range(len(test_dataset.targets)):
        if test_dataset.targets[i] in classes:
            idx.append(i)
            #print(test_dataset.targets[i])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(idx), sampler=torch.utils.data.sampler.SubsetRandomSampler(idx))
    return train_loader, test_loader, classes
