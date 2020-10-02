'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import copy
import os
import argparse

import resnet
import fcn
from dataset_maker import DatasetMaker, get_class_i
from imagenet_mini import get_imagenet_trainloader, get_imagenet_testloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight_decay')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr-decay', type=int, default=200, help='How often to decrease learning by gamma.')
parser.add_argument('--num-epoch', default=500, type=int, help='number of epoches')
parser.add_argument('--width', default=10, type=int, help='width')
parser.add_argument('--test-size', default=10000, type=int, help='number of test points')
parser.add_argument('--save-freq', default=100, type=int, metavar='N', help='save frequency')
parser.add_argument('--trials', default=1, type=int, help='number of trials')
parser.add_argument('--model', default='resnet', type=str)
args = parser.parse_args()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

'''
# train on the full set

trainset = torchvision.datasets.CIFAR10(
    root='../data', train=True, download=True, transform=transform_train)

# only consider half the training set (at random)
permute_index = np.split(np.random.permutation(len(trainset)), 2)
print(permute_index)
trainsubset = copy.deepcopy(trainset)
trainsubset.data = [trainsubset.data[index] for index in permute_index[0]]
trainsubset.targets = [trainsubset.targets[index] for index in permute_index[0]]

trainloader = torch.utils.data.DataLoader(
    trainsubset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

#classes = ('plane', 'car', 'bird', 'cat', 'deer',
#           'dog', 'frog', 'horse', 'ship', 'truck')
#NUM_CLASSES = len(classes)
NUM_CLASSES = 10
'''

'''
# train on a subset of the classes
trainset = torchvision.datasets.CIFAR100(root = '../data', train=True, download=True)
testset = torchvision.datasets.CIFAR100(root = '../data', train=False, download=True)

classes = range(10)
NUM_CLASSES = len(classes)

trainsubset = DatasetMaker([get_class_i(trainset.data, trainset.targets, i) for i in classes], transform_train)
testsubset = DatasetMaker([get_class_i(testset.data, testset.targets, i) for i in classes], transform_test)

trainloader = torch.utils.data.DataLoader(trainsubset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testsubset, batch_size=100, shuffle=False, num_workers=2)
'''


classes = range(2)
NUM_CLASSES=len(classes)
trainloader, transform = get_imagenet_trainloader(classes)
testloader = get_imagenet_testloader(classes, transform)

# Training
def train(net, trainloader):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #print(inputs)
        inputs, targets = inputs.cuda(), targets.cuda()
        targets_onehot = torch.FloatTensor(targets.size(0), NUM_CLASSES).cuda()
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets.view(-1, 1).long(), 1)
        optimizer.zero_grad()
        outputs = net(inputs)
        #print(outputs.size())
        #print(targets_onehot.size())
        loss = criterion(outputs, targets_onehot)
        #loss = criterion(outputs, targets) 
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss / total, 100. * correct / total


# Test
def test(net, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            targets_onehot = torch.FloatTensor(targets.size(0), NUM_CLASSES).cuda()
            targets_onehot.zero_()
            targets_onehot.scatter_(1, targets.view(-1, 1).long(), 1)
            outputs = net(inputs)
            loss = criterion(outputs, targets_onehot)
            #loss = criterion(outputs, targets)
            test_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return test_loss / total, 100. * correct / total

models = [
    [1, 1, 1, 1],
    [1, 1, 2, 1],
    [1, 2, 2, 1],
    [2, 2, 2, 1],
    [2, 2, 2, 2],
    [2, 2, 3, 2],
    [2, 3, 3, 2],
    #[3, 3, 3, 2],
    [3, 3, 3, 3],
    #[3, 3, 4, 3],
    [3, 4, 4, 3],
    #[3, 4, 5, 3],
    [3, 4, 6, 3],
    [3, 4, 8, 3],
    [3, 4, 10, 3],
    [3, 4, 12, 3],
    [3, 4, 14, 3],
]

models = [[1, 1, 1, 1]]

##################################################
# setup log file
##################################################
def init_logfile(filename, text):
    f = open(filename, 'w')
    f.write(text + "\n")
    f.close()


def log(filename, text):
    f = open(filename, 'a')
    f.write(text + "\n")
    f.close()

outdir = 'imagenet32_test'
if not os.path.exists(outdir):
    os.makedirs(outdir)
logfilename = os.path.join(outdir, 'log_width{}_{}classes.txt'.format(args.width, NUM_CLASSES))
init_logfile(logfilename, "depth\ttrain_loss\ttrain_acc\ttest_loss\ttest_acc")
for model in models:
    if args.model=='resnet':
        depth = 2*sum(model) + 2
        net = resnet.ResNet(resnet.BasicBlock, model, num_classes=NUM_CLASSES, width=args.width)
    elif args.model=='fcn':
        depth = 2
        net = fcn.Net(32, width=args.width)
    net.cuda()

    criterion = nn.MSELoss(reduction='mean').cuda()
    #criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=args.lr_decay, gamma=0.1)
    #init_test_loss, init_test_acc = test(net, testloader)
    #print(init_test_loss, init_test_acc)
    for epoch in range(1, args.num_epoch + 1):
        train_loss, train_acc = train(net, trainloader)
        test_loss, test_acc = test(net, testloader)

        print('depth: {}, epoch: {}, train_loss: {:.6f}, train acc: {}, test loss: {:.6f}, test acc: {}'.format(depth, epoch, train_loss, train_acc, test_loss, test_acc))
        scheduler.step(epoch)
    
        if epoch % args.save_freq == 0:
            torch.save(net.state_dict(), os.path.join(outdir, 'model_depth{}_width{}_classes{}_epoch{}.pkl'.format(depth, args.width, NUM_CLASSES, epoch)))

    # log data at end of training
    print('depth: {}, train_loss: {:.6f}, train acc: {}, test loss: {:.6f}, test acc: {}'.format(
        depth, train_loss, train_acc, test_loss, test_acc))
    log(logfilename, "{}\t{:.5}\t{:.5}\t{:.5}\t{:.5}".format(depth, train_loss, train_acc, test_loss, test_acc))
    torch.save(net.state_dict(), os.path.join(outdir, 'model_depth{}_width{}_classes{}.pkl'.format(depth, args.width, NUM_CLASSES)))
print('Program finished', flush=True)
