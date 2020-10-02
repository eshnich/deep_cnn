import fcn
import conv_net
import mat_fact_net
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.optim as optim
import numpy as np
from copy import deepcopy
import time

def update_stats(stats, key, train_loss, train_acc, test_loss, test_acc):
    stats[key]['train_loss'] = train_loss
    stats[key]['train_acc'] = train_acc
    stats[key]['test_loss'] = test_loss
    stats[key]['test_acc'] = test_acc

def train_net(train_loader, test_loader, depth, size, classes, width, model='conv', filters=3):
    start_time = time.time()
    
    # Use the following to instantiate a network
    if model == 'fcn':
        net = fcn.Net(depth, size, width, True, num_classes=len(classes))
    elif model == 'conv':
        net = conv_net.Net(depth, size, width, True, filters, num_classes = len(classes))
    elif model == 'matrix':
        net = mat_fact_net.MatFactNet(depth, size, True)
    else:
        raise Exception("Model not defined")
    print(model)

    # Use double precision
    #net.double()
    # Put the network on the GPU

    net.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=1e-4)
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum=0.9)	
    num_epochs = 2000
    terminate = False
    
    stats = {
        'best_train_acc': {'train_loss': np.float('inf'), 'train_acc': 0.0, 'test_loss': np.float('inf'), 'test_acc': 0.0},
        'best_test_acc': {'train_loss': np.float('inf'), 'train_acc': 0.0, 'test_loss': np.float('inf'), 'test_acc': 0.0},
        'best_train_loss': {'train_loss': np.float('inf'), 'train_acc': 0.0, 'test_loss': np.float('inf'), 'test_acc': 0.0},
    }
 
    class_map = {}
    for i in range(len(classes)):
        class_map[classes[i]] = i
    
    # prep test data
    for batch, (data, target) in enumerate(test_loader):
        test_data = data.cuda()
        test_target = (torch.sum(target.reshape(-1, 1).repeat(1, len(classes)) > torch.LongTensor(classes), 1)).long().cuda()
    print(test_target)
    print(test_data.type())
    print(test_target.type())

    for i in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        total = 0 
        for batch, (data, target) in enumerate(train_loader):
            data = data.cuda()
            target = (torch.sum(target.reshape(-1, 1).repeat(1, len(classes)) > torch.LongTensor(classes), 1)).long().cuda()
            # Take 1 step of GD
            train_loss, train_acc = train_step(net, data,
                                               target, optimizer, iteration=i)
            elements = target.size(0) 
            running_loss += train_loss*elements
            running_acc += train_acc*elements
            total += elements

        running_loss = running_loss/total
        running_acc = running_acc/total
        test_acc = test_accuracy(test_data, test_target, net) 
        test_loss = get_test_loss(test_data, test_target, net)

        # log best test accuracy
        if test_acc > stats['best_test_acc']['test_acc']:
            update_stats(stats, 'best_test_acc', running_loss, running_acc, test_loss, test_acc)
        
        # log best train loss
        if running_loss < stats['best_train_loss']['train_loss']:
            update_stats(stats, 'best_train_loss', running_loss, running_acc, test_loss, test_acc)

        # log best training accuracy
        if running_acc > stats['best_train_acc']['train_acc']:
            update_stats(stats, 'best_train_acc', running_loss, running_acc, test_loss, test_acc)
            if stats['best_train_acc']['train_acc'] == 1.0:
                # 100% training accuracy, done
                break
        
        if i%10 == 0:
            print(depth, i, running_loss, stats['best_train_acc']['train_loss'], "accuracy = ", stats['best_train_acc']['train_acc'], "test acc = ", stats['best_train_acc']['test_acc'])
            #print(time.time() - start_time)

    return stats

def test_accuracy(test_data, test_target, net):
    net.eval()
    outputs = net(test_data)
    preds = torch.argmax(outputs, dim=1)
    test_accuracy = float(torch.sum(preds == test_target))/float(test_target.size()[0])
    return test_accuracy

def get_test_loss(test_data, test_target, net):
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = criterion(net(test_data), test_target)
    return test_loss.cpu().data.numpy().item()
        
def train_step(net, inputs, targets, optimizer, iteration=None):
    # Set the network to training mode                                                                                                                                               
    net.train()
    # Zero out all gradients                                                                                                                                                         
    net.zero_grad()
    # Compute the loss (MSE in this case)                                                                                                                                            
    loss = 0.
    outputs = net(inputs)
    preds = torch.argmax(outputs, dim=1)
    accuracy = float(torch.sum(preds == targets))/float(targets.size()[0])
    criterion = torch.nn.CrossEntropyLoss()
    #if iteration==0:
    #    print("First output mean: ", outputs[0].mean())
    loss = criterion(outputs, targets)
    # Compute backprop updates                                                                                                                                                       
    loss.backward()
    # Take a step of GD                                                                                                                                                              
    optimizer.step()
    return loss.cpu().data.numpy().item(), accuracy
