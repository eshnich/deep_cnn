import trainer
import numpy as np
import torch
import dataset
import options_parser as op
from resnet_cifar10.imagenet_mini import get_imagenet_trainloader, get_imagenet_testloader

def main(options):
    #seed = options.seed
    #seeds = [4, 44, 3, 33, 333]
    seeds = [44, 3, 33, 333]
    size = options.size
    num_classes = options.num_classes
    width = options.width
    trials = options.trials
    run_idx = options.run_idx

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    #depths = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    #depths = [10, 12, 14, 16, 18, 20]
    #depths = [22, 24, 26, 28, 30]

    if run_idx == 0:
        depths = [1, 2, 3]
    elif run_idx == 1:
        depths = [4, 5, 6]
    elif run_idx == 2:
        depths = [7, 8, 9]
    elif run_idx == 3:
        depths = [10, 12, 15]

    #train_loader, test_loader, classes = dataset.make_dataset(size, num_classes)
    
    all_classes = [0, 1, 2, 3, 4, 7, 9, 14, 16, 17]
    '''
    0 kit_fox
    1 English_setter
    2 Siberian_husky
    3 Australian_terrier
    4 English_springer
    7 Egyptian_cat
    9 Persian_cat
    14 malamute
    16 Great_Dane
    17 Walker_hound
    '''
    classes = all_classes[:num_classes]
    train_loader, transform = get_imagenet_trainloader(classes)
    test_loader = get_imagenet_testloader(classes, transform, batch_size = 50*num_classes)
    classes_arg = range(num_classes) # the imagenet test loader already renumbers the classes, so we don't need to do this in the train script.

    file = open("imagenet32_log_extratrials/size{}_width{}_{}classes_{}.txt".format(size, width, num_classes, run_idx), "w")
    file.write('depth trial ' + str(seeds) + '\n')
    for trial in range(trials): 
    
        seed = seeds[trial]
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)

        for depth in depths:
                                                                                                                                  
            results = trainer.train_net(train_loader, test_loader, depth, size, classes_arg, width, options.model)
            file.write(str(depth) + " " + str(trial) + " " + str(results) + "\n")
    
    file.close()

if __name__ == "__main__":
    options = op.setup_options()
    main(options)
