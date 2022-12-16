
import torch
from torchvision import datasets

import os


######################################################################
#   functions from dlc_practical_prologue
######################################################################
def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def mnist_to_pairs(nb, train, target, one_hot_labels = True):
    input = torch.functional.F.avg_pool2d(train, kernel_size = 2)
    a = torch.randperm(input.size(0)) #permutation of indices
    nbTotalPairs = input.size(0)//2 # create maximum number of pairs, we take a specific number at the end
    a = a[:2 * nbTotalPairs].view(nbTotalPairs, 2)
    input = torch.cat((input[a[:, 0]], input[a[:, 1]]), 1)
    classes = target[a]
    
    pairs = input
    # create new targsets (doing it after the removal throws an error if the removal is flawed)
    target = (classes[:, 0] > classes[:, 1]).long()
    
    if one_hot_labels:
        target = convert_to_one_hot_labels(pairs, target)
        classes = torch.cat((convert_to_one_hot_labels(pairs, classes[:,0]).unsqueeze(1), convert_to_one_hot_labels(pairs, classes[:,1]).unsqueeze(1)), 1)
    else:
        target = target.float()

    return pairs[:nb], target[:nb], classes[:nb]


def generate_pair_sets(nb, data_dir=None, one_hot_labels = True):
    if data_dir is None:
        data_dir = os.environ.get('PYTORCH_DATA_DIR')
        if data_dir is None:
            data_dir = './data'

    train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
    train_input = train_set.data.view(-1, 1, 28, 28).float()
    train_target = train_set.targets

    test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)
    test_input = test_set.data.view(-1, 1, 28, 28).float()
    test_target = test_set.targets

    return mnist_to_pairs(nb, train_input, train_target, one_hot_labels) + \
           mnist_to_pairs(nb, test_input, test_target, one_hot_labels)



