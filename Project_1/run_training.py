# torch 
import torch # pytorch package
from torch import nn, optim
from torch.nn.modules import Module
import torch.optim as optim
from load_data import generate_pair_sets
from Visualization import *

from tqdm.notebook import tqdm

import os
import pickle


def res_save(result, filename, path=None):
    filename = filename.replace(" ", "_")
    if path == None:
            with open(os.path.join(os.path.curdir, "results", filename + ".pkl"), 'wb') as f:
                pickle.dump(result, f)
    else:
        try:
            with open( path + filename + ".pkl", 'wb') as f:
                pickle.dump(result, f)
            
        except:
            raise Exception("Please enter a valid path when using the optional path argument!")

def res_load(filename, path=None):
    if path == None:
        with open(os.path.join(os.path.curdir, "results", filename + ".pkl"), 'rb') as f:
            result = pickle.load(f)
    else:
        try:
            with open(path + filename + ".pkl", "rb") as f:
                result = pickle.load(f)

        except:
            raise Exception("Please enter a valid path when using the optional path argument!")
    return result


def nb_correct(pred, target, one_hot_labels = True):
    """
    Returns: number of wrong predictions (Int)
    """

    if one_hot_labels:
        _, pred_index = torch.max(pred, 1)
        _, right_index = torch.max(target, 1)
        wrong = (pred_index == right_index).sum().item()
    
    else:
        pred = torch.round(pred)  
        pred = torch.min(pred, torch.ones_like(pred)) # count overshooting one as one
        wrong = (pred == target).sum().item()
    return wrong

class Teacher():
    def __init__(self, optimizer, loss, device, one_hot_labels, size=1000):
        self.optimizer = optimizer
        self.loss = loss
        self.train_data, self.train_target, self.train_classes, self.test_data, self.test_target, self.test_classes = generate_pair_sets(size, one_hot_labels=one_hot_labels)
        self.train_data, self.train_target, self.train_classes, self.test_data, self.test_target, self.test_classes = self.train_data.to(device), self.train_target.to(device), self.train_classes.to(device), self.test_data.to(device), self.test_target.to(device), self.test_classes.to(device)
        self.one_hot_labels = one_hot_labels

    def train(self, model, batch_size):
        '''Train NN'''
        model.train()
        num_correct=0
        for inputs, targets, classes in zip(self.train_data.split(batch_size), self.train_target.split(batch_size), self.train_classes.split(batch_size)):
            if (model.is_aux and model.is_classes):  # avoiding nested if statement
                output, aux1, aux2 = model(inputs)
                if self.one_hot_labels:
                    loss =  self.loss(output, targets) + .2 * self.loss(aux1, classes[:, 0]) +  .2 * self.loss(aux2, classes[:, 1])
                else:    
                    loss =  self.loss(output, targets) + .2 * self.loss(aux1, classes[:, 0].squeeze().float()) +  .2 * self.loss(aux2, classes[:, 1].squeeze().float())

            elif (model.is_aux):
                output, aux1 = model(inputs)
                loss = self.loss(output, targets) + .2 * self.loss(aux1, targets)
            else:
                output = model(inputs)
                loss = self.loss(output, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            num_correct += nb_correct(output, targets, one_hot_labels = self.one_hot_labels)
        
        model_acc = num_correct/self.train_data.shape[0]

        return model_acc

    def test(self, model, batch_size):
        '''Tests NN performance on test set.'''
        # evaluate model
        model.eval()
        
        correct = 0
        test_loss = 0
        for inputs,targets in zip(self.test_data.split(batch_size), self.test_target.split(batch_size)):
            output = model(inputs)
            # compute test loss
            test_loss += self.loss(output, targets).item()
            # find most likely prediction
            correct += nb_correct(output, targets, one_hot_labels = self.one_hot_labels)

        return test_loss, correct/self.test_data.shape[0]


def run_trial(model, epochs, layers, device, name, batch_size = 50, loss=nn.CrossEntropyLoss(), optimizer_name="SGD", lr=.1, one_hot_labels=True, fconv=False):

    NN = model(layers, one_hot_labels=one_hot_labels, fconv=fconv)
       
    optimizer = getattr(optim, optimizer_name)(NN.parameters(), lr=lr)
    teacher = Teacher(optimizer, loss, device, one_hot_labels)

    train_accuracy = []
    test_loss = []
    test_accuracy = []
    best_test_accuracy = 0
    

    for epoch in range(1, epochs+1):
        train_accuracy.append(teacher.train(NN, batch_size))
        test_l, test_acc = teacher.test(NN, batch_size)
        test_loss.append(test_l)
        test_accuracy.append(test_acc)

        
        if epoch == epochs:
            print('[Final Result] Accuracy(Training): ({:.3f}%), Accuracy(Test): ({:.3f}%)'.format(
            100. * train_accuracy[epoch-1], 100 * test_accuracy[epoch-1]) + '\n' + 'ran for a total of {} epochs'.format(epochs))
            print('\n', '\n')
        else:
            print('Epoch: ', epoch, ', Accuracy(Training): ({:.3f}%), Accuracy(Test): ({:.3f}%)'.format(
                100. * train_accuracy[epoch-1], 100 * test_accuracy[epoch-1]), end='\r')

    if test_accuracy[-1] > best_test_accuracy:  # only for plotting of architesture
        best_test_accuracy = test_accuracy[-1]
        NN.save("best_model_" + name.replace(" ", "_") + '_epochs_{}'.format(epochs))

    return test_accuracy

def run_analysis(model, nb_trials, epochs, layers, device, name, batch_size = 50, lr=.1, loss=nn.CrossEntropyLoss(), optimizer_name="SGD", one_hot_labels=True, fconv=False):
    test_accuracy = []
    
    
    for _ in tqdm(range(nb_trials)):
        test_accuracy.append(run_trial(model, epochs, layers, device, name, batch_size, loss, optimizer_name, lr, one_hot_labels, fconv))

    mean = torch.mean(torch.tensor(test_accuracy), 0)
    std = torch.std(torch.tensor(test_accuracy), 0)

    plot_analysis(mean, std, epochs, nb_trials, name)

    return name, mean, std
    

    