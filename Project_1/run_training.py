# torch 
import torch # pytorch package, allows using GPUs
from torch import nn, optim
from torch.nn.modules import Module
import torch.optim as optim
from load_data import generate_pair_sets
from Visualization import *

from tqdm.notebook import tqdm

def nb_correct(pred, target):
    """
    Finds the index of the best pred and the index of the right classification,
    compares and counts the wrong predictions.

    Returns: number of wrong predictions (Int)
    """
    _, pred_index = torch.max(pred, dim=1)
    _, right_index = torch.max(target, dim=1)
    right = (pred_index == right_index).sum().item()
    return right

class teacher():
    def __init__(self, optimizer, loss, device, size=1000):
        self.optimizer = optimizer
        self.loss = loss
        self.train_data, self.train_target, self.train_classes, self.test_data, self.test_target, self.test_classes = generate_pair_sets(size)
        self.train_data, self.train_target, self.train_classes, self.test_data, self.test_target, self.test_classes = self.train_data.to(device), self.train_target.to(device), self.train_classes.to(device), self.test_data.to(device), self.test_target.to(device), self.test_classes.to(device)

    def train(self, model, batch_size):
        model.train()
        
        num_correct=0
        for inputs, targets, classes in zip(self.train_data.split(batch_size), self.train_target.split(batch_size), self.train_classes.split(batch_size)):
            if model.aux:
                output, aux1, aux2 = model(inputs)
                loss = self.loss(output, targets) + self.loss(aux1, classes[:, 0]) + self.loss(aux2, classes[:, 1])

            else:
                output = model(inputs)
                #print(y_hat.size(), x_hat.size(), data.size(), self.train_target.size())
                loss = self.loss(output, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            num_correct += nb_correct(output, targets)
        
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
            correct += nb_correct(output, targets)

        return test_loss, correct/self.test_data.shape[0]


def run_trial(model, epochs, layers, device, batch_size = 50, loss=nn.CrossEntropyLoss(), optimizer_name="SGD", lr=.1,BN=True, DO=.25):

    NN = model(layers, BN=BN, DO=DO)
       
    optimizer = getattr(optim, optimizer_name)(NN.parameters(), lr=lr)
    Teacher = teacher(optimizer, loss, device)

    train_accuracy = []
    test_loss = []
    test_accuracy = []
    

    for epoch in range(1, epochs+1):
        train_accuracy.append(Teacher.train(NN, batch_size))
        test_l, test_acc = Teacher.test(NN, batch_size)
        test_loss.append(test_l)
        test_accuracy.append(test_acc)

        
        if epoch == epochs:
            print('[Final Result] Accuracy(Training): ({:.3f}%), Accuracy(Test): ({:.3f}%)'.format(
            100. * train_accuracy[epoch-1], 100 * test_accuracy[epoch-1]) + '\n' + 'ran for a total of {} epochs'.format(epochs))
            print('\n', '\n')
        else:
            print('Epoch: ', epoch, ', Accuracy(Training): ({:.3f}%), Accuracy(Test): ({:.3f}%)'.format(
                100. * train_accuracy[epoch-1], 100 * test_accuracy[epoch-1]), end='\r')

    return test_accuracy

def run_analysis(model, nb_trials, epochs, layers, device, batch_size = 50, lr=.1, loss=nn.CrossEntropyLoss(), optimizer_name="SGD", BN=True, DO=.25):
    test_accuracy = []

    # use model_struct to get the name, save ONNX file and print out the structure of the current model 
    # wrapping in function ensures the proper calling of the dummy models destructor before going on with the actual traning
    
    name = model_struct(model, layers, BN, DO)
    
    
    for _ in tqdm(range(nb_trials)):
        test_accuracy.append(run_trial(model, epochs, layers, device, batch_size, loss, optimizer_name, lr, BN, DO))

    mean = torch.mean(torch.tensor(test_accuracy), 0)
    std = torch.std(torch.tensor(test_accuracy), 0)

    plot_analysis(mean, std, epochs, nb_trials, name)

    return name, mean, std
    

    