# torch 
import torch # pytorch package, allows using GPUs
from torch import nn, optim
from torch.nn.modules import Module
import torch.optim as optim
from load_data import generate_pair_sets
from Visualization import plot_analysis

from tqdm.notebook import tqdm


class teacher():
    def __init__(self, optimizer, loss, device, size=1000):
        self.optimizer = optimizer
        self.loss = loss
        self.train_data, self.train_target, self.train_classes, self.test_data, self.test_target, self.test_classes = generate_pair_sets(size)
        self.train_data, self.train_target, self.train_classes, self.test_data, self.test_target, self.test_classes = self.train_data.to(device), self.train_target.to(device), self.train_classes.to(device), self.test_data.to(device), self.test_target.to(device), self.test_classes.to(device)

    def train(self, model, batch_size):
        model.train()
        
        num_correct=0
        for inputs,targets in zip(self.train_data.split(batch_size), self.train_target.split(batch_size)):
            output = model(inputs)
            #print(y_hat.size(), x_hat.size(), data.size(), self.train_target.size())
            loss = self.loss(output, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            _, predicted = torch.max(output,1)
            num_correct += (predicted==targets).sum().item()
        
        model_acc = num_correct/self.train_data.shape[0]

        return model_acc

    def test(self, model):
        '''Tests NN performance on test set.'''
        # evaluate model
        model.eval()

        output = model(self.test_data)
        
        # no need for batching
        # compute test loss
        test_loss = self.loss(output, self.test_target).item()
        # find most likely prediction
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # update number of correct predictions
        correct = pred.eq(self.test_target.view_as(pred)).sum().item()

        return test_loss, correct/self.test_data.shape[0]


def run_trial(model, epochs, layers, device, batch_size = 50, loss=nn.NLLLoss(), optimizer_name="SGD", lr=.1):

    NN = model(layers, activation = nn.ReLU(), out_activation = nn.LogSoftmax(dim=1))
       
    optimizer = getattr(optim, optimizer_name)(NN.parameters(), lr=lr)
    Teacher = teacher(optimizer, loss, device)

    train_accuracy = []
    test_loss = []
    test_accuracy = []
    

    for epoch in tqdm(range(1, epochs+1)):
        train_accuracy.append(Teacher.train(NN, batch_size))
        test_l, test_acc = Teacher.test(NN)
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

def run_analysis(model, nb_trials, epochs, layers, device, batch_size = 50, loss=nn.NLLLoss(), optimizer_name="SGD", lr=.1):
    test_accuracy = []
    
    for _ in range(nb_trials):
        test_accuracy.append(run_trial(model, epochs, layers, device, batch_size, loss, optimizer_name, lr))

    mean = torch.mean(torch.tensor(test_accuracy), 0)
    std = torch.std(torch.tensor(test_accuracy), 0)

    plot_analysis(mean, std, epochs, nb_trials)
    

    