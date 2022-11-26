# torch 
import torch # pytorch package, allows using GPUs
from torch import nn, optim
from torch.nn.modules import Module
import torch.optim as optim
from load_data import generate_pair_sets

from tqdm.notebook import tqdm


class teacher():
    def __init__(self, optimizer, loss, device, size=1000):
        self.optimizer = optimizer
        self.loss = loss
        self.train_data, self.train_target, self.train_classes, self.test_data, self.test_target, self.test_classes = generate_pair_sets(size)
        self.train_data, self.train_target, self.train_classes, self.test_data, self.test_target, self.test_classes = self.train_data.to(device), self.train_target.to(device), self.train_classes.to(device), self.test_data.to(device), self.test_target.to(device), self.test_classes.to(device)

    def train(self, model, epoch):
        model.train()

        self.optimizer.zero_grad()
        output = model(self.train_data)
        #print(y_hat.size(), x_hat.size(), data.size(), self.train_target.size())
        loss = self.loss(output.squeeze(), self.train_target)
        loss.backward()
        self.optimizer.step()

        _,predicted = torch.max(output,1)
        num_correct = (predicted==self.train_target).sum().item()
        model_acc = num_correct/self.train_data.shape[0]
        model_loss = loss.item()

        #print('Epoch: ', epoch, ' - train_loss: ',model_loss, ' - train_acc: ',model_acc)
        return model_acc

    def test(self, model):
        '''Tests NN performance on test set.'''
        # evaluate model
        model.eval()

        output = model(self.test_data)
        
        # compute test loss
        test_loss = self.loss(output, self.train_target).item()
        # find most likely prediction
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # update number of correct predictions
        correct = pred.eq(self.train_target.view_as(pred)).sum().item()

        return test_loss, correct/self.test_data.shape[0]


def run_trial(model, epochs, layers, device, loss=nn.NLLLoss(), optimizer_name="Adam", lr=.1):

    NN = model(layers, activation = nn.ReLU(), out_activation = nn.LogSoftmax(dim=1))
       
    optimizer = getattr(optim, optimizer_name)(NN.parameters(), lr=lr)
    Teacher = teacher(optimizer, loss, device)

    train_accuracy = []
    test_loss = []
    test_accuracy = []
    

    for epoch in tqdm(range(epochs)):
        train_accuracy.append(Teacher.train(NN, epoch))
        test_l, test_acc = Teacher.test(NN)
        test_loss.append(test_l)
        test_accuracy.append(test_acc)

        
        if epoch == epochs:
            print('Final Result Accuracy(Training): ({:.3f}%), Accuracy(Test): ({:.3f}%)'.format(
            100. * train_accuracy[epoch-1], 100 * test_accuracy[epoch-1]))
            print('\n', '\n')
        elif epoch == epochs-1:
            print('Epoch: ', epoch, ', Accuracy(Training): ({:.3f}%), Accuracy(Test): ({:.3f}%)'.format(
                100. * train_accuracy[epoch-1], 100 * test_accuracy[epoch-1]), end='\r')
            print('', end = '\r')
            print('')
        else:
            print('Epoch: ', epoch, ', Accuracy(Training): ({:.3f}%), Accuracy(Test): ({:.3f}%)'.format(
                100. * train_accuracy[epoch-1], 100 * test_accuracy[epoch-1]), end='\r')

    return test_accuracy, test_loss