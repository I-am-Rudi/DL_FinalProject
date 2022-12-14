import torch # pytorch package, allows using GPUs
from visualization import *
import framework.nn_fw as nn

from tqdm.notebook import tqdm

def compute_nb_errors(pred, target, one_hot_labels = False):
    """
    Returns: number of wrong predictions (Int)
    """

    if one_hot_labels:
        _, pred_index = torch.max(pred, 1)
        _, right_index = torch.max(target, 1)
        wrong = (pred_index != right_index).sum().item()
    
    else:
        pred = torch.round(pred)
        pred = torch.min(pred, torch.ones_like(pred))
        wrong = (pred != target).sum().item()
    return wrong


class Teacher():
    def __init__(self, optimizer, loss, data, device, batch_size):

        self.optimizer = optimizer
        self.loss = loss
        self.train_data, self.train_target, self.test_data, self.test_target = data
        self.train_data, self.train_target, self.test_data, self.test_target = self.train_data.to(device), self.train_target.to(device), self.test_data.to(device), self.test_target.to(device)
        self.batch_size = batch_size
        self.device = device

    def train(self, model, lr):
        
        num_errors=0
        for inputs, targets in zip(self.train_data.split(self.batch_size), self.train_target.split(self.batch_size)):
            
            output = model(inputs)

            optim = nn.SGD(lr, inputs.shape[0])
            optim.to_device(self.device)

            loss = nn.MSE(targets)
            
            grads = model.backward(loss)
            model.update(optim)
            
            num_errors += compute_nb_errors(output, targets)
        
        model_acc = 1 - num_errors/self.train_data.shape[0]

        return model_acc

    def test(self, model):
        '''Tests NN performance on test set.'''
        
        num_errors = 0
        test_loss = 0
        for inputs,targets in zip(self.test_data.split(self.batch_size), self.test_target.split(self.batch_size)):
            output = model(inputs)

            # compute test loss
            loss = self.loss(targets)
            test_loss += loss(output)

            # find most likely prediction
            num_errors += compute_nb_errors(output, targets)

        return test_loss, 1 - num_errors/self.test_data.shape[0]


def run_trial(model, layers, data, epochs, device, batch_size = 50, loss=nn.MSE, optimizer=nn.SGD, lr=.0001, name=""):

    NN = model(*layers)
    NN.to_device(device)
       
    teacher = Teacher(optimizer, loss, data, device, batch_size)

    train_accuracy = []
    test_loss = []
    test_accuracy = []
    best_test_accuracy = 0

    for epoch in range(1, epochs+1):
        train_accuracy.append(teacher.train(NN, lr))
        test_l, test_acc = teacher.test(NN)
        test_loss.append(test_l)
        test_accuracy.append(test_acc)

        
        if epoch == epochs:
            print('[Final Result] Accuracy(Training): ({:.3f}%), Accuracy(Test): ({:.3f}%)'.format(
            100. * train_accuracy[epoch-1], 100 * test_accuracy[epoch-1]) + '\n' + 'ran for a total of {} epochs'.format(epochs))
            print('\n', '\n')
        else:
            print('Epoch: ', epoch, '/', epochs,', Accuracy(Training): ({:.3f}%), Accuracy(Test): ({:.3f}%)'.format(
                100. * train_accuracy[epoch-1], 100 * test_accuracy[epoch-1]), end='\r')

    if test_accuracy[-1] > best_test_accuracy:
        best_test_accuracy = test_accuracy[-1]
        NN.save("best_model_" + name + '_epochs_{}'.format(epochs))
    return test_accuracy

def run_analysis(model, layers, data, nb_trials, epochs, device, name = "framework", batch_size = 50, lr=.001, loss=nn.MSE, optimizer =nn.SGD ):
    test_accuracy = []
    
    for _ in tqdm(range(nb_trials)):
        test_accuracy.append(run_trial(model, layers, data, epochs, device, batch_size, loss, optimizer, lr, name))

    mean = torch.mean(torch.tensor(test_accuracy), 0)
    std = torch.std(torch.tensor(test_accuracy), 0)

    plot_analysis(mean, std, epochs, nb_trials, name)

    return mean, std
    

    