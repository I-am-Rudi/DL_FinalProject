import os
import pickle #  for storing and loading networks

# torch 
import torch 
from torch import nn, optim
from torch.nn.modules import Module
from torchvision import datasets # load data
from torch.autograd import Variable
import torch.optim as optim

def load_model(filename, path=None):
        if path == None:
            with open(os.path.join(os.path.curdir, "saved_models", filename + ".pkl"), 'rb') as f:
                model = pickle.load(f)
        else:
            try:
                with open(path + filename + ".pkl", "rb") as f:
                    model = pickle.load(f)

            except:
                raise Exception("Please enter a valid path when using the optional path argument!")
        return model


def convolutionize(layers, in_size):  # inspired by the lecture on fully convolutional neural networks
    
    result_layers = nn.ModuleList()   
    
    x = torch.zeros((1, ) + in_size) 
    
    for m in layers:
        if isinstance(m, nn.Linear):
            n = nn.Conv2d(in_channels = x.size(1), out_channels = m.weight.size(0), kernel_size = (x.size(2), x.size(3)))

            # copying of weights and biases can be left out as we do the transformation when initializing

            m=n
        x = m(x)
        result_layers.append(m)
    return result_layers

class LinearBase(nn.Module):
    def __init__(self, lin_layers, activation, out_activation, device, one_hot_labels, fconv):
        super().__init__()
        self.is_aux = False
        self.is_classes = False

        self.layers = nn.ModuleList()
        
        if not fconv:
            self.layers.append(nn.Flatten())

        self.lin_layers = [1024] + lin_layers
        
        ################## FFN ##################  
        # variable number of hidden layers and neurons, controlled by 
        for i, layer in enumerate(lin_layers):
            self.layers.append(nn.Linear(self.lin_layers[i], layer))
            self.layers.append(activation)
        
        if one_hot_labels:
            self.layers.append(nn.Linear(lin_layers[-1], 2))
        
        else:
            self.layers.append(nn.Linear(lin_layers[-1], 1))
        
        self.layers.append(out_activation)

        if fconv:
            self.layers = convolutionize(self.layers, (64, 4, 4))


    def save(self, filename, path=None):
        if path == None:
            with open(os.path.join(os.path.curdir, "saved_models", filename + ".pkl"), 'wb') as f:
                pickle.dump(self, f)
        else:
            try:
                with open( path + filename + ".pkl", 'wb') as f:
                    pickle.dump(self, f)
                
            except:
                raise Exception("Please enter a valid path when using the optional path argument!")

class Plain_model(LinearBase):
    def __init__(self, lin_layers, activation = nn.ReLU(), out_activation = nn.Sigmoid(), device=None, one_hot_labels = True, fconv=False):
        super().__init__(lin_layers, activation, out_activation, device, one_hot_labels, fconv)  # call Linear Base constructor
        
        ########### helpful parameters ###############
        self.name = "Simple convolutional network" 

        ########### setting up conv layers ###########
        self.ch1 = nn.ModuleList()
        self.ch2 = nn.ModuleList()
        
        #channel 1
        self.ch1.append(nn.Conv2d(1, 32, kernel_size=3))
        self.ch1.append(activation)
        self.ch1.append(nn.MaxPool2d(kernel_size=(2,2), stride = 2))
        self.ch1.append(nn.Conv2d(32, 32, kernel_size=3))
        self.ch1.append(activation)

        #channel 2
        self.ch2.append(nn.Conv2d(1, 32, kernel_size=3))
        self.ch2.append(activation)
        self.ch2.append(nn.MaxPool2d(kernel_size=(2,2), stride = 2))
        self.ch2.append(nn.Conv2d(32, 32, kernel_size=3))
        self.ch2.append(activation)

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.to(self.device)


    def forward(self, input_data):
        x_ch1 = input_data[:, 0, :,:].unsqueeze(1)
        x_ch2 = input_data[:,1, :, :].unsqueeze(1)
        
        for ch_1, ch_2 in zip(self.ch1, self.ch2):
            x_ch1 = ch_1(x_ch1)
            x_ch2 = ch_2(x_ch2)
        x = torch.cat((x_ch1, x_ch2), 1)
        for layer in self.layers:
            x = layer(x)
        return torch.squeeze(x)

class Ws_model(LinearBase):
    def __init__(self, lin_layers, activation = nn.ReLU(), out_activation = nn.Sigmoid(), device=None, one_hot_labels = True, fconv=False):
        super().__init__(lin_layers, activation, out_activation, device, one_hot_labels, fconv)  # call Linear Base constructor
        
        ########### helpful parameters ###############
        self.name = "Weight sharing network"
        ########### setting up conv layers ###########
        self.conv = nn.ModuleList()
        
        #channel 1
        self.conv.append(nn.Conv2d(1, 32, kernel_size=3))
        self.conv.append(activation)
        self.conv.append(nn.MaxPool2d(kernel_size=(2,2), stride = 2))
        self.conv.append(nn.Conv2d(32, 32, kernel_size=3))
        self.conv.append(activation)

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.to(self.device)

    def forward(self, input_data):
        x1 = input_data[:, 0, :,:].unsqueeze(1)
        x2 = input_data[:,1, :, :].unsqueeze(1)
        
        for conv in self.conv:
            x1 = conv(x1)
            x2 = conv(x2)
        x = torch.cat((x1, x2), 1)
        for layer in self.layers:
            x = layer(x)
        return torch.squeeze(x)

class Class_aux_model(LinearBase):
    def __init__(self, lin_layers, activation = nn.ReLU(), out_activation = nn.Sigmoid(), device=None, one_hot_labels = True, fconv=False):
        super().__init__(lin_layers, activation, out_activation, device, one_hot_labels, fconv)  # call Linear Base constructor
        ########### helpful parameters ###############
        self.is_aux = True
        self.is_classes = True 
        self.name = "Auxiliary classifier network using classes"

        ########### setting up blocks ###########
        self.conv = nn.ModuleList()
        self.aux = nn.ModuleList()

        
        # conv layers
        self.conv.append(nn.Conv2d(1, 32, kernel_size=3))
        self.conv.append(activation)
        self.conv.append(nn.MaxPool2d(kernel_size=(2,2), stride = 2))
        self.conv.append(nn.Conv2d(32, 32, kernel_size=3))
        self.conv.append(activation)

        # auxiliary classifier
        self.aux.append(nn.Conv2d(32, 64, kernel_size=3))
        self.aux.append(activation)
        if not fconv:
            self.aux.append(nn.Flatten())
        
        self.aux.append(nn.Linear(256, 128))
        self.aux.append(activation)
        
        if one_hot_labels:
            self.aux.append(nn.Linear(128, 10))
        else:
            self.aux.append(nn.Linear(128, 1))

        if fconv:
            self.aux = convolutionize(self.aux, (32, 4, 4))

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.to(self.device)

    def forward(self, input_data):
        x1 = input_data[:, 0, :,:].unsqueeze(1)
        x2 = input_data[:,1, :, :].unsqueeze(1)
        
        for conv in self.conv:
            x1 = conv(x1)
            x2 = conv(x2)

        
        x = torch.cat((x1, x2), 1)
        for layer in self.layers:
            x = layer(x)

        if self.training:
            class_x1 = torch.clone(x2)
            class_x2 = torch.clone(x1)
            for aux in self.aux:
                class_x1 = aux(class_x1)
                class_x2 = aux(class_x2)

            return torch.squeeze(x), torch.squeeze(class_x1), torch.squeeze(class_x2)
        
        else:
            return torch.squeeze(x)


class Simple_aux_model(LinearBase):
    def __init__(self, lin_layers, activation = nn.ReLU(), out_activation = nn.Sigmoid(), device=None, one_hot_labels = True, fconv=False):
        super().__init__(lin_layers, activation, out_activation, device, one_hot_labels, fconv)  # call Linear Base constructor
        
        ########### helpful parameters ###############
        self.is_aux = True
        self.name = "Simple Auxiliary classifier network"

        ########### setting up blocks ###########
        self.conv = nn.ModuleList()
        self.aux = nn.ModuleList()

        
        # conv layers
        self.conv.append(nn.Conv2d(1, 32, kernel_size=3))
        self.conv.append(activation)
        self.conv.append(nn.MaxPool2d(kernel_size=(2,2), stride = 2))
        self.conv.append(nn.Conv2d(32, 32, kernel_size=3))
        self.conv.append(activation)

        # auxiliary classifier (most simple implementation of this classifier just do same task and add up the losses)
        for i, layer in enumerate(lin_layers):
            if i > 1:  # skipping the first linear layer + activation
                self.aux.append(nn.Linear(self.lin_layers[i], layer))
                self.aux.append(activation)

        if one_hot_labels:
            self.aux.append(nn.Linear(lin_layers[-1], 2))
        else:
            self.aux.append(nn.Linear(lin_layers[-1], 1))

        self.aux.append(out_activation)
        
        if fconv:
            self.aux = convolutionize(self.aux, (64, 1, 1))

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.to(self.device)

    def forward(self, input_data):
        x1 = input_data[:, 0, :,:].unsqueeze(1)
        x2 = input_data[:,1, :, :].unsqueeze(1)
        
        for conv in self.conv:
            x1 = conv(x1)
            x2 = conv(x2)

        
        x = torch.cat((x1, x2), 1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 2:
                x1 = torch.clone(x)
        
        if self.training:
            for aux in self.aux:
                x1 = aux(x1)

        
        if self.training:
            return torch.squeeze(x), torch.squeeze(x1)
        
        else:
            return torch.squeeze(x)

