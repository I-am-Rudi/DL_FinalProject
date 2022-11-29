# torch 
import torch # pytorch package, allows using GPUs
import torch.nn.functional as F # implements forward and backward definitions of an autograd operation
from torch import nn, optim
from torch.nn.modules import Module
from torchvision import datasets # load data
from torch.autograd import Variable
import torch.optim as optim

class plain_model(nn.Module):
    def __init__(self, lin_layers, activation = nn.ReLU(), out_activation = nn.Sigmoid(), device=None, BN=True, DO=.25):
        super().__init__()
        ########### helpful parameters ###############
        self.aux = False
        if not BN and (DO == None):
            self.name = "Simple convolutional network" 
        else:
            self.name = "Improved convolutional network"

        ########### setting up conv layers ###########
        self.ch1 = nn.ModuleList()
        self.ch2 = nn.ModuleList()
        self.layers = nn.ModuleList()
        
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
    
        #flatten and full connected layers
        self.layers.append(nn.Flatten())

        layers_full = [1024] + lin_layers
        
        ################## FFN ##################  
        # variable number of hidden layers and neurons, controlled by 
        for i, layer in enumerate(lin_layers):
            self.layers.append(nn.Linear(layers_full[i], layer))
            if activation is not None:
                assert isinstance(activation, Module), \
                self.layers.append(activation)
            if BN:
                self.layers.append(nn.BatchNorm1d(layer))

        self.layers.append(nn.Linear(lin_layers[-1], 2))
        
        if DO != None:
            self.layers.append(nn.Dropout(DO))
        self.layers.append(out_activation)
        
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.to(self.device)

    def forward(self, input_data, train=True):
        x_ch1 = input_data[:, 0, :,:].unsqueeze(1)
        x_ch2 = input_data[:,1, :, :].unsqueeze(1)
        
        for ch_1, ch_2 in zip(self.ch1, self.ch2):
            x_ch1 = ch_1(x_ch1)
            x_ch2 = ch_2(x_ch2)
        x = torch.cat((x_ch1, x_ch2), 1)
        for layer in self.layers:
            x = layer(x)
        return x

class ws_model(nn.Module):
    def __init__(self, lin_layers, activation = nn.ReLU(), out_activation = nn.Sigmoid(), device=None, BN=True, DO=.25):
        super().__init__()
        ########### helpful parameters ###############
        self.aux = False
        if BN or DO != None:
            self.name = "Weight sharing network (no BN/Dropout)" 
        else:
            self.name = "Weight sharing network"

        ########### setting up conv layers ###########
        self.conv = nn.ModuleList()
        self.layers = nn.ModuleList()
        
        #channel 1
        self.conv.append(nn.Conv2d(1, 32, kernel_size=3))
        self.conv.append(activation)
        self.conv.append(nn.MaxPool2d(kernel_size=(2,2), stride = 2))
        self.conv.append(nn.Conv2d(32, 32, kernel_size=3))
        self.conv.append(activation)
    
        #flatten and full connected layers
        self.layers.append(nn.Flatten())

        layers_full = [1024] + lin_layers
        
        ################## FFN ##################  
        # variable number of hidden layers and neurons, controlled by 
        for i, layer in enumerate(lin_layers):
            self.layers.append(nn.Linear(layers_full[i], layer))
            if activation is not None:
                assert isinstance(activation, Module), \
                self.layers.append(activation)
            
            if BN:
                self.layers.append(nn.BatchNorm1d(layer))

        self.layers.append(nn.Linear(lin_layers[-1], 2))

        if DO != None:
            self.layers.append(nn.Dropout(DO))
        self.layers.append(out_activation)
        
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.to(self.device)

    def forward(self, input_data, train=True):
        x1 = input_data[:, 0, :,:].unsqueeze(1)
        x2 = input_data[:,1, :, :].unsqueeze(1)
        
        for conv in self.conv:
            x1 = conv(x1)
            x2 = conv(x2)
        x = torch.cat((x1, x2), 1)
        for layer in self.layers:
            x = layer(x)
        return x

class naive_aux_model(nn.Module):
    def __init__(self, lin_layers, activation = nn.ReLU(), out_activation = nn.Sigmoid(), device=None, BN=True, DO=.25):
        super().__init__()
        self.aux = True
        if BN or DO != None:
            self.name = "Weight sharing network (no BN/Dropout)" 
        else:
            self.name = "Weight sharing network"

        ########### setting up blocks ###########
        self.conv = nn.ModuleList()
        self.aux = nn.ModuleList()
        self.layers = nn.ModuleList()

        
        # conv layers
        self.conv.append(nn.Conv2d(1, 32, kernel_size=3))
        self.conv.append(activation)
        self.conv.append(nn.MaxPool2d(kernel_size=(2,2), stride = 2))
        self.conv.append(nn.Conv2d(32, 32, kernel_size=3))
        self.conv.append(activation)

        # auxiliary classifier
        self.aux.append(nn.Conv2d(32, 64, kernel_size=3))
        self.aux.append(activation)
        self.aux.append(nn.Flatten())
        self.aux.append(nn.Linear(256, 128))
        self.aux.append(activation)
        self.aux.append(nn.Linear(128, 10))

    
        # flatten and full connected layers
        self.layers.append(nn.Flatten())

        layers_full = [1024] + lin_layers
        
        ################## FFN ##################  
        # variable number of hidden layers and neurons, controlled by 
        for i, layer in enumerate(lin_layers):
            self.layers.append(nn.Linear(layers_full[i], layer))
            if activation is not None:
                assert isinstance(activation, Module), \
                self.layers.append(activation)
            
            if BN:
                self.layers.append(nn.BatchNorm1d(layer))

        self.layers.append(nn.Linear(lin_layers[-1], 2))

        if DO != None:
            self.layers.append(nn.Dropout(DO))
        self.layers.append(out_activation)
        
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.to(self.device)

    def forward(self, input_data, train=True):
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

            return x, class_x1, class_x2
        
        else:
            return x
