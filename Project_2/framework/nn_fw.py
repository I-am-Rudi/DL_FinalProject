import os
import torch 
import pickle 
torch.set_grad_enabled(False)  # explicitly enforces expectation for the task 


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

class Module:
    def __init__(self):
        self.has_params = False
        self.device = torch.device('cpu')

    # Defining __call__ method to keep the easy syntax for the forward prop
    def __call__(self, *input):
        return self.forward(*input)
    
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, gradwrtoutput):
        raise NotImplementedError
    
    def params(self):
        return []
class Linear(Module):
    def __init__(self, in_size, out_size, bias=True):
        super().__init__()
        self.has_params = True
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias

        self.w = torch.distributions.uniform.Uniform(0, 1).sample([out_size, in_size])  # torch.rand(out_size, in_size) * torch.sqrt(torch.tensor([2])/in_size)
        
        if bias:
            self.b = torch.distributions.uniform.Uniform(0, 1).sample([out_size]) # torch.rand(out_size) * torch.sqrt(torch.tensor([2])/in_size)
        else:
            self.b = torch.rand([0]) # change that

    def derivative(self, _, activation=False):  # to be clear this is not the derivative of the linear layer it just eliminates the need to check for the existence of an activation
        if activation:
            return self.w.t()  # usual case where architecture: linear->activation 
        else:
            return torch.tensor([1])  # replaces the derivative of the activation if no activation is specified (a(x) = x -> a'(x) = 1)

    def forward(self, input):
        """take in the tuple of inputs and return the output of the linear layer as a tuple"""

        self.x = input  # store for backward pass
        input = torch.einsum("jk,ik->ij", self.w, input) + self.b
        self.s = input  # store for backward pass

        return input

    def params(self):
        return [self.w, self.b]

    def backward(self, prev_layer, grad):
        self.db = prev_layer.derivative(self.s) * grad
        #self.db = torch.einsum("ik,jk->ij", prev_layer.derivative(self.s), grad)
        self.dw = torch.einsum('ik,ij->ikj', self.db, self.x)
        return self.db
    
    def to_device(self, device):
        self.w = self.w.to(device)
        self.b = self.b.to(device)

        self.device = device

    def update_params(self, optimizer):
        if self.bias:
            self.b -= optimizer(self.db)
        self.w -= optimizer(self.dw)
    
    def reset(self):
        self.w = torch.distributions.uniform.Uniform(0, 1).sample([self.out_size, self.in_size])  # torch.rand(out_size, in_size) * torch.sqrt(torch.tensor([2])/in_size)
        
        if self.bias:
            self.b = torch.distributions.uniform.Uniform(0, 1).sample([self.out_size]) # torch.rand(out_size) * torch.sqrt(torch.tensor([2])/in_size)


class Model(Module):
    """Base class for defining general models"""

    def __init__(self):
        super().__init__()
        self.has_params = True
    
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

class Sequential(Model):
    def __init__(self, *layers):
        super().__init__()
        self.layers = [layer for layer in layers]

        # in the case of layers being reused I want them to be reset, when initializing a new model with them:
        for layer in layers:  
            if layer.has_params:
                layer.reset()
            

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        
        return input

    def backward(self, loss):  # the loss functions will be initialized with target, they get one parameter as an instance which will be the output
        
        grads = [torch.tensor([1]).unsqueeze(1).to(self.device)]  # makes the Backprop of the loss work even if the ouput layers is not an activation
        self.layers.append(loss)
        self.layers = self.layers[::-1]
        
        for i in range(1, len(self.layers)):
            if i == 1:
                grads.append(self.layers[i].backward(self.layers[i-1], grads[i-1], loss = True))
            else:
                grads.append(self.layers[i].backward(self.layers[i-1], grads[i-1]))


        self.layers = self.layers[::-1]
        self.layers = self.layers[:-1]  # reformat the layer variable
        grads = grads[1:]
        grads = grads[::-1]
        return grads    

    def update(self, optimizer):
        for layer in self.layers:
            if layer.has_params:
                layer.update_params(optimizer)  # putting the actual parameter update inside of the class for the layer, should give more flexibility fo adding new types

    def params(self):
        params = []
        for layer in self.layers:
            params.append(layer.params)
        return params

    def to_device(self, device):
        for layer in self.layers:
            layer.to_device(device)
        self.device = device
################################################################
# Optimizer
################################################################

class Optimizer(Module):
    def __init__(self, lr, batch_size, device=None):
        super().__init__()
        self.lr = torch.tensor([lr])
        self.has_params = True
        self.batch_size = torch.tensor([batch_size])
    
    def params(self):
        return [self.lr]
    
    def to_device(self, device):
        self.lr = self.lr.to(device)
        self.batch_size = self.batch_size.to(device)
        self.device = device

class SGD(Optimizer):
    
    def forward(self, grad):
        return self.lr/self.batch_size * torch.sum(grad, 0)

################################################################
# Loss functions
################################################################

class Loss(Module):
    def __init__(self, target):
        #super().__init__(self)
        self.target = target  # I choose this initialization to make the loss compatible with the Backpropagation 
class MSE(Loss):

    def forward(self, pred):
        return (1/pred.size()[0]) * torch.sum(torch.pow(pred - self.target, 2))

    def derivative(self, pred, activation = False):
        return (2 * (pred - self.target))

################################################################
# Activation functions
################################################################

class Activation(Module):

    def __call__(self, input):
        self.x = self.forward(input)
        return self.x

    def backward(self, prev_layer, grad, loss = False):
        if loss:
            return torch.einsum("ik,jk->ij", prev_layer.derivative(self.x, activation=True), grad) 
        else:
            return torch.einsum("ik,jk->ji", prev_layer.derivative(self.x, activation=True), grad) 
    def to_device(self, device):
        self.device = device

class ReLU(Activation):
    
    def forward(self, input):
        return torch.max(input, torch.zeros_like(input))
        

    def derivative(self, input, activation = False):
        if activation:
            raise Exception("Chaining of two activation functions directly after one another!")

        der_bool = (self.forward(input) != torch.tensor([0]).to(self.device))  # True if ReLU of x is not zero
        return der_bool + torch.tensor([0]).to(self.device)  # only to give back non-boolean tensor explicitly

class Sigmoid(Activation):
    
    def forward(self, input):
        return 1/(1+torch.exp(-input))

    def derivative(self, input, activation = False):
        if activation:
            raise Exception("Chaining of two activation functions directly after one another!")
        
        return self.forward(input) * (1 - self.forward(input))

class Tanh(Activation):

    def __init__(self):
        super().__init__()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        return 2 * self.sigmoid.forward(2 * input) - 1

    def derivative(self, input, activation = False):
        if activation:
            raise Exception("Chaining of two activation functions directly after one another!")
        
        return 4 * self.sigmoid.derivative(2*input)  # chain rule  