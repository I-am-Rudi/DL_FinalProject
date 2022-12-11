import os
import torch 
import pickle 
torch.set_grad_enabled(False)  # explicitly enforces expectation for the task 


def load_model(filename, path=None):
        if path == None:
            model = pickle.load(open(os.path.join(os.path.curdir, "saved_models", filename + ".pkl")))
        else:
            try:
                model = pickle.load(self, open( path + filename + ".pkl"))

            except:
                raise Exception("Please enter a valid path when using the optional path argument!")
        return model

class Module:
    def __init__(self):
        self.has_params = False

    # Defining __call__ method to keep the easy syntax for the forward prop
    def __call__(self, *input):
        return self.forward(*input)
    
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, gradwrtoutput):
        raise NotImplementedError
    
    def params(self):
        raise NotImplementedError
    
class Linear(Module):
    def __init__(self, in_size, out_size, bias=True):
        self.has_params = True
        
        self.w = torch.rand(out_size, in_size)
        
        if bias:
            self.b = torch.rand(out_size)
        else:
            self.b = torch.rand([0])

    def derivative(self, _, activation=False):  # to be clear this is not the derivative of the linear layer it just eliminates the need to check for the existence of an activation
        if activation:
            return self.w  # usual case where architecture: linear->activation 
        else:
            return torch.tensor([1])  # replaces the derivative of the activation if no activation is specified (a(x) = x -> a'(x) = 1)

    def forward(self, input):
        """take in the tuple of inputs and return the output of the linear layer as a tuple"""

        self.x = input  # store for backward pass
        input = torch.einsum("jk,ik->ij", self.w, input) + self.b
        self.s = input  # store for backward pass

        return input
    
    def backward(self, prev_layer, grad):
        
        self.db = prev_layer.derivative(self.s) * grad
        self.dw = torch.einsum('ik,ij->ikj', self.db, self.x)

        return {"type": "Linear", "w": self.dw, "b": self.db} 

    def update_params(self, optimizer):
        self.b -= optimizer(self.db, self.x.shape[0])
        self.w -= optimizer(self.dw, self.x.shape[0])


class Model(Module):
    """Base class for define general models"""

    def __init__(self):
        self.has_params = True
    
    def save(filename, path=None):
        if path == None:
            pickle.dump(self, open(os.path.join(os.path.curdir, "saved_models", filename + ".pkl")))
        else:
            try:
                pickle.dump(self, open( path + filename + ".pkl"))
            except:
                raise Exception("Please enter a valid path when using the optional path argument!")

class Sequential(Model):
    def __init__(self, *layers):
        super().__init__()
        self.layers = [i for i in layers]

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        
        return input

    def backward(self, loss):  # the loss functions will be initialized with target, they get one parameter as an instance which will be the output
        
        grads = [1]  # makes the Backprop of the loss work even if the ouput layers is not an activation
        self.layers.append(loss)
        self.layers = self.layers[::-1]
        
        for i in range(1, len(self.layers)):
            grads.append(self.layers[i].backward(self.layers[i-1], grads[i-1]))

        self.layers = self.layers[::-1]
        self.layers = self.layers[:-1]  # reformat the layer variable

        grads = grads[1:]

        return grads
    
    def update(self, optimizer):
        for layer in self.layers:
            if layer.has_params:
                layer.update_params(optimizer)  # putting the actual parameter update inside of the class for the layer, should give more flexibility fo adding new types

################################################################
# Optimizer
################################################################

class Optimizer(Module):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

class SGD(Optimizer):
    
    def forward(self, grad, batch_size):
        return self.lr/batch_size * torch.sum(grad, 0)

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
        return 2 * (pred - self.target)

################################################################
# Activation functions
################################################################

class Activation(Module):

    def backward(self, prev_layer, grad):
        return prev_layer.derivative(self.x, activation=True)

class ReLU(Activation):
    
    def forward(self, input):
        self.x = input.apply_(lambda x: max(0, x))
        return self.x

    def derivative(self, input, activation = False):
        if activation:
            raise Exception("Chaining of two activation functions directly after one another!")

        der_bool = (self.forward(input) != torch.tensor([0]))  # True if ReLU of x is not zero
        return der_bool + torch.tensor([0])  # only to give back non-boolean tensor explicitly

class Sigmoid(Activation):
    
    def forward(self, input):
        self.x = 1/(1+torch.exp(-input))
        return self.x

    def derivative(self, input, activation = False):
        if activation:
            raise Exception("Chaining of two activation functions directly after one another!")
        
        return self.forward(input) * (1 - self.forward(input))

class Tanh(Activation):

    def __init__(self):
        super().__init__()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        self.x = 2 * self.sigmoid(2 * input) - 1
        return self.x

    def derivative(self, input, activation = False):
        if activation:
            raise Exception("Chaining of two activation functions directly after one another!")
        
        return 4 * self.sigmoid.derivative(2*input)  # chain rule  