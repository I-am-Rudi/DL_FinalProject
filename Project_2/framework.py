import torch 
import pickle 
torch.set_grad_enabled(False) #explicitly enforces expectation for the task 

class Module:

    # Defining __call__ method to keep the easy syntax for the forward prop
    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, gradwrtoutput):
        raise NotImplementedError
    
    def params(self):
        raise NotImplementedError
    
class Linear(Module):
    def __init__(self, in_size, out_size, bias=True):
        self.w = torch.rand(out_size, in_size)
        
        if bias:
            self.b = torch.rand(out_size)
        else:
            self.b = torch.rand([0])

    def derivative(s, activation=False): # to be clear this is not the derivative of the linear layer it just eliminates the need to check for the existence of an activation
        if activation:
            return self.w # usual case where architecture: linear->activation 
        else:
            return torch.tensor([1]) # replaces the derivative of the activation if no activation is specified (a(x) = x -> a'(x) = 1)

    def forward(self, input):
        """take in the tuple of inputs and return the output of the linear layer as a tuple"""

        self.x = input # store for backward pass

        
        input = torch.mv(self.w, input)  + self.b

        self.s = input # store for backward pass

        return input
    
    def backward(self, prev_layer, grad):
        
        self.db = prev_layer.derivative(self.s) * grad
        self.dw = torch.einsum('ik,ij->ikj', self.db, self.x)

        return {"type": "Linear", "w": self.dw, "b": self.db} 

    def update_params(self, optimizer):
        self.b -= optimizer.lr/(self.x.shape[0]) * torch.sum(self.db)
        self.w -= optimizer.lr/(self.x.shape[0]) * torch.sum(self.dw)


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        
        return input

    def backward(self, output, loss): # the loss functions will be initialized with target, they get one parameter as an instance which will be the output
        
        grads = []
        self.layers.append(loss)
        self.layers = self.layers[::-1]
        
        for i in range(1, rev_layers):
            grads.append(self.layers[i].backward(self.layers[i-1], ))

################################################################
# Activation functions
################################################################

class Activation(Module):

    def backward(self, prev_layer, grad):
        return  prev_layer.derivative( 0, activation = True)

class ReLU(Activation):
    
    def forward(self, input):
        return input.apply_(lambda x: max(0, x))

    def derivative(self,input):
        der_bool = (self.forward(input) != torch.tensor([0])) # True if ReLU of x is not zero
        return der_bool + torch.tensor([0]) # only to give back non-boolean tensor explicitly

class Sigmoid(Activation):
    
    def forward(self, input):
        return 1/(1+torch.exp(-input))

    def derivative(self,input):
        return self.forward(input) * (1 - self.forward(input))

class Tanh(Activation):

    self.sigmoid = Sigmoid()

    def forward(self, input):
        return 2 * self.sigmoid(2*input) - 1

    def derivative(self, input, activation = False):
        if activation:
            raise Exception("Chaining of two activation functions directly after one another!")
        return 4 * self.sigmoid.derivative(2*input) # chain rule  