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

    def derivative(s): # to be clear this is not the derivative of the linear layer it just eliminates the need to check for the existence of an activation
        return torch.tensor([1])  # replaces the derivative of the activation if no activation is specified (a(x) = x -> a'(x) = 1)

    def forward(self, input):
        """take in the tuple of inputs and return the output of the linear layer as a tuple"""

        self.x = input # store for backward pass

        
        input = torch.einsum('ik,jk->ikj', self.w, input)  + self.b

        self.s = input # store for backward pass

        return input
    
    def backward(self, prev_layer, grad):
        
        self.db = prev_layer.derivative(self.s) * grad
        self.dw = torch.einsum('ik,ij->ikj', self.db, self.x)

        return {"type": "Linear", "w": self.dw, "b": self.db} 

    def update_params(self, optimizer):
        self.b -= optimizer.lr/(self.x.shape[0]) * torch.sum(self.db)
        self.w -= optimizer.lr/(self.x.shape[0]) * torch.sum(self.dw)


################################################################
# Activation functions
################################################################

class ReLU(Module):
    
    def forward(self, input):
        return input.apply_(lambda x: max(0, x))

    def derivative(self,input):
        der_bool = (self.forward(input) != torch.tensor([0])) # True if ReLU of x is not zero
        return der_bool + torch.tensor([0]) # only to give back non-boolean tensor explicitly

class Sigmoid(Module):
    
    def forward(self, input):
        return 1/(1+torch.exp(-input))

    def derivative(self,input):
        return self.forward(input) * (1 - self.forward(input))

class Tanh(Module):

    self.sigmoid = Sigmoid()

    def forward(self, input):
        return 2 * self.sigmoid(2*input) - 1

    def derivative(self, input):
        return 4 * self.sigmoid.derivative(2*input) # chain rule  