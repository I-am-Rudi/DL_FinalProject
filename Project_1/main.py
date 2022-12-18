# %%
import os,sys
# torch 
import torch 
from torch import nn, optim
from torch.nn.modules import Module
from torchvision import datasets # load data
from torch.autograd import Variable
import torch.optim as optim

#own modules

from run_training import *
from Networks import *
from Visualization import *

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using cuda device for all tensor calculations.")
else:
    device = torch.device('cpu')
    print("Using cpu device for all tensor calculations.")

# %%
nb_trials = 150
epochs = 100
batch_size = 50
one_hot_labels = True
lr=.01
layers = [64, 64, 32]

# %%
name = "Simple convolutional network"
pm_results = run_analysis(Plain_model, nb_trials, epochs, layers, device, name, batch_size=batch_size, lr=lr, one_hot_labels=one_hot_labels)
res_save(pm_results, name)

# %%
name = "Weight sharing network"
ws_results = run_analysis(Ws_model,  nb_trials, epochs, layers, device, name, batch_size=batch_size, lr=lr, one_hot_labels=one_hot_labels)
res_save(ws_results, name)

# %%
name = "Simple Auxiliary classifier network"
s_aux_results = run_analysis(Simple_aux_model,  nb_trials, epochs, layers, device, name, batch_size=batch_size, lr=lr, one_hot_labels=one_hot_labels)
res_save(s_aux_results, name)

# %%
name = "Auxiliary classifier network using classes"
c_aux_results = run_analysis(Class_aux_model,  nb_trials, epochs, layers, device, name, batch_size=batch_size, lr=lr, one_hot_labels=one_hot_labels)
res_save(c_aux_results, name)

# %%
name = "Fully convolutional network"
pm_results = run_analysis(Plain_model, nb_trials, epochs, layers, device, name, batch_size=batch_size, lr=lr, one_hot_labels=one_hot_labels, fconv=True)
res_save(pm_results, name)

# %%
name = "Fully convolutional weight sharing network"
ws_results = run_analysis(Ws_model,  nb_trials, epochs, layers, device, name, batch_size=batch_size, lr=lr, one_hot_labels=one_hot_labels, fconv=True)
res_save(ws_results, name)

# %%
name = "Fully convolutional auxiliary classifier network"
s_aux_results = run_analysis(Simple_aux_model,  nb_trials, epochs, layers, device, name, batch_size=batch_size, lr=lr, one_hot_labels=one_hot_labels, fconv=True)
res_save(s_aux_results, name)

# %%
name = "Fully convolutional auxiliary classifier network using classes"
c_aux_results = run_analysis(Class_aux_model,  nb_trials, epochs, layers, device, name, batch_size=batch_size, lr=lr, one_hot_labels=one_hot_labels, fconv=True)
res_save(c_aux_results, name)

# %%
one_hot_labels = False  # The naming in the following part should be "... one_hot_labels=False" this is fixed before plotting

name = "Simple convolutional network one_hot_labels"
pm_results = run_analysis(Plain_model, nb_trials, epochs, layers, device, name, batch_size=batch_size, lr=lr, one_hot_labels=one_hot_labels, loss=nn.MSELoss())
res_save(pm_results, name)

# %%
name = "Weight sharing network one_hot_labels"
ws_results = run_analysis(Ws_model,  nb_trials, epochs, layers, device, name, batch_size=batch_size, lr=lr, one_hot_labels=one_hot_labels, loss=nn.MSELoss())
res_save(ws_results, name)

# %%
name = "Simple Auxiliary classifier network one_hot_labels"
s_aux_results = run_analysis(Simple_aux_model,  nb_trials, epochs, layers, device, name, batch_size=batch_size, lr=lr, one_hot_labels=one_hot_labels, loss=nn.MSELoss())
res_save(s_aux_results, name)

# %%
name = "Auxiliary classifier network using classes one_hot_labels"
c_aux_results = run_analysis(Class_aux_model,  nb_trials, epochs, layers, device, name, batch_size=batch_size, lr=lr, one_hot_labels=one_hot_labels, loss=nn.MSELoss())
res_save(c_aux_results, name)


