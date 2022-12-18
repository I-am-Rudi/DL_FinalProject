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
epochs = 100
nb_trials = 150

# %%
sm = res_load("Simple_convolutional_network")
ws = res_load("Weight_sharing_network")

plot_comparison(epochs, nb_trials, sm, ws)

# %%
ws = res_load("Weight_sharing_network")
saux = res_load("Simple_Auxiliary_classifier_network")
caux = res_load("Auxiliary_classifier_network_using_classes")

plot_comparison(epochs, nb_trials, ws, saux)

# %%
saux = res_load("Simple_Auxiliary_classifier_network")
caux = res_load("Auxiliary_classifier_network_using_classes")

plot_comparison(epochs, nb_trials, saux, caux)

# %%
sm = res_load("Simple_convolutional_network")
fc = res_load("Fully_convolutional_network")

plot_comparison(epochs, nb_trials, sm, fc)

# %%
ws = res_load("Weight_sharing_network")
fcws = res_load("Fully_convolutional_weight_sharing_network")
plot_comparison(epochs, nb_trials, ws, fcws)

# %%
saux = res_load("Simple_Auxiliary_classifier_network")
fcsaux = res_load("Fully_convolutional_auxiliary_classifier_network")
plot_comparison(epochs, nb_trials, saux, fcsaux)

# %%
caux = res_load("Auxiliary_classifier_network_using_classes")
fccaux = res_load("Fully_convolutional_auxiliary_classifier_network_using_classes")
plot_comparison(epochs, nb_trials, caux, fccaux)

# %%
sm = res_load("Simple_convolutional_network")
oh = res_load("Simple_convolutional_network_one_hot_labels")
oh = (oh[0] + "=False", oh[1], oh[2])  # not adding the False was an oversight on my part, if I planned to add the false later i wouldn't have used tuples
plot_comparison(epochs, nb_trials, sm, smoh)

# %%
ws = res_load("Weight_sharing_network")
oh = res_load("Weight_sharing_network_one_hot_labels")
oh = (oh[0] + "=False", oh[1], oh[2])  # not adding the False was an oversight on my part, if I planned to add the false later i wouldn't have used tuples
plot_comparison(epochs, nb_trials, ws, oh)

# %%
saux = res_load("Simple_Auxiliary_classifier_network")
oh = res_load("Simple_Auxiliary_classifier_network_one_hot_labels")
oh = (oh[0] + "=False", oh[1], oh[2])  # not adding the False was an oversight on my part, if I planned to add the false later i wouldn't have used tuples
plot_comparison(epochs, nb_trials, saux, oh)

# %%
# The Auxiliary classifier network using classes with one_hot_labels=False was unable to learn the features of the dataset in a meaningful way

# %%
wsoh = res_load("Weight_sharing_network_one_hot_labels")
wsoh = (wsoh[0] + "=False", wsoh[1], wsoh[2])  # not adding the False was an oversight on my part, if I planned to add the false later i wouldn't have used tuples
sauxoh = res_load("Simple_Auxiliary_classifier_network_one_hot_labels")
sauxoh = (sauxoh[0] + "=False", sauxoh[1], sauxoh[2])  # not adding the False was an oversight on my part, if I planned to add the false later i wouldn't have used tuples
plot_comparison(epochs, nb_trials, wsoh, sauxoh)

# %%
smoh = res_load("Simple_convolutional_network_one_hot_labels")
smoh = (smoh[0] + "=False", smoh[1], smoh[2])  # not adding the False was an oversight on my part, if I planned to add the false later i wouldn't have used tuples
wsoh = res_load("Weight_sharing_network_one_hot_labels")
wsoh = (wsoh[0] + "=False", wsoh[1], wsoh[2])  # not adding the False was an oversight on my part, if I planned to add the false later i wouldn't have used tuples
plot_comparison(epochs, nb_trials, smoh, wsoh)


