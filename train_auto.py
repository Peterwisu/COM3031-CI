"""

Computational Intelligence Coursework

Name : Wish, Taimoor, Ionut


"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader  as DataLoader 
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import torch.optim as optim
from model.autoencoder import AutoEncoder
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize 
import seaborn as sns
import pandas as pd
import os

# import utility function 
from utils import save_logs , plot_diff ,cm_plot ,roc_plot , save_model


# softmax activation function
softmax = nn.Softmax(dim=1)


"""
*****
train :  Training stage of a model
*****


******
inputs
******
    
    model : Model for training

    device :  string contain name of device use for training and evaluation

    loss_criterion : Objective function 

    optimizer : Optimizer use for optimizing model

    training_set : training dataset

    testing_set : testing dataset

    nepochs : numbers of epochs

    classes :  array cotaining a name of classes

*******
returns
*******

    None
"""
def train(model, device, loss_criterion, optimizer, training_set, validation_set,nepochs, classes, savename):
    
    global global_epochs

    global_epochs = 1
    
    # array of logs
    loss_train_logs = np.array([])
    loss_vali_logs  = np.array([])


    while global_epochs < nepochs+1:

        progress_bar = tqdm(enumerate(training_set))
        
        running_loss = 0  # Total loss in epochs
    
        iter_inbatch = 0  # Iteration in batch

        for _ , (images, labels) in progress_bar:
            
            # move dataset to same device as model
            images = images.to(device)
            labels = labels.to(device)

            # set model to training stage
            model.train()

            # reset optimizer
            optimizer.zero_grad() 

            # forward pass
            predicted = model(images)


            loss = loss_criterion(predicted, images)

    

            

            # calculate loss 
            
 

            # backprop and optmizer update weight
            loss.backward()
            optimizer.step()
        
            running_loss +=loss.item()
            iter_inbatch +=1
            
            
            progress_bar.set_description("Training Epochs : {} , Loss : {} ".format(global_epochs,(running_loss/iter_inbatch)))

        # get loss in current epoch
        train_loss = running_loss/iter_inbatch # calculate a mean loss (total loss / iteration)


        eval_model(model,validation_set,loss_criterion,device)

        original, generate = visualize(model, validation_set, device)

    
        writer.add_figure("Image/original",original,global_epochs)     
        writer.add_figure("Image/generate",generate,global_epochs)

        # increment epoch
        global_epochs +=1


def eval_model(model, data_loader, loss_criterion, device) :

    
    eval_prog = tqdm(enumerate(data_loader))

    eval_running_loss = 0 

    eval_iter_inbatch  = 0 

    with torch.no_grad():

        for _, (images, _ ) in eval_prog: 

            images = images.to(device)

            
            model.eval()

            predicted = model(images)

            loss = loss_criterion(predicted, images)
            
            eval_running_loss += loss.item()
            eval_iter_inbatch +=1

            eval_prog.set_description("EVAL : loss {} ".format(eval_running_loss/eval_iter_inbatch))




            

    
def visualize(model,data,device):

    
    data_iter = iter(data)

    images, labels = next(data_iter)

    #generate image

    model.eval()

    gen = model(images.to(device))
    
    gen_img = gen.detach().clone().cpu().numpy()

    images =  images.numpy()
    
    
    #Original Images
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
    for idx in np.arange(5):
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        print(images[idx].shape)
        plt.imshow(images[idx].T)
        ax.set_title(classes[labels[idx]])
    
    original = ax.get_figure()
    plt.close(fig)

    #Reconstructed Images
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
    for idx in np.arange(5):
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        plt.imshow(gen_img[idx].T)
        ax.set_title(classes[labels[idx]])
    
    generate = ax.get_figure()
    plt.close(fig)


    return original , generate



    
    





"""
Main function  
"""

if __name__ == "__main__":
    

    savename ="AUTo"

    #  Setup tensorboard
    writer = SummaryWriter("../CI_logs/{}".format(savename))

    device = "cuda" if torch.cuda.is_available else "cpu"

    batch_size = 32

    nepochs = 100

    print("Using  **{}** as a device ".format(device))
    print("Batch Size : {}".format(batch_size))
    print("Iteration : {} epochs".format(nepochs))
    

    print("Loading dataset ....")
    transform = transforms.Compose([#transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
                                    #transforms.RandomRotation(degrees=(0, 180)),
                                    
                                    #transforms.RandomGrayscale(p=0.1),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                    ]
                                    )
    
    # Prepare Dataset
    training_set = torchvision.datasets.CIFAR10(root='./../data', train=True, download=True, transform=transform)

    testing_set  = torchvision.datasets.CIFAR10(root='./../data', train=False, download=True, transform=transform)


    training_data, validation_data = random_split(training_set, [40000, 10000])
    
    

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True, num_workers=2)

    print("Training Dataset: {}".format(len(training_data)))
    print("Validation Dataset: {}".format(len(validation_data)))
    print("Testing Dataset: {}".format(len(testing_set)))
    
    # labels of dataset
    classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print("Classes in dataset  : {} ".format(classes))

    
    # Classifier Models
    model = AutoEncoder().to(device)

    print(model)

    
    
    # Loss function Objective function 
    loss = nn.L1Loss()

    # Optimizer 
    optimizer = optim.Adam([params  for params in model.parameters() if params.requires_grad], lr=0.0001)
    
    print("Total parameters : {}".format(sum(params.numel() for params in model.parameters())))
    print("Trainable parameters : {}".format(sum(params.numel() for params in model.parameters() if params.requires_grad)))


    train(model, device, loss, optimizer, train_loader, validation_loader, nepochs, classes, savename) 

    test(model, device, loss, test_loader, classes)

    save_model(model,"{}.pth".format(savename)) 
    
    # close tensorboard writer
    writer.close()


