"""

Computational Intelligence Coursework

Name : Wish, Taimoor, Ionut


"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader  as DataLoader 
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import torch.optim as optim
from model import Classifier
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt 
import numpy as np
from ga import GA

"""

    Train Model

"""
def train(ga, device, loss_criterion, training_set, testing_set,nepochs):
    
    
    
     
    global global_epochs

    global_epochs = 1
    
    # array of logs
    train_logs = np.array([])
    eval_logs  = np.array([])

    while global_epochs < nepochs+1:

        progress_bar = enumerate(training_set)
        
        running_loss = 0  # Total loss in epochs
        iter_inbatch = 0  # Iteration in batch
        
        

        for _ , (images, labels) in progress_bar:
            
            # move dataset to same device as model
            images = images.to(device)
            labels = labels.to(device)

            loss = ga.optimize_NN(images,labels)

            running_loss +=loss
            iter_inbatch +=1
            
            
            #progress_bar.set_description("Training Epochs : {} , Loss : {}".format(global_epochs,(running_loss/iter_inbatch)))

        # get loss in current iteration
        train_loss = running_loss/iter_inbatch
        eval_loss = eval_model(ga, device, loss_criterion, testing_set)
        
        print("Epoch : {}, TRAIN LOSS : {}, EVAL LOSS : {} ".format(global_epochs,train_loss, eval_loss))

        # append in array 
        train_logs = np.append(train_logs, train_loss)
        eval_logs = np.append(eval_logs, eval_loss)
        
        # Plot Figure
        figure = plot_diff(train_logs, eval_logs)

        # Add logs to tensorboard
        writer.add_scalar("Loss/Train",train_loss,global_epochs)
        writer.add_scalar("Loss/Eval",eval_loss,global_epochs)
        writer.add_figure("Loss/plot",figure,global_epochs)        
        

        # increment epoch
        global_epochs +=1

"""

    Evaluate Model

"""

def eval_model(ga, device, loss_criterion, testing_set):
    
    eval_progress_bar = enumerate(testing_set)
    eval_running_loss = 0 
    eval_iter_inbatch = 0
    with torch.no_grad():
        for _ , (images, labels) in eval_progress_bar:

            images = images.to(device)
            labels = labels.to(device)
            #  Set model to evaluation stage
            
            emodel = ga.model
            emodel.eval()
            
            predicted = emodel(images)

            # calculate loss
            eval_loss = loss_criterion(predicted,labels)

            eval_running_loss += eval_loss.item()

            eval_iter_inbatch +=1

            #eval_progress_bar.set_description("Evaluation Epochs : {} , Loss : {}".format(global_epochs, (eval_running_loss/eval_iter_inbatch)))

    
    return eval_running_loss/eval_iter_inbatch

"""
Plot loss difference
"""
def plot_diff(train,evaluate):
    
    fig, (ax) = plt.subplots(nrows=1,ncols=1)
    ax.plot(train, color='r' , label='Train')
    ax.plot(evaluate, color='b', label='Eval')
    ax.legend(loc="upper right")
    ax.set_title("Loss")
    figure = ax.get_figure()
    plt.close(fig)
    
    return figure
    

if __name__ == "__main__":
    

    savename ="CIFAR-10_SGD"

    #  Setup tensorboard
    writer = SummaryWriter("../CI_logs/{}".format(savename))

    device = "cuda" if torch.cuda.is_available else "cpu"

    #batch_size = 2
    
    nepochs = 100

    print("Using  **{}** as a device ".format(device))
    #print("Batch Size : {}".format(batch_size))
    print("Iteration : {} epochs".format(nepochs))
    

    print("Loading dataset ....")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
    # Prepare Dataset
    training_set = torchvision.datasets.CIFAR10(root='./../data', train=True, download=True, transform=transform)

    testing_set  = torchvision.datasets.CIFAR10(root='./../data', train=False, download=True, transform=transform)

    train_loader = DataLoader(training_set, batch_size=len(training_set), shuffle=True, num_workers=2)

    test_loader = DataLoader(testing_set, batch_size=len(testing_set), shuffle=False, num_workers=2)

    print("Training Dataset: {}".format(len(training_set)))
    print("Testing Dataset: {}".format(len(testing_set)))
    
    # labels of dataset
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print("Classes in dataset  : {} ".format(classes))

    
    # Classifier Models

    model = Classifier().to(device)

    # Loss function Objective function 
    CrossEntropy = nn.CrossEntropyLoss()
 
    print("Total parameters : {}".format(sum(params.numel() for params in model.parameters())))
    print("Trainable parameters : {}".format(sum(params.numel() for params in model.parameters() if params.requires_grad)))

    parameters_size =sum(params.numel() for params in model.parameters())
    
    """
   
    for i in model.parameters():

        print(len(i))
    """ 

    ga = GA(CrossEntropy,population_size=10,dimension=parameters_size,numOfBits=100)
    ga.initNN(model=model,device=device, data=train_loader)


    #print(np.array(pso.population).shape)
    

    train(ga, device, CrossEntropy, train_loader, test_loader, nepochs) 

    writer.close()

