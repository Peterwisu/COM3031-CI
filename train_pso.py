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
from pso import PSO
from torch.nn import DataParallel
# import utility function 
from utils import save_logs , plot_diff ,cm_plot ,roc_plot


# softmax activation function
softmax = nn.Softmax(dim=1)

"""

    Train Model

"""
def train(pso, device, loss_criterion, training_set, validation_set,nepochs, classes):
    
    global global_epochs

    global_epochs = 1
    
     # array of logs
    loss_train_logs = np.array([])
    loss_vali_logs  = np.array([])
    acc_train_logs = np.array([])
    acc_vali_logs = np.array([])


    while global_epochs < nepochs+1:

        progress_bar = tqdm((training_set))
        running_acc = 0
        running_loss = 0  # Total loss in epochs
        iter_inbatch = 0  # Iteration in batch
        
        

        for  (images, labels) in progress_bar:
            
            # move dataset to same device as model
            images = images.to(device)
            labels = labels.to(device)

            loss, acc = pso.optimize_NN(global_epochs,nepochs,images,labels)
            running_acc +=acc
            running_loss +=loss
            iter_inbatch +=1
            
            
            #progress_bar.set_description("Training Epochs : {} , Loss : {}".format(global_epochs,(running_loss/iter_inbatch)))

        # get loss in current iteration
        train_loss = running_loss/iter_inbatch
        train_acc = running_acc/iter_inbatch
        vali_loss , vali_acc ,cm_plot , roc_plot = eval_model(pso, device, loss_criterion, validation_set, classes)
        
        print("Epoch : {} , TRAIN LOSS : {} , TRAIN ACC {} , VALI LOSS : {} , VAL_ACC : {} ".format(global_epochs,train_loss, train_acc, vali_loss, vali_acc))

         # append loss and accuracy in current epochs in array
        loss_train_logs = np.append(loss_train_logs, train_loss)
        loss_vali_logs = np.append(loss_vali_logs, vali_loss)
        acc_train_logs = np.append(acc_train_logs, train_acc)
        acc_vali_logs = np.append(acc_vali_logs,vali_acc)
        
        # Plot Figure
        loss_figure = plot_diff(loss_train_logs, loss_vali_logs,'PSO Loss') # loss different
        acc_figure = plot_diff(acc_train_logs, acc_vali_logs,'PSO Accuracy') # accuracy different


        # Add logs to tensorboard
        # add scalar values
        writer.add_scalar("Loss/Train",train_loss,global_epochs)
        writer.add_scalar("Loss/Vali",vali_loss,global_epochs)
        writer.add_scalar("Acc/Train",train_acc,global_epochs)
        writer.add_scalar("Acc/Vali", vali_acc, global_epochs)
        
        # add figures
        writer.add_figure("Plot/loss",loss_figure,global_epochs)
        writer.add_figure("Plot/acc",acc_figure,global_epochs)
        writer.add_figure("Plot/cm",cm_plot, global_epochs)
        writer.add_figure("Plot/roc",roc_plot, global_epochs)


        # save alls logs to csv files
        save_logs(loss_train_logs, loss_vali_logs, acc_train_logs, acc_vali_logs, save_name='pso_logs.csv')    
        

        # increment epoch
        global_epochs +=1

"""

    Evaluate Model

"""

def eval_model(pso, device, loss_criterion, testing_set, classes):
    
    eval_progress_bar = tqdm((testing_set))
    eval_running_loss = 0 
    eval_running_acc = 0
    eval_iter_inbatch = 0
    
    
    
    # store a predict proba and label in array and also its ground truth
    eval_pred_labels = np.array([])
    eval_pred_probas = []
    eval_gt_labels = np.array([])
    emodel = pso.model
    for (images, labels) in eval_progress_bar:
        images = images.to(device)
        labels = labels.to(device)
            #  Set model to evaluation stage
            
        
        emodel.eval()
            
            
        predicted = emodel(images)

            # calculate loss
        
            
        #eval_loss = loss_criterion(predicted,labels)
        
        eval_loss, eval_acc, pred_label, gt_label, pred_proba = objective(predicted,labels,loss_criterion)
        
     
        eval_pred_labels = np.append(eval_pred_labels , pred_label)

        eval_pred_probas.append(pred_proba)

        eval_gt_labels = np.append(eval_gt_labels, gt_label)

        eval_running_acc +=eval_acc
        eval_running_loss += eval_loss.item()

        eval_iter_inbatch +=1

            #eval_progress_bar.set_description("Evaluation Epochs : {} , Loss : {}".format(global_epochs, (eval_running_loss/eval_iter_inbatch)))

    # concatenate probabilites array in to a shape  of (Number of image, prob of n_classes) 
    # this contains probabilites predict for each classes for each images
    eval_pred_probas = np.concatenate(eval_pred_probas,axis=0)

    # Plot ROC curve
    roc_fig = roc_plot(eval_pred_probas,eval_gt_labels, classes)
    # Plot Confusion matrix
    cm_fig = cm_plot(eval_pred_labels, eval_gt_labels, classes) 
    
    # for name, param in emodel.named_parameters():
    #     print(name, torch.isnan(param.grad))
    return (eval_running_loss/eval_iter_inbatch) , (eval_running_acc/eval_iter_inbatch) ,cm_fig, roc_fig



"""
******************
Objective function :  Calculate a loss and accuracy of a model from a predict value and its ground truth
******************

******
inputs
******

    
    predicted : predicted(output) generated from a models

    labels : ground truth of the data

    loss_criterion :  Loss function or objective function


*******
returns
*******
    
    loss : loss of the objective funciton

    accuracy :  accuracy of output of the model 

    pred_labels : labels predicted from model

    gt_labels : ground truth of the data

    proba : predicted probabilites of each labels(classes)

""" 


def objective(predicted, labels, loss_criterion):
    

    # calcuate an objective loss 
    loss = loss_criterion(softmax(predicted), labels)
    
    # get probabilites of each label
    proba = softmax(predicted).cpu().detach().numpy()
    # get predicted label
    pred_labels = [np.argmax(i) for i in proba]
    pred_labels = np.array(pred_labels)

    
    # Calculated accuracy 
    correct = 0
    accuracy = 0
    
    # allocate label to cpu
    gt_labels = labels.cpu().detach().numpy()

    for p ,g in zip(pred_labels,gt_labels):

        if p == g:
            correct+=1

    accuracy = 100 * (correct/len(gt_labels))
    
    # return (loss, accuracy,  predicted labels, ground truth labels, predicted probabilites)
    return  loss, accuracy, pred_labels, gt_labels, proba

    

if __name__ == "__main__":
    

    savename ="CIFAR-10_PSO"

    #  Setup tensorboard
    writer = SummaryWriter("../CI_logs/{}".format(savename))

    device = "cuda" if torch.cuda.is_available else "cpu"

    batch_size = 10000
    
    nepochs = 100

    print("Using  **{}** as a device ".format(device))
    print("Batch Size : {}".format(batch_size))
    print("Iteration : {} epochs".format(nepochs))
    

    print("Loading dataset ....")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
    # Prepare Dataset
    training_set = torchvision.datasets.CIFAR10(root='./../data', train=True, download=True, transform=transform)

    testing_set  = torchvision.datasets.CIFAR10(root='./../data', train=False, download=True, transform=transform)

    

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    test_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    print("Training Dataset: {}".format(len(training_set)))
    print("Testing Dataset: {}".format(len(testing_set)))
    
    # labels of dataset
    classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print("Classes in dataset  : {} ".format(classes))

    
    # Classifier Models

    
    
    model = Classifier().to(device)

    if device == "cuda" and torch.cuda.device_count() > 1 :

        model = DataParallel(model) 
        model.to(device)
        print(" Usining {} gpu for training".format(torch.cuda.device_count()))
        batch_size = batch_size * torch.cuda.device_count()


    
    # Loss function Objective function 
    CrossEntropy = nn.CrossEntropyLoss()
 
    print("Total parameters : {}".format(sum(params.numel() for params in model.parameters())))
    print("Trainable parameters : {}".format(sum(params.numel() for params in model.parameters() if params.requires_grad)))

    parameters_size =sum(params.numel() for params in model.parameters() if params.requires_grad)
    
    
   
    # for i in model.parameters():

    #     print(i)
    
    

    pso = PSO(CrossEntropy,25, parameters_size, pso_type='global', num_neighbours=5)
    pso.initNN(model=model,device=device)
    print(np.array(pso.population).shape)
    print(pso.pso_type)

    train(pso, device, CrossEntropy, train_loader, test_loader, nepochs, classes) 

    writer.close()

