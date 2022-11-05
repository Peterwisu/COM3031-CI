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
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize 
import seaborn as sns
import pandas as pd
import os


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
def train(model, device, loss_criterion, optimizer, training_set, testing_set,nepochs, classes):
    
    global global_epochs

    global_epochs = 1
    
    # array of logs
    loss_train_logs = np.array([])
    loss_eval_logs  = np.array([])
    acc_train_logs = np.array([])
    acc_eval_logs = np.array([])

    while global_epochs < nepochs+1:

        progress_bar = tqdm(enumerate(training_set))
        
        running_loss = 0  # Total loss in epochs
        running_acc = 0
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

            # calculate loss 
            loss,acc , _ ,_ ,_= objective(predicted, labels, loss_criterion)
 

            # backprop and optmizer update weight
            loss.backward()
            optimizer.step()
            running_acc +=acc 
            running_loss +=loss.item()
            iter_inbatch +=1
            
            
            progress_bar.set_description("Training Epochs : {} , Loss : {} , Acc : {} ".format(global_epochs,(running_loss/iter_inbatch), (running_acc/iter_inbatch)))

        # get loss in current epoch
        train_loss = running_loss/iter_inbatch # calculate a mean loss (total loss / iteration)
        train_acc = running_acc/iter_inbatch

        # calculate evaluation loss and accuracy
        # Plot confusion matrix and ROC curve with evaluation results
        eval_loss , eval_acc ,cm_plot , roc_plot = eval_model(model, device, loss_criterion, testing_set, classes)

        # append loss and accuracy in current epochs in array
        loss_train_logs = np.append(loss_train_logs, train_loss)
        loss_eval_logs = np.append(loss_eval_logs, eval_loss)
        acc_train_logs = np.append(acc_train_logs, train_acc)
        acc_eval_logs = np.append(acc_eval_logs,eval_acc)
        
        # Plot Figure
        loss_figure = plot_diff(loss_train_logs, loss_eval_logs) # loss different
        acc_figure = plot_diff(acc_train_logs, acc_eval_logs) # accuracy different


        # Add logs to tensorboard
        # add scalar values
        writer.add_scalar("Loss/Train",train_loss,global_epochs)
        writer.add_scalar("Loss/Eval",eval_loss,global_epochs)
        writer.add_scalar("Acc/Train",train_acc,global_epochs)
        writer.add_scalar("Acc/Eval", eval_acc, global_epochs)
        
        # add figures
        writer.add_figure("Plot/loss",loss_figure,global_epochs)
        writer.add_figure("Plot/acc",acc_figure,global_epochs)
        writer.add_figure("Plot/cm",cm_plot, global_epochs)
        writer.add_figure("Plot/roc",roc_plot, global_epochs)


        # save alls logs to csv files
        save_logs(loss_train_logs, loss_eval_logs, acc_train_logs, acc_eval_logs)
        

        # increment epoch
        global_epochs +=1



"""
**********
eval_model :  Evaluation stage
**********

******
inputs
******

    model : Evaluation model

    device : device using in evaluate model ( CPU or CUDA) 

    loss_criterion : Objective funciton 

    testing_set : testing dataset use for evaluation 

    classes : array containing name of classes


*******
returns
*******

    evaluation_loss : Loss during evaluation stage 
    
    evaluation_accuracy : Accuracy during evaluation stage

    cm_fig  : Confusion Matrix figure

    roc_fig : ROC  curve  figure

"""

def eval_model(model, device, loss_criterion, testing_set, classes):
    
    eval_progress_bar = tqdm(enumerate(testing_set))


    eval_running_acc = 0
    eval_running_loss = 0 
    eval_iter_inbatch = 0

    # store a predict proba and label in array and also its ground truth
    eval_pred_labels = np.array([])
    eval_pred_probas = []
    eval_gt_labels = np.array([])
    
 
    for _ , (images, labels) in eval_progress_bar:

        images = images.to(device)
        labels = labels.to(device)
        #  Set model to evaluation stage
        model.eval()

        predicted = model(images)

        # calculate loss
        eval_loss, eval_acc, pred_label, gt_label, pred_proba = objective(predicted,labels,loss_criterion)

        eval_pred_labels = np.append(eval_pred_labels , pred_label)

        eval_pred_probas.append(pred_proba)

        eval_gt_labels = np.append(eval_gt_labels, gt_label)

        eval_running_loss += eval_loss.item()
        eval_running_acc += eval_acc
        eval_iter_inbatch +=1

        eval_progress_bar.set_description("Evaluation Epochs : {} , Loss : {}, Accuracy : {}".format(global_epochs, (eval_running_loss/eval_iter_inbatch),(eval_running_acc/eval_iter_inbatch)))

    # concatenate probabilites array in to a shape  of (Number of image, prob of n_classes) 
    # this contains probabilites predict for each classes for each images
    eval_pred_probas = np.concatenate(eval_pred_probas,axis=0)

    # Plot ROC curve
    roc_fig = roc_plot(eval_pred_probas,eval_gt_labels, classes)
    # Plot Confusion matrix
    cm_fig = cm_plot(eval_pred_labels, eval_gt_labels, classes)

    
    # return  ( evaluation loss , evaluation accuracy ,  confusion matrix figure, roc curve figure)
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
    loss = loss_criterion(predicted, labels)
    
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


"""
**************
Plot roc curve  :  Plot a ROC curve figure  (only for multi-label classifcation ROC curve)
**************


******
inputs
******

    pred_probas : predicted probabilities of each labels(classes)

    gt_label : ground truth of a labels

    classes : array or list conatining a class name eg. [ 'dog', 'cat', 'cow' ] 


*******
returns
*******

    plot_roc : ROC curve figure
"""

def roc_plot(pred_probas,gt_label, classes):
    

    # dictionart for False positive rate, True positive rate, and Area Under the curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    #  number of classes
    n_classes = len(classes)
    # generate numnber  from 0 to n classes
    n_labels = np.arange(0, n_classes)
     
    #  convert labels to one hot encoding 
    labels = label_binarize(gt_label, classes=n_labels)

    # calculate roc curve and area under curve for each class
    for i in range(n_classes):

        # calculate roc curve
        fpr[i] , tpr[i] , _  = roc_curve(labels[:,i],pred_probas[:,i])
        # calcuate area under the curve
        roc_auc[i] = auc(fpr[i], tpr[i])


    # calculate the micro and marco
    fpr["micro"] , tpr["micro"], _ = roc_curve(labels.ravel(), pred_probas.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([ fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):

        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes


    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    # plot fig 
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(9,7)
    # micro line
    ax.plot(fpr["micro"], tpr["micro"], label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]), color='deeppink', linestyle=":", linewidth=4)
    # macro line 
    ax.plot(fpr["macro"], tpr["macro"], label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]), color="navy", linestyle=":", linewidth = 4) 
    # curve line for each classes
    for i in range(n_classes):

        ax.plot( fpr[i], tpr[i], label="ROC curve of class {0} (area = {1:0.2f})".format(i,roc_auc[i]),)


    ax.plot([0,1],[0,1], "k--", lw=2)
    ax.legend(loc="lower right")
    ax.set_title("Receiving Operating Characteristic (ROC) Curve")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plot_roc = ax.get_figure()
    plt.close(fig)
    
    return plot_roc

"""
*********************
Plot confusion matrix
*********************

******
inputs
******
    
    pred_labels : predicted labels from model

    gt_labels : ground truth labels of a data

    class_name : array or list conatining a class name eg. [ 'dog', 'cat', 'cow' ]

*******
returns
*******

   plot_cm  :  Confusion matrix figure 


"""

def cm_plot(pred_labels, gt_labels, class_name):
    
    # calculate a matrix
    cm = confusion_matrix(gt_labels, pred_labels,normalize="all")

    cm_df = pd.DataFrame(cm, index =class_name, columns = class_name)

    # plot fig

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(9,7)
    fig.tight_layout()
    ax.set_title("Confusion Matrix for CIFAR 10 datasets")
    plot_cm  = sns.heatmap(cm_df, annot=True).get_figure()
    plt.close(fig)

    return plot_cm

"""
*********
Plot difference between training and evaluation
*********

******
inputs
******
    
    train :  array containing a training logs value (accuracy or loss)

    evaluate :  array containg a evaluation log value 

*******
returns
*******


    figure : Plot difference figure


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


"""
*********
save_logs : Saving a logs into csv file
*********

******
inputs
******
    

    train_loss :  array of training logs
    
    eval_loss :  array  of evaluation logs

    train_acc : array of training accuracy

    eval_acc : array of evaluation accuracy 


*******
returns
*******


    None 

"""
def save_logs(train_loss,eval_loss,train_acc,eval_acc):

    # save path    
    path = "./logs/"

    # check if path exist if not create it
    if not os.path.exists(path):

        os.mkdir(path)
    
    # create empty dataframe 
    df = pd.DataFrame()
   
    # assign the logs to dataframe
    df['train_loss'] = train_loss 
    df['eval_loss'] = eval_loss
    df['train_acc'] = train_acc
    df['eval_acc'] = eval_acc
    
    #  assign file name to a path
    savepath = os.path.join(path,'gd_logs.csv') 
    # save file 
    df.to_csv(savepath, index=False)
    





"""
Main function  
"""

if __name__ == "__main__":
    

    savename ="CIFAR-10_SGD"

    #  Setup tensorboard
    writer = SummaryWriter("../CI_logs/{}".format(savename))

    device = "cuda" if torch.cuda.is_available else "cpu"

    batch_size = 32

    nepochs = 100

    print("Using  **{}** as a device ".format(device))
    print("Batch Size : {}".format(batch_size))
    print("Iteration : {} epochs".format(nepochs))
    

    print("Loading dataset ....")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
    # Prepare Dataset
    training_set = torchvision.datasets.CIFAR10(root='./../data', train=True, download=True, transform=transform)

    testing_set  = torchvision.datasets.CIFAR10(root='./../data', train=False, download=True, transform=transform)

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True, num_workers=2)

    print("Training Dataset: {}".format(len(training_set)))
    print("Testing Dataset: {}".format(len(testing_set)))
    
    # labels of dataset
    classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print("Classes in dataset  : {} ".format(classes))

    
    # Classifier Models
    model = Classifier().to(device)

    # Loss function Objective function 
    CrossEntropy = nn.CrossEntropyLoss()

    # Optimizer 
    optimizer = optim.SGD([params  for params in model.parameters() if params.requires_grad], lr=0.001)
    
    print("Total parameters : {}".format(sum(params.numel() for params in model.parameters())))
    print("Trainable parameters : {}".format(sum(params.numel() for params in model.parameters() if params.requires_grad)))


    train(model, device, CrossEntropy, optimizer, train_loader, test_loader, nepochs, classes) 
    
    # close tensorboard writer
    writer.close()

