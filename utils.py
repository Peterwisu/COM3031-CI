"""
This files contain utility function
"""
import numpy as np
import torch 
import torch.nn
import matplotlib.pyplot as plt
import pandas
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize 
import seaborn as sns
import pandas as pd
import os
import torch.nn as nn

# softmax activation function
softmax = nn.Softmax(dim=1)


"""
***************
Fitness_Dataset :  Custom Dataloader for initializing an individual in Genetic Algorithms
***************
"""
class Fitness_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target = self.dataset[index]
        return np.array(example), target

    def __len__(self):
        return len(self.dataset)



"""
*********************
Gaussian Regulariser: 
*********************

******
inputs
******

    model : model that is optimzing

    device : device of that tensor containing a model's parameters 

*******
outputs
*******

    l2_norm : Total sum square of a model weight


"""
def gaussian_regularizer(model,device):

    # # Total sum square weight
    l2_norm = torch.tensor(0.).to(device)

    for params in model.parameters():


        #l2_norm  += torch.norm(params)
        #l2_norm  += params.norm(2)
        l2_norm += params.square().sum()



    return l2_norm

        

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

        ax.plot( fpr[i], tpr[i], label="ROC curve of class {0} (area = {1:0.2f})".format(  classes[i],roc_auc[i]),)


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
    cm = (cm *  10000)
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
def plot_diff(train,evaluate, name):
    
     
    fig, (ax) = plt.subplots(nrows=1,ncols=1)
    ax.plot(train, color='r' , label='Train')
    ax.plot(evaluate, color='b', label='Eval')
    if "loss" in name.lower():
        ax.legend(loc="upper right")
    if "accuracy" in name.lower():
        ax.legend(loc="lower right")
    ax.set_title("{}".format(name))
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
def save_logs(train_loss,eval_loss,train_acc,eval_acc, save_name,train_reg=None,NSGA=False):

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

    if NSGA == True:

        df['train_reg'] = train_reg
    
    #  assign file name to a path
    #savepath = os.path.join(path,'gd_logs.csv') 

    savepath = os.path.join(path, save_name)

    # save file 
    df.to_csv(savepath, index=False)

"""
**********
save_model : Save a checkpoint of a model (its weights) 
**********

******
inputs
******


    model : 

    save_name : 

*******
returns
*******

    None

"""
def save_model(model, save_name):

    # save path 
    path =  "./ckpt/"

    # check if path exist if not create it 
    if not os.path.exists(path):

        os.mkdir(path)
    
    savepath =  os.path.join(path, save_name)

    # save checkpoint
    torch.save(model.state_dict(), savepath)

    print("Save Model Checkpoint at {}".format(savepath))
    
"""
*********
Plot difference between training and evaluation
*********

******
inputs
******
    
    fronts : list of all optimal fronts

*******
returns
*******


    figure : Plot Pareto front figure


"""
def plot_pareto_front(all_fronts, first_front):
    
 

    fig , (ax) = plt.subplots(ncols=1, nrows=1)
    
    ax.scatter(all_fronts[:,0], all_fronts[:,1], c='b', marker='x')
    ax.scatter(first_front[:,0], first_front[:,1], s=150, facecolor='none', edgecolors='g', linewidths=2)
    ax.plot(first_front[:,0], first_front[:,1], c='r', linestyle='dashdot')
    ax.set_title("NSGA II Pareto front")
    ax.set_xlabel('Accuracy')
    ax.set_ylabel("Sum of square weights of a model")
    ax.grid()
    #plt.axis("tight")
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
def save_pareto_front(all_fronts,first_front, save_name):

    # save path
    path = "./logs/pareto_front/"

    # check if path exist if not create it
    if not os.path.exists(path):

        os.mkdir(path)
    
    # create empty dataframe 
    df1 = pd.DataFrame(all_fronts,columns=['loss','weight'])
    
    df2 = pd.DataFrame(first_front, columns=['loss','weight'])
     
    #  assign file name to a path
    #savepath = os.path.join(path,'gd_logs.csv') 

    savepath1 = os.path.join(path,'all-front {}'.format(save_name))
    savepath2 = os.path.join(path,'first-front{}'.format(save_name))

    # save file 
    df1.to_csv(savepath1, index=False)
    df2.to_csv(savepath2, index=False)
    



"""
************
predict_plot
************

******
inputs
******

        model : model for predicting classes of images
        
        data : dataloader containing a images dataset
        
        device : device using in evaluate model ( CPU or CUDA) 

*******
returns
*******

        fig : figure containing a prediction of each images


"""
def predict_plot(model,data,classes,device):
    
    data_iter = iter(data)
    
    images, label = next(data_iter)
   
    # get predict labels 
    model.eval() 
    y_pred = model(images.to(device)) 
    proba = softmax(y_pred).detach().clone().cpu().numpy()
    pred_labels = [np.argmax(i) for i in proba]
    pred_labels = np.array(pred_labels)
    
    images= images.numpy()
    
    
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
    
    for idx in np.arange(5):
        
        ax = fig.add_subplot(1,5, idx+1, xticks=[], yticks=[])
        img = images[idx]/2 + 0.5
        plt.imshow(img.T)
        
        gt = classes[label[idx]]
        y_pred = classes[pred_labels[idx]]
        correct = False
        if gt == y_pred:
            correct =True
        
        result = "Ground Truth: {} \n Prediction: {}".format(gt,y_pred)
        if correct:
            ax.set_title(result, color='green')
        else:
            ax.set_title(result, color='red')
        ax.axis("off")
    plt.axis("off")  
    fig = ax.get_figure()
    plt.close(fig)  
    return fig
    




    
    












    
