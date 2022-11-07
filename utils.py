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
def save_logs(train_loss,eval_loss,train_acc,eval_acc, save_name):

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
    #savepath = os.path.join(path,'gd_logs.csv') 

    savepath = os.path.join(path, save_name)

    # save file 
    df.to_csv(savepath, index=False)
    