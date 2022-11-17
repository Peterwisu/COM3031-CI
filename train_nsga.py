"""

Computational Intelligence Coursework

Name : Wish, Taimoor, Ionut


"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader  as DataLoader
from tqdm import tqdm
from model import Classifier
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from optimizer.nsga2 import NSGA_II
from utils import save_logs , plot_diff ,cm_plot ,roc_plot, plot_pareto_front, save_pareto_front
from extractor import Extractor
from utils import Fitness_Dataset


# softmax activation function
softmax = nn.Softmax(dim=1)

"""

    Train Model

"""
def train(ga, device, loss_criterion, training_set, testing_set,nepochs, classes, cnn , savename):
    
    
    
     
    global global_epochs

    global_epochs = 1
    
    # array of logs
    loss_train_logs = np.array([])
    loss_vali_logs  = np.array([])
    acc_train_logs = np.array([])
    acc_vali_logs = np.array([])
    print('Start training')
    while global_epochs < nepochs+1:

        progress_bar = tqdm((training_set))
        running_acc = 0 
        running_loss = 0  # Total loss in epochs
        running_reg = 0
        iter_inbatch = 0  # Iteration in batch
        
        

        # for  (images, labels) in progress_bar:
            
        #     # move dataset to same device as model
        #     images = images.to(device)
        #     labels = labels.to(device)

        features , labels  = cnn.extract_features(data=progress_bar, device=device)
        loss, reg,  acc = ga.optimize_NN(features,labels)
            

        running_loss +=loss
        running_reg +=reg
        running_acc +=acc
        iter_inbatch +=1
            
            
            #progress_bar.set_description("Training Epochs : {} , Loss : {}".format(global_epochs,(running_loss/iter_inbatch)))

        # get loss in current iteration
        train_acc = running_acc/iter_inbatch
        train_loss = running_loss/iter_inbatch
        train_reg = running_reg
        
        
        all_front , first_front = ga.get_pareto_front()
    
        pareto_plot = plot_pareto_front(all_fronts=all_front,
                                        first_front=first_front)
        
        vali_loss, vali_acc , cm_plot , roc_plot = eval_model(ga, device, loss_criterion, testing_set, classes, cnn)
        
        print("Epoch : {}, TRAIN LOSS : {}, TRAIN REG : {} , TRAIN ACC : {} , VALI LOSS : {} , VALI ACC : {}".format(global_epochs,train_loss, train_reg, train_acc, vali_loss, vali_acc))

        # append in array 
        loss_train_logs = np.append(loss_train_logs, train_loss)
        loss_vali_logs = np.append(loss_vali_logs, vali_loss)
        
        acc_train_logs = np.append(acc_train_logs, train_acc)
        acc_vali_logs = np.append(acc_vali_logs, vali_acc)
        
        # Plot Figure
        loss_figure = plot_diff(loss_train_logs, loss_vali_logs," NSGA Loss")
        acc_figure = plot_diff(acc_train_logs, acc_vali_logs,'NSGA Accuracy') # accuracy different


        # Add logs to tensorboard
        writer.add_scalar("Loss/Train",train_loss,global_epochs)
        writer.add_scalar("Loss/Vali",vali_loss,global_epochs)
        writer.add_scalar("Acc/Train",train_acc,global_epochs)
        writer.add_scalar("Acc/Vali", vali_acc,global_epochs)
        
        writer.add_figure("Plot/loss",loss_figure,global_epochs)  
        writer.add_figure("Plot/acc",acc_figure,global_epochs) 
        writer.add_figure("Plot/cm",cm_plot,global_epochs) 
        writer.add_figure("Plot/roc",roc_plot,global_epochs)    
        writer.add_figure("Plot/pareto_front", pareto_plot, global_epochs)   
        
        # save alls logs to csv files
        save_logs(loss_train_logs, loss_vali_logs, acc_train_logs, acc_vali_logs, save_name='{}.csv'.format(savename))    
        
        
        # increment epoch
        global_epochs +=1

"""

    Evaluate Model

"""

def eval_model(ga, device, loss_criterion, testing_set,classes,cnn):
    
    eval_progress_bar = tqdm(testing_set)
    eval_running_loss = 0 
    eval_running_acc = 0
    eval_iter_inbatch = 0
    
    # store a predict proba and label in array and also its ground truth
    eval_pred_labels = np.array([])
    eval_pred_probas = []
    eval_gt_labels = np.array([])
    with torch.no_grad():
        # for _ , (images, labels) in eval_progress_bar:

        #     images = images.to(device)
        #     labels = labels.to(device)
        #     #  Set model to evaluation stage
        
        features, labels = cnn.extract_features(data=eval_progress_bar, device =device)
            
        emodel = ga.model
        emodel.eval()
            
        predicted = emodel(features)

            # calculate loss
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
    
    return (eval_running_loss/eval_iter_inbatch) , (eval_running_acc/eval_iter_inbatch) ,cm_fig, roc_fig

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
    

    savename ="CIFAR-10_NSGA"

    #  Setup tensorboard
    writer = SummaryWriter("../CI_logs/{}".format(savename))

    device = "cuda" if torch.cuda.is_available else "cpu"

    batch_size = 5000
    
    nepochs = 2

    print("Using  **{}** as a device ".format(device))
    #print("Batch Size : {}".format(batch_size))
    print("Iteration : {} epochs".format(nepochs))
    

    print("Loading dataset ....")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
    # Prepare Dataset
    training_set = torchvision.datasets.CIFAR10(root='./../data', train=True, download=True, transform=transform)

    testing_set  = torchvision.datasets.CIFAR10(root='./../data', train=False, download=True, transform=transform)
    
    training_data , validation_data = torch.utils.data.random_split(training_set, [40000,10000])
    
    

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)
    
    validation_loader  = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    test_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=False, num_workers=2)

    print("Training Dataset: {}".format(len(training_set)))
    print("Testing Dataset: {}".format(len(testing_set)))
    
    # labels of dataset
    classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print("Classes in dataset  : {} ".format(classes))

    
    # Classifier Models

    model = Classifier(size="fc").to(device)

    # Loss function Objective function 
    CrossEntropy = nn.CrossEntropyLoss()
 
    print("Total parameters : {}".format(sum(params.numel() for params in model.parameters())))
    print("Trainable parameters : {}".format(sum(params.numel() for params in model.parameters() if params.requires_grad)))

    parameters_size =sum(params.numel() for params in model.parameters())
    
    # Pretrain featuresd extractor (CNN)
    
    cnn = Extractor('large', './ckpt/gd.pth')
    ga = NSGA_II(CrossEntropy,population_size=20,dimension=parameters_size,numOfBits=50, crossPoint=20)
    
    # put an extracted features in batach form
    features , labels = cnn.extract_features(train_loader, device=device)
    data = zip(features.cpu(), labels.detach().cpu())
    fitness_data = [(x,y) for x,y in data] 
    fitness_loader = DataLoader(Fitness_Dataset(fitness_data), batch_size=40000, shuffle=False , num_workers=2, drop_last=False)
    
    
    print("Initializing poppulation")
    ga.initNN(model=model,device=device, data=fitness_loader)
    print("Finish initializing population")


    #print(np.array(pso.population).shape)
    

    train(ga, device, CrossEntropy, train_loader, test_loader, nepochs, classes,cnn, savename) 
    all_fronts , first_front=  ga.get_pareto_front() 
    save_pareto_front(all_fronts, first_front, save_name="{}.csv".format(savename))
  

    writer.close()

