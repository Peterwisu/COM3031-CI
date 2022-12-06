"""

Computational Intelligence Coursework

Name : Wish, Taimoor, Ionut


"""
import torch
import torch.nn as nn
from model.classifier import Classifier



"""

Extractor : class for loading pretrain features extractor model

"""
class Extractor():

    def __init__(self, name= None, path = None):
        
        
        # size name of a model 
        self.name = name
        # path of checkpoint or weight in .pt or .pth 
        self.path = path 
        
         
        if name is None or path is None:
            
            print("Name of the model or path of is weight is None ")
            exit()
            
        else:
            
            # Create classifier      
            self.model = Classifier(size=name)
            # load weight 
            self.weight = torch.load(path) 
            # Load pretrained weight to a model 
            self.model.load_state_dict(self.weight) 
            # Empty the FC layer of the model to use one CNN layer as a fearure  extractor 
            self.model.fc = nn.Identity()
            
            # remove gradient calulation from weight 
            for params in self.model.parameters():
                
                params.requires_grad = False
                
            print("Finish Loading pretrain CNN model")
            print(self.model)
                
    
    """
    ****************
    extract_features :  Extract features from dataset
    ****************
    
    
    ******** 
    inputs : 
    ******** 
    
            data : dataloader contain data to extract features
            
            device : device using in evaluate model ( CPU or CUDA) 
    
    ********* 
    returns :
    ********* 

            all_features : all features extracted from dataset
            
            all_labels : all ground truth labels of dataset
    
    
    """ 
    def  extract_features(self,data, device):

        all_features = []
        all_labels = []
        for (x , y) in data:
            
            x = x.to(device)
            y = y.to(device)
            
            
            self.model = self.model.to(device)
            
            with torch.no_grad():
                
                self.model.eval()
                
                features = self.model(x).detach()
                
                all_features.append(features)
                all_labels.append(y)
        
        # concatenate all tensor in a list to one tensor  (total number of dataset , value)        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        return all_features, all_labels
        
        

        
        
        
        
        
                
                
            
            
            
    
    
            
