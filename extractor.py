import torch
import torch.nn as nn
from model.classifier import Classifier
from model.autoencoder import AutoEncoder

class Extractor():

    def __init__(self, name= None, path = None):
        
        self.name = name
        
        self.path = path
        
        
        
        if name is None or path is None:
            
            print("Name of the model or path of is weight is None ")
            exit()
            
        else:
            
            self.model = AutoEncoder()
            
            self.weight = torch.load(path)
            
            
            # Load pretrained weight to a model 
            self.model.load_state_dict(self.weight)
           
            # Empty the FC layer of the model to use one CNN layer as a fearure s extractor 
            self.model.decoder = nn.Identity()
            self.model.act =nn.Identity()
            
            for params in self.model.parameters():
                
                params.requires_grad = False
                
            print("Finish Loading pretrain CNN model")
            print(self.model)
                
    
    
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
        total_features = torch.cat(all_features, dim=0)
        total_labels = torch.cat(all_labels, dim=0)
        
        return total_features, total_labels
        
        
a = Extractor('large','ckpt/AUTo.pth')   
print(a)
        
        
        
        
        
                
                
            
            
            
    
    
            
