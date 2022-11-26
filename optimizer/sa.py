# import torch.nn as nn
# import torch 
# import numpy as np
# from torch.optim.optimizer import Optimizer


# class SimulatedAnnealing(Optimizer):
    
#     def __init__(self, params, 
#                  temperature=10000,
#                  k=1,
#                  c=1) :
        
        
#         defaults = dict(temperature=temperature, # temperature
#                         k=k, # Boltzman constant
#                         c=c, # reduction factor
#                         iteration=0)
     
        
        
#         super(SimulatedAnnealing,self).__init__(params,defaults)
        
    
#     def step(self, closure=None):
        
        
        
#         for group in self.param_groups:
            
#             # Clone all the paramter weight
#             cloned = [params.clone() for params  in group['params']]
           
            
            
#             for params in group['params']:
                
#                 random = 
            
            
            
           
        
        
        
        
