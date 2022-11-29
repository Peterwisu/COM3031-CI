"""
Computaional Intelligence Coursework 
"""


import torch
import torch.nn as nn
from torch.nn import MaxPool2d , Upsample



class AutoEncoder(nn.Module):


    def __init__ (self):

        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
                                # Conv_block(in_channels=3, out_channels=32, kernel_size=(1,1), do_batchnorm=True),  # B , 64 , 28 , 28
                                  
                                  
                                #   MaxPool2d(3, (2,2)), # B, 64 , 14 ,14 

                                #   Conv_block(in_channels=32, out_channels=64, kernel_size=(3,3), do_batchnorm=True), #  B ,128 , 12 , 12
                                

                                #   MaxPool2d(3, (2,2)), # B , 128 , 5 ,5 

                                #   Conv_block(in_channels=64, out_channels=128,kernel_size=(3,3), do_batchnorm=True), # B , 256 , 3 , 3
                               
                                #   MaxPool2d(2, (2,2)), # B , 128 , 5 ,5  
                                Conv_block(3,16,3,padding=1)   ,
                                MaxPool2d(2,2),
                                Conv_block(16,4,3,padding=1) ,
                                MaxPool2d(2,2)
                                                   
                    
                                    )

        self.decoder = nn.Sequential( 
                                #   DeConv_block(in_channels=128, out_channels=64, kernel_size=(3,3)),  # B , 64 , 28 , 28
                                  
                                  
                                #   Upsample(scale_factor=2, mode='bilinear'),

                                #   DeConv_block(in_channels=64, out_channels=32, kernel_size=(5,5)),

                                #   Upsample(scale_factor=2, mode='bilinear'),

                                #   DeConv_block(in_channels=32, out_channels=16, kernel_size=(5,5)), #  B ,128 , 12 , 12

                                  

                                #   DeConv_block(in_channels=16, out_channels=3,kernel_size=(5,5)), # B , 256 , 3 , 
                                DeConv_block(4,16,2,stride=2),
                                DeConv_block(16,3,2,stride=2)
                                  )
        
        self.act = nn.Sigmoid()


        
    def forward(self, inputs): 
        
        # print(inputs.shape)
        outputs = self.encoder(inputs)
        #print(outputs.shape)
        #print(outputs.shape)
        outputs = self.decoder(outputs)
        #print(outputs.shape)
        # exit()

        #print(outputs.shape)
        #print(outputs[0])
        #exit()
    
        outputs = self.act(outputs)
        
        return outputs


"""

 Convolutional Layer for AutoEncoder  (With batchnorm and dropout)

"""
class Conv_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, do_dropout=False, do_batchnorm=False, dropout_rate=None):
        
        super().__init__()

        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size= kernel_size, stride=stride , padding=padding)

        self.activation = nn.ReLU()
        
        self.do_dropout = do_dropout

        self.do_batchnorm = do_batchnorm

        if self.do_dropout:

            self.drop_layer = nn.Dropout2d(dropout_rate)

        if self.do_batchnorm:

            self.batchnorm_layer = nn.BatchNorm2d(out_channels)
            

    def forward(self, inputs):

        cnn_out = self.conv_layer(inputs)

        cnn_out = self.activation(cnn_out)

        if self.do_batchnorm:

            cnn_out = self.batchnorm_layer(cnn_out)

        if self.do_dropout:

            cnn_out = self.dropout_layer(cnn_out)
        
        return cnn_out

"""

 DeConvolutional Layer (With batchnorm and dropout)

"""
class DeConv_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,do_act=False, do_dropout=False, do_batchnorm=False, dropout_rate=None):
        
        super().__init__()

        self.conv_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels= out_channels, kernel_size= kernel_size, stride=stride , padding=padding)

        self.activation = nn.ReLU()
        
        self.do_dropout = do_dropout

        self.do_batchnorm = do_batchnorm
        
        self.do_act = do_act

        if self.do_dropout:

            self.drop_layer = nn.Dropout2d(dropout_rate)

        if self.do_batchnorm:

            self.batchnorm_layer = nn.BatchNorm2d(out_channels)
            

    def forward(self, inputs):

    

        cnn_out = self.conv_layer(inputs)

        if self.do_act:
            
            cnn_out = self.activation(cnn_out)

    
        if self.do_batchnorm:

            cnn_out = self.batchnorm_layer(cnn_out)

        if self.do_dropout:

            cnn_out = self.dropout_layer(cnn_out)
        
        return cnn_out


