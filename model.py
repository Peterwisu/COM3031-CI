"""

Computational Intelligence Coursework  

Wish, Taimoor, Ionut 


"""

import torch 
import torch.nn as nn 
from torch.nn import MaxPool2d


"""
Main Classifier
"""
class Classifier(nn.Module):
    def __init__(self):
        
        super(Classifier,self).__init__()

        self.conv = nn.Sequential(Conv_block(in_channels=3, out_channels=64, kernel_size=(3,3), do_batchnorm=True),  # B , 64 , 28 , 28

                                  MaxPool2d(3, (2,2)), # B, 64 , 14 ,14 

                                  Conv_block(in_channels=64, out_channels=128, kernel_size=(3,3), do_batchnorm=True), #  B ,128 , 12 , 12

                                  MaxPool2d(3, (2,2)), # B , 128 , 5 ,5 

                                  Conv_block(in_channels=128, out_channels=256 ,kernel_size=(3,3), do_batchnorm=True), # B , 256 , 3 , 3

                                  MaxPool2d(3, (2,2)),  # B , 256 , 1 , 1
                                    
                                  nn.Flatten()
                        
                )


        self.fc = nn.Sequential(Linear_layer(256,128, do_dropout=True, do_batchnorm=True, dropout_rate=0.25),
                                Linear_layer(128,64 , do_dropout=True, do_batchnorm=True, dropout_rate=0.25),
                                Linear_layer(64,10, do_dropout=False, do_batchnorm=False, do_activation=False),
                                # No activation function in last layer since the softmax in already implement in CrossEntropy
                                )


    def forward(self, inputs): 

        cnn_out = self.conv(inputs)
        fc_out = self.fc(cnn_out)

        return fc_out

"""

Linear Layer for Fully Connected Layers 

"""
class Linear_layer(nn.Module):

    def __init__(self, in_features, out_features, do_dropout=False, do_batchnorm=False, do_activation=True, dropout_rate=None):

        super().__init__()
        self.linear =  nn.Linear(in_features, out_features)

        self.do_activation = do_activation        

        self.do_dropout = do_dropout

        self.do_batchnorm = do_batchnorm

        if self.do_activation:

            self.activation = nn.ReLU()

        if self.do_dropout:

            self.drop_layer = nn.Dropout(dropout_rate)

        if self.do_batchnorm:

            self.batchnorm_layer = nn.BatchNorm1d(out_features)

        
    def forward(self, inputs):

        linear_out = self.linear(inputs)

        if self.do_activation:

            linear_out = self.activation(linear_out)

        if self.do_batchnorm:

            linear_out = self.batchnorm_layer(linear_out)

        if self.do_dropout:

            linear_out = self.drop_layer(linear_out)


        return  linear_out


"""

 Convolutional Layer (With batchnorm and dropout)

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
