"""
Computaional Intelligence Coursework 
"""




import torch
import torch.nn as nn
from torch.nn import MaxPool2d , Upsample



class AutoEncoder(nn.Module):


    def __init__ (self):

        super(AutoEncoder, self).__init__()

        self.encoder = Encoder()

        self.decoder = Decoder()

        
    def forward(self, inputs): 
        

        outputs = self.encoder(inputs)

        outputs = self.decoder(outputs)

    

        
        return outputs




class Encoder(nn.Module):


    def __init__ (self): 
        
        super(Encoder,self).__init__()

        self.encoder = nn.Sequential(Conv_block(in_channels=3, out_channels=64, kernel_size=(1,1), do_batchnorm=True),  # B , 64 , 28 , 28
                                  
                                  
                                  MaxPool2d(3, (2,2)), # B, 64 , 14 ,14 

                                  Conv_block(in_channels=64, out_channels=128, kernel_size=(3,3), do_batchnorm=True), #  B ,128 , 12 , 12
                                

                                  MaxPool2d(3, (2,2)), # B , 128 , 5 ,5 

                                  Conv_block(in_channels=128, out_channels=256,kernel_size=(3,3), do_batchnorm=True), # B , 256 , 3 , 3
                               

                                
                                    
                    
                                    )

    

    def forward(self, inputs):

        outputs = self.encoder(inputs)


        return outputs


class Decoder(nn.Module):

    def __init__ (self):

        super(Decoder,self).__init__()


        self.decoder = nn.Sequential( DeConv_block(in_channels=256, out_channels=64, kernel_size=(3,3)),  # B , 64 , 28 , 28
                                  
                                  
                                  Upsample(scale_factor=2, mode='bilinear'),

                                  DeConv_block(in_channels=64, out_channels=32, kernel_size=(3,3)),

                                  Upsample(scale_factor=2, mode='bilinear'),

                                  DeConv_block(in_channels=32, out_channels=16, kernel_size=(3,3)), #  B ,128 , 12 , 12

#                                  Upsample(scale_factor=2, mode='bilinear'), 

                                  DeConv_block(in_channels=16, out_channels=3,kernel_size=(3,3)), # B , 256 , 3 , 
                               

                                
                                    
                                  )

    
    def forward(self, inputs):

        

        outputs = self.decoder(inputs)

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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, do_dropout=False, do_batchnorm=False, dropout_rate=None):
        
        super().__init__()

        self.conv_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels= out_channels, kernel_size= kernel_size, stride=stride , padding=padding)

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


