import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleCNN(nn.Module):
    def __init__(self, num_filters, filter_sizes,dropout,activation, batch_norm, input_size,fc_hidden_sizes,num_classes):
        #Initialize the pytorch neural network..
        super(FlexibleCNN, self).__init__()

        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.num_filters = num_filters # list
        self.filter_sizes = filter_sizes #list
        self.input_size=input_size #tuple
        self.fc_hidden_sizes=fc_hidden_sizes #list
        self.num_classes=num_classes
        self.flatten = nn.Flatten() 

        #Call the functions to create Convolution layers and Fully Connected layers
        # self.conv_layers = self.create_conv_layers()
        # self.fc_layers = self.create_fc_layers()

        # Mentioning the defined operations at each convolution layer to sequential..
        self.conv_layers = nn.Sequential(*self.create_conv_layers())
        self.fc_layers = nn.Sequential(*self.create_fc_layers())
             

    def create_conv_layers(self):
        layers = []
        channels = 3  #RGB 

        for idx, (filters, size) in enumerate(zip(self.num_filters, self.filter_sizes)):

            #For each layer we are adding the below there.

            #Number of filters and filter size. 
            layers.append(nn.Conv2d(channels, filters, size,stride=1, padding=0))

            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'elu':
                layers.append(nn.ELU())
            elif self.activation == 'selu':
                layers.append(nn.SELU())
            elif self.activation == 'silu':
                layers.append(nn.SiLU())
            elif self.activation == 'gelu':
                layers.append(nn.GELU())
            else:
                layers.append(nn.Mish())

            # You can add new activation functions If you want (Question1)

            layers.append(nn.MaxPool2d(2))

            if self.batch_norm:
                layers.append(nn.BatchNorm2d(filters))
            
            
            channels = filters  

        return layers

    def create_fc_layers(self):
        # Calculate flattened size from conv layers
        flattened_size = self.output_conv_layers()  

        layers = []
        features = flattened_size

        for hidden_size in self.fc_hidden_sizes:
            # Add linear layer
            layers.append(nn.Linear(features, hidden_size))
            
            # Add activation function
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'elu':
                layers.append(nn.ELU())
            elif self.activation == 'selu':
                layers.append(nn.SELU())
            elif self.activation == 'silu':
                layers.append(nn.SiLU())
            elif self.activation == 'gelu':
                layers.append(nn.GELU())
            else:
                layers.append(nn.Mish())
            
            # Add dropout
            layers.append(nn.Dropout(self.dropout))
            
            features = hidden_size

        # Final output layer (no activation, no batch norm)
        layers.append(nn.Linear(features, self.num_classes))

        # return nn.Sequential(*layers)
        return layers
    


    def output_conv_layers(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *self.input_size)
            conv_output = self.conv_layers(dummy_input)
            return conv_output.view(1, -1).size(1) 


    def forward(self, x):
        x = self.conv_layers(x)
        x= self.flatten(x)
        x = self.fc_layers(x)
        return x
