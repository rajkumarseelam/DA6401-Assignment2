import torch
import torch.nn as nn
from torchvision import models

class FinetuneCNN():
    def __init__(self,pt=True):
      # # Loading pretrained ResNet50 model
      self.fine_tune_model=models.resnet50(pretrained=pt)

    def Freezelayers(self):
        # Freezing all the layers
        for i in self.fine_tune_model.parameters():
            i.requires_grad=False

        # Fetching nodes at last layer
        features=self.fine_tune_model.fc.in_features
        # Mapping the last to our required 10 classes
        self.fine_tune_model.fc= nn.Linear(features, 10)
            
