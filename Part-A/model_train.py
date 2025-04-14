import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer_name, learning_rate, num_epochs,weight_decay):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # If you are using Kaggle You can use the two which they provided..
        self.model=torch.nn.DataParallel(model,device_ids = [0,1]).to(self.device)

        #If you are running on colab comment the above and uncommnet the below.

        # self.model = model.to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.weight_decay= weight_decay
        self.learning_rate=learning_rate
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        
        # Initialize optimizer
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate,weight_decay=self.weight_decay)
        elif optimizer_name.lower() == 'nadam':
            self.optimizer = optim.NAdam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif optimizer_name.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate,weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self):
        #Initialize the training requirements that we have defined in model
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            #weight update
            self.optimizer.step()
            
            running_loss += loss.item()
            maxi, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def train(self):
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Store history for analysis
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)
            self.train_acc.append(train_acc)
            self.val_acc.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.4f}")
            #As size of dataset is more try to remove cache after each epoch
            torch.cuda.empty_cache() 
            
   