import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

class Trainer:
    def __init__(self, model, train_loader, val_loader,test_loader, optimizer_name, learning_rate, num_epochs,weight_decay):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=torch.nn.DataParallel(model,device_ids = [0,1]).to(self.device)
        # self.model = model.to(self.device)

        #Note I'm running the model on Kaggle which supports two gpu .. If you are running on Single device Gpu then uncomment the above mentioned self.model and comment the line where it contains nn.dataparallel
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader=test_loader
        self.num_epochs = num_epochs
        self.weight_decay= weight_decay
        self.learning_rate=learning_rate
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        
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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def validate(self):
        #Set for evalution 
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
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.4f}")
            #Added for removing the cache
            torch.cuda.empty_cache() 

    def confusion_matrix(self,count,Capture_Img=True,plot=True):

        class_names=["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]
    
        confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)

        #Set for evalution 
        self.model.eval()
        correct = 0
        total = 0

        captured_samples = [] 

        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="testing"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
        
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Updating Confusion Matrix
                for true, predict in zip(labels.cpu(), predicted.cpu()):
                  confusion_matrix[true,predict] += 1

                #Capture images if flag is set and count is not reached
                if Capture_Img and len(captured_samples) < count:
                    for i in range(images.size(0)):
                        if len(captured_samples) >= count:
                            break
                        img = images[i].cpu().numpy()
                        true_label = labels[i].item()
                        pred_label = predicted[i].item()
                        captured_samples.append((img, true_label, pred_label))

        Acc=correct/total

        print(f"Test Acc: {Acc*100:.4f}")
        if plot:
            plt.figure(figsize=(10, 7))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap="Greens", xticklabels=class_names, yticklabels=class_names)
            plt.xlabel("Predicted Label")
            plt.ylabel("Actual Label")   
            plt.title("Confusion Matrix")
            plt.show()
        else:
            plt.figure(figsize=(10, 7))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap="Greens", xticklabels=class_names, yticklabels=class_names)
            plt.xlabel("Actual Label")
            plt.ylabel("Predicted Label")
            plt.title("Confusion Matrix")
            Img_name="confusion_matrix.png"
            plt.savefig(Img_name)
            plt.close()
            wandb.log({"confusion_matrix": wandb.Image(Img_name)})

        if Capture_Img and len(captured_samples) > 0:
            rows = (count + 2) // 3  # Calculate rows needed (ceiling division)
            cols = 3
            fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows))
            axes_flat = axes.flatten()

            for i, ax in enumerate(axes_flat):
                ax.axis('off')
                if i < len(captured_samples):
                    img, true, pred = captured_samples[i]
                    # Adjusting mean and std of data
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = img.transpose(1, 2, 0)  
                    img = img * std + mean  # Un_normalize
                    img = np.clip(img, 0, 1)
                    ax.imshow(img)
                    ax.set_title(f"True: {class_names[true]}\nPred: {class_names[pred]}")
            plt.tight_layout()
            plt.show()
    
        return confusion_matrix
            
   