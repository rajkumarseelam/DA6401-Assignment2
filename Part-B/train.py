from model import FinetuneCNN
from data_loader import DataLoaderHelper
from model_trainer import Trainer
import argparse
import wandb
from pathlib import Path

if __name__ == "__main__":

    # Added the default parameters of my best config
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", "-we",help = "Wandb Entity rollno", default="cs24m042-iit-madras-foundation")
    parser.add_argument("--wandb_project", "-wp",help="Project name", default="DA6401-Assignment-2")
    parser.add_argument("--num_epochs","-e", help= "Number of epochs to train CNN", type= int, default=10)
    parser.add_argument("--batch_size","-b",help="Batch size used to train CNN", type =int, default=32)
    parser.add_argument("--augmentation", "-au", default="false", choices=["true", "false"])
    parser.add_argument("--learning_rate","-lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay","-w_d", default=0,type=float)
    parser.add_argument("--base_dir", "-br",type=str,help='Base directory containing train/val folders', default="inaturalist_12K")
    parser.add_argument("--optimizer_name","-o",help="batch size is used to train neural network", default= "nadam", choices=['nadam', 'adam'])

    args = parser.parse_args()

    train_directory=Path(args.base_dir) / 'train'
    test_directory=Path(args.base_dir) / 'val'

    augmentation=False
    if(args.augmentation =="true"):
        augmentation=True

    num_classes=10 


    wandb.login()
    run_name = "optimizer {}   batch_size {} augmentation {} weight_decay {} learning_rate {}  ".format(
            args.optimizer_name,
            args.batch_size,
            args.augmentation,
            args.weight_decay,
            args.learning_rate
          )
    wandb.init(project=args.wandb_project,entity=args.wandb_entity,name=run_name)

    #Loading the data
    input_val=(224,224) # standard Resnet size
    data_loader = DataLoaderHelper(train_directory,test_directory,input_val,args.batch_size,augmentation)

    train_loader, val_loader,test_loader = data_loader.get_dataloaders()

    # Loading the pretrained Resnet Model
    model=FinetuneCNN()
    model.Freezelayers()
    
    #Intializing the train model
    trainer = Trainer(model.fine_tune_model,train_loader,val_loader,test_loader,args.optimizer_name,args.learning_rate,args.num_epochs,args.weight_decay)

    #Training the model
    trainer.train()

    #Plot the confusion matrix and 3*10 images with their predicted and Actual value
    trainer.confusion_matrix(plot=False) #Optional
    #Note for the If you send plot=False parameter the plots will be logged into the wandb.


    # Log the data to wandb.
    for epoch in range(args.num_epochs):
            wandb.log({
                'train_accuracy': trainer.train_acc_history[epoch]*100,
                'train_loss': trainer.train_loss_history[epoch],
                'val_accuracy': trainer.val_acc_history[epoch]*100,
                'val_loss': trainer.val_loss_history[epoch],
                'epoch' : epoch
            })
    wandb.finish()




