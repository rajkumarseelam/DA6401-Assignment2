from model import FlexibleCNN
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
    parser.add_argument("--batch_size","-b",help="Batch size used to train CNN", type =int, default=64)
    parser.add_argument("--augmentation", "-au", default="false", choices=["true", "false"])
    parser.add_argument("--learning_rate","-lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay","-w_d", default=0.0005,type=float)
    parser.add_argument("--base_dir", "-br",type=str,help='Base directory containing train/val folders', default="inaturalist_12K")
    parser.add_argument("--activation", "-ac",choices=['relu', 'elu', 'selu', 'silu', 'gelu','mish'], default="silu")
    parser.add_argument("--num_filters", "-nf", nargs=5, type=int, default=[128, 128, 64, 64, 32])
    parser.add_argument("--filter_sizes", "-fs", nargs=5, type=int, default=[5,5,5,5,5])
    parser.add_argument("--input_dim", "-in", type=int, default=224)
    parser.add_argument("--batch_norm", "-bn", default="true", choices=["true", "false"])
    parser.add_argument("--optimizer_name","-o",help="batch size is used to train neural network", default= "rmsprop", choices=['nadam', 'adam', 'rmsprop'])
    parser.add_argument("--hidden_size", "-dl", nargs=1, type=int, default=[512])
    parser.add_argument("--dropout", "-dp", default=0.4, type=float)

    args = parser.parse_args()

    train_directory=Path(args.base_dir) / 'train'
    test_directory=Path(args.base_dir) / 'val'

    augmentation=False
    if(args.augmentation =="true"):
        augmentation=True

    batch_norm= False
    if(args.batch_norm =="true"):
        batch_norm=True

    num_classes=10 


    wandb.login()
    run_name="optimizer {} activation {} num_filters {} dropout {} filter_sizes {} batch_size {} augmentation {} weight_decay {} batch_norm {} ".format(
            args.optimizer,
            args.activation,
            args.num_filters,
            args.dropout,
            args.filter_sizes,
            args.batch_size,
            args.augmentation,
            args.weight_decay,
            args.batch_norm
          )
    wandb.init(project=args.wandb_project,entity=args.wandb_entity,name=run_name)

    #Loading the data
    input_val=(args.input_dim,args.input_dim)
    data_loader = DataLoaderHelper(train_directory,test_directory,input_val,args.batch_size,augmentation)

    train_loader, val_loader,test_loader = data_loader.get_dataloaders()

    #Build the CNN model using pytorch as per the given requirements
    model = FlexibleCNN(args.num_filters,args.filter_sizes,args.dropout,args.activation,batch_norm,input_val,args.hidden_size,num_classes)
    
    #Intializing the train model
    trainer = Trainer(model,train_loader,val_loader,test_loader,args.optimizer_name,args.learning_rate,args.num_epochs,args.weight_decay)

    #Training the model
    trainer.train()

    #Plot the confusion matrix and 3*10 images with their predicted and Actual value
    trainer.confusion_matrix(plot=False)
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




