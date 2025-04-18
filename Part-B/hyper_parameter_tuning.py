import wandb
from model import FinetuneCNN
from data_loader import DataLoaderHelper
from model_trainer import Trainer


input_dim=(224,224)
num_classes=10

# Add your directory here 
train_directory='/kaggle/input/dataset1/inaturalist_12K/train'
test_directory='/kaggle/input/dataset1/inaturalist_12K/val'
epochs=10

# Sweep configuration dictionary for wandb
sweep_configuration = {
    'method': 'bayes',
    'name' : 'cnn-finetunining',
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'
    },
    'parameters': {
        'weight_decay': {
            'values':[0, 0.0005, 0.5]
        },
        'augmentation': {
            'values': [True, False]
        },
       
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'optimizer': {
            'values': ['nadam', 'adam']
        },
        'batch_size': {
            'values': [32, 64]
        }
    }
}

def train_sweep(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config


        run.name = "optimizer {}   batch_size {} augmentation {} weight_decay {} learning_rate {}  ".format(
            config.optimizer,
            config.batch_size,
            config.augmentation,
            config.weight_decay,
            config.learning_rate
          )
        # Initialize data loaders
        data_loader = DataLoaderHelper(
            train_directory,test_data_dir=test_directory,
            input_size=input_dim,
            batch_size=config.batch_size,
            augmentation=config.augmentation
        )
        train_loader, val_loader,test_loader = data_loader.get_dataloaders()
        
        # Initialize model
        model=FinetuneCNN()
        model.Freezelayers()
        
        # Initialize trainer
        trainer = Trainer(
            model.fine_tune_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer_name=config.optimizer,
            learning_rate=config.learning_rate,
            num_epochs=epochs,
            weight_decay=config.weight_decay
        )
        
        # Train the model
        trainer.train()
        
        # Log final metrics
        for epoch in range(epochs):
            wandb.log({
                'train_accuracy': trainer.train_acc_history[epoch]*100,
                'train_loss': trainer.train_loss_history[epoch],
                'val_accuracy': trainer.val_acc_history[epoch]*100,
                'val_loss': trainer.val_loss_history[epoch],
                'epoch' : epoch
            })

if __name__ == "__main__":

    
    sweep_id = wandb.sweep(sweep_configuration, project="DA6401-Assignment-2")

    # Start sweep
    wandb.agent(sweep_id, function=train_sweep, count=1)