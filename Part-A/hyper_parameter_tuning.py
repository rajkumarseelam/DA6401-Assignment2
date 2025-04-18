import wandb


from model import FlexibleCNN
from data_loader import DataLoaderHelper
from model_trainer import Trainer

input_dim=(400,400)
num_classes=10

# Add your directory here 
train_directory='/kaggle/input/cnndataset1/inaturalist_12K/train'
test_directory='/kaggle/input/cnndataset1/inaturalist_12K/val'
epochs=10

# Sweep configuration dictionary for wandb
sweep_configuration = {
    'method': 'bayes',
    'name' : 'cnn-hyperparameter-tuning_test',
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'
    },
    'parameters': {
        'num_filters': {
          'values': [[64,128,256,512, 1024], [32,32,32,32,32],[32,64,64,128,128],[128,128,64,64,32],[32,64,128,256,512]]
        },
        'filter_sizes': {
          'values': [[3,3,3,3,3], [5,5,5,5,5], [5,3,5,3,5]]
        },
        'weight_decay': {
            'values':[0, 0.0005, 0.5]
        },
        'augmentation': {
            'values': [True, False]
        },
        'dropout': {
            'values': [0, 0.2, 0.4]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'activation': {
            'values': ['relu', 'elu', 'selu', 'silu', 'gelu','mish']
        },
        'optimizer': {
            'values': ['nadam', 'adam', 'rmsprop']
        },
        'batch_norm':{
            'values': [True, False]
        },
        'batch_size': {
            'values': [32, 64]
        },
        'fc_hidden_sizes':{
            'values': [128, 256, 512]
        }
    }
}

def train_sweep(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config


        run.name = "optimizer {} activation {} num_filters {} dropout {} filter_sizes {} batch_size {} augmentation {} weight_decay {} batch_norm {} ".format(
            config.optimizer,
            config.activation,
            config.num_filters,
            config.dropout,
            config.filter_sizes,
            config.batch_size,
            config.augmentation,
            config.weight_decay,
            config.batch_norm
          )
        # Initialize data loaders
        data_loader = DataLoaderHelper(
            train_directory,test_data_dir=test_directory,
            input_size=input_dim,
            batch_size=config.batch_size,
            augmentation=config.augmentation
        )
        train_loader, val_loader,test_loader = data_loader.get_dataloaders()
        
        hidden_sizes = [config.fc_hidden_sizes]
        # Initialize model
        model = FlexibleCNN(
            num_filters=config.num_filters,
            filter_sizes=config.filter_sizes,
            dropout=config.dropout,
            activation=config.activation,
            batch_norm=config.batch_norm,
            input_size=input_dim,
            fc_hidden_sizes=hidden_sizes,
            num_classes=num_classes
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
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