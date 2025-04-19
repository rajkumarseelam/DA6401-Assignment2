## Assignment Outline

- In this Particular Assignment Part B , we are trying to use a pretrained model.
- Here I'm using Resnet50 and training the model by Freezing all layers except the last layer.

## How to Load the data ?

**File:** `data_loader.py` 

```bash
data_loader = DataLoaderHelper(train_directory,test_directory,input_val,args.batch_size,augmentation)
train_loader, val_loader,test_loader = data_loader.get_dataloaders()
```
- DataLoaderHelper function will Load the data and split it into training and validation by doing required transformations. It requires batch_size and augmentation flag and input_val in Tuple format.

## How to create a model ?
**File:** `model.py` 

```bash
model=FinetuneCNN()
model.Freezelayers()
```

- We are just loading the Pretrained Resnet50 model and Freezing all the layers except the last one.

## How to train the model ?
**File:** `model_trainer.py` 
```bash
trainer = Trainer(model,train_loader,val_loader,test_loader,args.optimizer_name,args.learning_rate,args.num_epochs,args.weight_decay)
trainer.train()
```

- Trainer function takes pretrained model , optimizer name , weight_decay, Number of epochs and the train and validation and test data loaders.



```text
train.py is the file one needs to run to see the running of the model and it takes follow arguments. For better modularity purpose Each functionality is implemented in separate files.

Note: Download the data_loader.py, model.py ,model_trainer.py  files before running the train.py.
```




## Command-Line Arguments

Below are the command-line arguments supported by the script, sepcifing default values and the inputs it will take
  
  
- `--wandb_entity`, `-we`  
  **Description:** WandB entity used to track experiments. Typically your WandB username or team name.  
  **Type:** `str`  
  **Default:** `cs24m042-iit-madras-foundation`

- `--wandb_project`, `-wp`  
  **Description:** Project name used in WandB for organizing experiment logs.  
  **Type:** `str`  
  **Default:** `DA6401-Assignment-2`

- `--num_epochs`, `-e`  
  **Description:** Number of epochs to train the CNN model.  
  **Type:** `int`  
  **Default:** `10`

- `--batch_size`, `-b`  
  **Description:** Batch size used to train the neural network.  
  **Type:** `int`  
  **Default:** `32`

- `--augmentation`, `-au`  
  **Description:** Indicates whether data augmentation should be applied.  
  **Choices:** `true`, `false`  
  **Default:** `false`

- `--learning_rate`, `-lr`  
  **Description:** Learning rate for the optimizer.  
  **Type:** `float`  
  **Default:** `0.0001`

- `--weight_decay`, `-w_d`  
  **Description:** Weight decay factor for regularization.  
  **Type:** `float`  
  **Default:** `0`

- `--base_dir`, `-br`  
  **Description:** Base directory containing the dataset with `train/` and `val/` folders.  
  **Type:** `str`  
  **Default:** `inaturalist_12K`  
  **Note:** Avoid using trailing backslashes (`\`).



- `--optimizer_name`, `-o`  
  **Description:** Optimizer used for training the neural network.  
  **Choices:** `nadam`, `adam`  
  **Default:** `nadam`
  
  

  
## How to do a sample run with default parameters ?

```bash
!python train.py -br "/kaggle/input/cnndataset1/inaturalist_12K"

```
Note: Better login first into wandb before running train.py .
```
!pip install wandb

import wandb
wandb.login(key='Your key')
```
- If you are running on Gpu device which contains more than one, I have mentioned in model_trainer file how to utilize it's functionality.