## Assignment Outline

- In this Particular Assignment Part A , I have trained CNN from the Scratch

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
model = FlexibleCNN(args.num_filters,args.filter_sizes,args.dropout,args.activation,batch_norm,input_val,args.hidden_size,num_classes)
```

- To create a model , it takes Number of filters and their sizes at each layer , dropout , Activation function, batch_norm(True,False) ,hidden_size, num_classes.
- As mentioned in Assignment the code should be Flexible for different number of Full connected layers .. Currently nargs set to 1 in parser arguments for hidden_size, You can increase that and can send the appropriate layer sizes.

## How to train the model ?
**File:** `model_trainer.py` 
```bash
trainer = Trainer(model,train_loader,val_loader,test_loader,args.optimizer_name,args.learning_rate,args.num_epochs,args.weight_decay)
trainer.train()
```

- Trainer function takes CNN model, optimizer name , weight_decay, Number of epochs and the train and validation and test data loaders.



```text
train.py is the file one needs to run to see the running of the model and it takes follow arguments. For better modularity purpose Each functionality is implemented in separate files.
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
  **Default:** `64`

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
  **Default:** `0.0005`

- `--base_dir`, `-br`  
  **Description:** Base directory containing the dataset with `train/` and `val/` folders.  
  **Type:** `str`  
  **Default:** `inaturalist_12K`  
  **Note:** Avoid using trailing backslashes (`\`).

- `--activation`, `-ac`  
  **Description:** Activation function used in the neural network layers.  
  **Choices:** `relu`, `elu`, `selu`, `silu`, `gelu`, `mish`  
  **Default:** `silu`

- `--num_filters`, `-nf`  
  **Description:** Number of filters for convolutional layers, specified as a list of integers.  
  **Type:** `int[]`  
  **Default:** `[128, 128, 64, 64, 32]`

- `--filter_sizes`, `-fs`  
  **Description:** Sizes of the filters for convolutional layers, specified as a list of integers.  
  **Type:** `int[]`  
  **Default:** `[5, 5, 5, 5, 5]`

- `--input_dim`, `-in`  
  **Description:** Size of the input image (image will be resized to `input_dim x input_dim`).  
  **Type:** `int`  
  **Default:** `224`

- `--batch_norm`, `-bn`  
  **Description:** Indicates whether batch normalization is applied in CNN layers.  
  **Choices:** `true`, `false`  
  **Default:** `true`

- `--optimizer_name`, `-o`  
  **Description:** Optimizer used for training the neural network.  
  **Choices:** `nadam`, `adam`, `rmsprop`  
  **Default:** `rmsprop`

- `--hidden_size`, `-dl`  
  **Description:** Number of units in the dense (fully connected) layer.  
  **Type:** `int[]`  
  **Default:** `[512]`

- `--dropout`, `-dp`  
  **Description:** Dropout rate for the dropout layers.  
  **Type:** `float`  
  **Default:** `0.4`
  
  

  
## How to do a sample run with default parameters ?

```bash
!python train.py -br "/kaggle/input/cnndataset1/inaturalist_12K"

```

- If you are running on Gpu device which contains more than one, I have mentioned in model_trainer file how to utilize it's functionality.