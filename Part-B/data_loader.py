import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

class DataLoaderHelper:
    def __init__(self, train_data_dir,test_data_dir, input_size, batch_size, augmentation):
        self.data_dir = train_data_dir
        self.test_dir = test_data_dir
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.input_size = input_size  # tuple like (400, 400)

        self.transform = self.get_transform()
        self.train_data, self.val_data = self.load_train_val_data()
        self.test_data = self.load_test_data()

    def get_transform(self):
        if self.augmentation:
            #Rotating the image.
            transforms_list = [
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
            ]
        else:
            transforms_list = [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
            ]

        #Normalize the input for better performance
        transforms_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )
        return transforms.Compose(transforms_list)

    def load_train_val_data(self):
        full_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        total_size = len(full_dataset)
        indices = list(range(total_size))

        #Train_test_split does not allow tensor data.. (So need to split based on the indices)
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

        # print(f"Total: {total_size} | Train: {len(train_idx)} | Val: {len(val_idx)}")
        return Subset(full_dataset, train_idx), Subset(full_dataset, val_idx)

    def load_test_data(self):
        return datasets.ImageFolder(root=self.test_dir, transform=self.transform)

    def get_dataloaders(self):
        #Addded pin_memory and workers to increase the loading speed
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size,
                                shuffle=False, num_workers=2, pin_memory=True)

        test_loader = DataLoader(self.test_data, batch_size=self.batch_size,
                                     shuffle=False, num_workers=2, pin_memory=True)
      

        return train_loader, val_loader, test_loader