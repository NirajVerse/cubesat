import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import *

class OHIDFFDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

def create_dataframes():

    fire_dir = os.path.join(DATASET_PATH, 'fire')
    nonfire_dir = os.path.join(DATASET_PATH, 'non_fire')
    

    all_images = []
    all_labels = []
    
  
    if os.path.exists(fire_dir):
        for img_name in os.listdir(fire_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(fire_dir, img_name))
                all_labels.append(1)
    
 
    if os.path.exists(nonfire_dir):
        for img_name in os.listdir(nonfire_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(nonfire_dir, img_name))
                all_labels.append(0)  
    

    df = pd.DataFrame({
        'image_path': all_images,
        'label': all_labels
    })
    
    train_df, test_df = train_test_split(
        df, 
        test_size=TEST_SIZE / len(df), 
        stratify=df['label'], 
        random_state=42
    )
    
    return train_df, test_df

def get_data_loaders():

    train_df, test_df = create_dataframes()
    

    train_transform = transforms.Compose([
        transforms.Resize((RESIZE_TO, RESIZE_TO)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
 
    test_transform = transforms.Compose([
        transforms.Resize((RESIZE_TO, RESIZE_TO)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
  
    train_dataset = OHIDFFDataset(train_df, transform=train_transform)
    test_dataset = OHIDFFDataset(test_df, transform=test_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    return train_loader, test_loader