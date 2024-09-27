import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataset(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])      
    ])
    dataset = datasets.ImageFolder(root='D:\downloads\\animals\\animals\\animals', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    classes_name=dataset.class_to_idx
    return dataloader,classes_name

