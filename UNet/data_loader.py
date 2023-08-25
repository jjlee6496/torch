import torch
from torchvision import transforms as T
from dataset import PennFundanDataset

def get_data_loaders(root, batch_size, training=True):
    if training:
        transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=(-30, 30)),
            # T.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
       transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        
    train_dataset = PennFundanDataset(root, transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, training)
    
    return train_loader