# src/data_processing/data_loader.py
import os
import json
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from src import config

# --- Define Transformations ---
# We still need the same image transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config.IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Helper functions to apply transforms ---
def apply_train_transforms(batch):
    """Applies training transformations to a batch of images."""
    batch['image'] = [train_transform(img.convert("RGB")) for img in batch['image']]
    return batch

def apply_val_transforms(batch):
    """Applies validation transformations to a batch of images."""
    batch['image'] = [val_transform(img.convert("RGB")) for img in batch['image']]
    return batch


def create_dataloaders():
    """
    Creates training and validation dataloaders using Hugging Face datasets.
    This will automatically download and cache the Food-101 dataset.
    """
    print("Loading Food-101 dataset from Hugging Face...")
    # This single line handles downloading and loading the dataset
    # It might take a while the first time you run it.
    food_dataset = load_dataset("food101", split=['train', 'validation'])
    
    train_dataset = food_dataset[0] # The 'train' split
    val_dataset = food_dataset[1]   # The 'validation' split

    # Get class names from the dataset features
    class_names = train_dataset.features['label'].names
    
    # Save the class names mapping for prediction later
    os.makedirs(os.path.dirname(config.CLASS_NAMES_PATH), exist_ok=True)
    with open(config.CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"Saved class names to {config.CLASS_NAMES_PATH}")

    # Set the transforms for the datasets
    train_dataset.set_transform(apply_train_transforms)
    val_dataset.set_transform(apply_val_transforms)

    # Create the DataLoaders
    # Note: We need a custom collate function because the dataset returns a dict
    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['image'] for x in batch]),
            'labels': torch.tensor([x['label'] for x in batch])
        }
    
    # We need to adapt the trainer to handle this new format,
    # or simplify the dataloader output. Let's simplify.
    
    # Let's re-wrap it to be simpler for our existing trainer.
    # We'll create a lightweight wrapper.
    class HFDatasetWrapper:
        def __init__(self, hf_dataset):
            self.hf_dataset = hf_dataset
        
        def __len__(self):
            return len(self.hf_dataset)

        def __getitem__(self, idx):
            item = self.hf_dataset[idx]
            return item['image'], item['label']

    final_train_dataset = HFDatasetWrapper(train_dataset)
    final_val_dataset = HFDatasetWrapper(val_dataset)

    train_loader = DataLoader(
        final_train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=os.cpu_count()//2
    )
    val_loader = DataLoader(
        final_val_dataset,
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=os.cpu_count()//2
    )

    print(f"Dataset loaded. Found {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    return train_loader, val_loader, class_names