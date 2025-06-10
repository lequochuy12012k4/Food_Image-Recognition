# src/models/neural_network_model.py
import torch
import torch.nn as nn
from torchvision import models
from src import config

def create_food_model(num_classes=config.NUM_CLASSES):
    """
    Loads a pre-trained ResNet50 model and adapts it for food classification.
    """
    # Load a pre-trained ResNet50 model with the latest recommended weights
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze the parameters of the pre-trained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer (the classifier)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    print("ResNet50 model created and final layer replaced.")
    return model