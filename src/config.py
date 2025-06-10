# src/config.py
import torch

# -- Project Paths --
DATA_PATH = "data/processed/food-101/"
MODEL_PATH = "models/food_recognizer_resnet50.pth"
CLASS_NAMES_PATH = "models/class_names.json" # To save the class mapping

# -- Model Hyperparameters --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_CLASSES = 101
LEARNING_RATE = 0.001
NUM_EPOCHS = 10 # Start with 10, increase for better performance

# -- Image Transformations --
IMG_SIZE = 224