# src/models/model_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from src import config

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=config.LEARNING_RATE)
        self.best_val_loss = float('inf')

    def _train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=loss.item())
        
        return running_loss / len(self.train_loader.dataset)

    def _validate_one_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels.data)

        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = correct_predictions.double() / len(self.val_loader.dataset)
        return val_loss, val_acc

    def train(self):
        print("Starting model training...")
        for epoch in range(config.NUM_EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
            
            train_loss = self._train_one_epoch()
            val_loss, val_acc = self._validate_one_epoch()
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save the best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
                torch.save(self.model.state_dict(), config.MODEL_PATH)
                print(f"Model improved. Saved to {config.MODEL_PATH}")

        print("\nTraining finished.")