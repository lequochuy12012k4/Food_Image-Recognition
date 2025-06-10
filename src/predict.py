# src/predict.py
import torch
import json
from PIL import Image
from torchvision import transforms

from src import config
from src.models.neural_network_model import create_food_model
from src.utils.recipe_api_client import get_recipe_info

class Predictor:
    def __init__(self, model_path, class_names_path):
        self.device = config.DEVICE
        
        # Load class names
        with open(class_names_path) as f:
            self.class_names = json.load(f)
        
        # Load model
        self.model = create_food_model(num_classes=len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path):
        # 1. Image Recognition
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_food_name = self.class_names[predicted_idx.item()]
        confidence_score = confidence.item()
        
        # Clean up name for API call (e.g., 'baby_back_ribs' -> 'baby back ribs')
        formatted_food_name = predicted_food_name.replace('_', ' ')
        
        print(f"\nModel recognized: '{formatted_food_name}' with {confidence_score:.2%} confidence.")
        
        # 2. Information Retrieval
        print("Fetching recipe details...")
        recipe_details = get_recipe_info(formatted_food_name)
        
        # 3. Display Final Result
        if not recipe_details:
            print(f"\nSorry, could not find a recipe for '{formatted_food_name}'.")
            return

        print("\n--- Here is your recipe! ---")
        print(f"Dish: {recipe_details['name']}")
        
        print("\nIngredients:")
        for ingredient in recipe_details['ingredients']:
            print(f"- {ingredient}")
        
        print("\nInstructions:")
        # Simple formatting to make instructions more readable
        instructions_text = recipe_details['instructions']
        if instructions_text:
            print(instructions_text.replace('<ol>', '').replace('</ol>', '').replace('<ul>', '').replace('</ul>', '').replace('<li>', '\n- ').replace('</li>', ''))
        else:
            print("No instructions provided.")