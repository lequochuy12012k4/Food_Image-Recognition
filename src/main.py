# main.py
import argparse
from src import config
from src.data_processing.data_loader import create_dataloaders
from src.models.neural_network_model import create_food_model
from src.models.model_trainer import ModelTrainer
from src.predict import Predictor

def main():
    parser = argparse.ArgumentParser(description="Food Recognition and Recipe Model")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Training parser ---
    parser_train = subparsers.add_parser("train", help="Train the food recognition model")

    # --- Prediction parser ---
    parser_predict = subparsers.add_parser("predict", help="Predict food and get recipe from an image")
    parser_predict.add_argument("image_path", type=str, help="Path to the image file")

    args = parser.parse_args()

    if args.command == "train":
        print(f"Using device: {config.DEVICE}")
        
        # 1. Create DataLoaders
        train_loader, val_loader, _ = create_dataloaders()
        
        # 2. Create Model
        model = create_food_model()
        
        # 3. Train Model
        trainer = ModelTrainer(model, train_loader, val_loader, config.DEVICE)
        trainer.train()

    elif args.command == "predict":
        # Initialize the predictor
        predictor = Predictor(model_path=config.MODEL_PATH, class_names_path=config.CLASS_NAMES_PATH)
        
        # Perform prediction
        predictor.predict(args.image_path)

if __name__ == "__main__":
    main()