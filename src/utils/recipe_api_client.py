# src/utils/recipe_api_client.py
import os
import requests
from dotenv import load_dotenv

load_dotenv() # Loads variables from .env file

API_KEY = os.getenv("SPOONACULAR_API_KEY")
BASE_URL = "https://api.spoonacular.com"

def get_recipe_info(food_name: str):
    """
    Fetches recipe information for a given food name using the Spoonacular API.
    """
    if not API_KEY:
        raise ValueError("SPOONACULAR_API_KEY not found in .env file. Please add it.")

    # 1. Search for a recipe ID by food name
    search_endpoint = f"{BASE_URL}/recipes/complexSearch"
    search_params = {"query": food_name, "number": 1, "apiKey": API_KEY}
    
    try:
        response = requests.get(search_endpoint, params=search_params)
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
        search_results = response.json()
        
        if not search_results or not search_results.get('results'):
            print(f"Warning: No search results found for '{food_name}'.")
            return None

        recipe_id = search_results['results'][0]['id']
        
        # 2. Get detailed recipe information using the ID
        info_endpoint = f"{BASE_URL}/recipes/{recipe_id}/information"
        info_params = {"apiKey": API_KEY}
        response = requests.get(info_endpoint, params=info_params)
        response.raise_for_status()
        recipe_data = response.json()

        # 3. Extract and format the required information
        ingredients = [ing['original'] for ing in recipe_data.get('extendedIngredients', [])]
        instructions = recipe_data.get('instructions', 'No instructions available.')
        
        return {
            "name": recipe_data.get('title', food_name),
            "ingredients": ingredients,
            "instructions": instructions
        }
    except requests.exceptions.RequestException as e:
        print(f"An API error occurred: {e}")
        return None