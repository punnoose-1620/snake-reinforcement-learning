#!/usr/bin/env python3
"""
Script to run AI evaluation on 15x10 grid with the latest model
"""
import subprocess
import sys
import os

def run_ai_evaluation():
    """Run AI evaluation with 15x10 grid and latest model"""
    
    # Check if models directory exists
    if not os.path.exists("models"):
        print("No models directory found. Please train a model first.")
        return
    
    # Get list of model files
    model_files = [f for f in os.listdir("models") if f.endswith('.pth')]
    if not model_files:
        print("No trained models found. Please train a model first.")
        return
    
    # Use the latest final model (prefer final models over episode models)
    final_models = [f for f in model_files if 'final' in f]
    if final_models:
        latest_model = sorted(final_models)[-1]
    else:
        latest_model = sorted(model_files)[-1]
    
    model_path = os.path.join("models", latest_model)
    print(f"Using model: {latest_model}")
    print(f"Grid size: 15x10")
    print("Starting AI evaluation...")
    
    # Import and run the AI evaluation directly
    try:
        from snake_game import play_ai_visually
        play_ai_visually(model_path, grid_width=15, grid_height=10)
    except ImportError as e:
        print(f"Error importing snake_game: {e}")
        return
    except Exception as e:
        print(f"Error running AI evaluation: {e}")
        return

if __name__ == "__main__":
    run_ai_evaluation()
