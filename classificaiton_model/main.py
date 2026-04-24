import os
import torch
import numpy as np
import random
from config import *
from models import get_model
from data_loader import get_data_loaders
from trainer import train_model
from evaluate_all import evaluate_all_models

def setup_seed(seed):
    """Set random seeds to ensure reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_all_models():
    """Train all models, running each model multiple times."""
    # Create required directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    for model_name in MODELS:
        print(f"\nStart training model: {model_name}")
        
        for run_id in range(1, NUM_RUNS + 1):
            print(f"\n  Run {run_id}/{NUM_RUNS}")
            setup_seed(42 + run_id)  # Use a different seed for each run
            
            # Get data loaders
            train_loader, test_loader = get_data_loaders()
            
            # Create model
            if model_name == 'logistic':
                model = get_model(model_name, pretrained=False)  # Logistic regression does not need pretraining
            else:
                model = get_model(model_name, pretrained=True)
            
            # Train model
            print(f"  Start training {model_name} (run {run_id})")
            train_model(model, train_loader, test_loader, model_name, run_id)

if __name__ == "__main__":
    # Train all models
    train_all_models()
    
    # Evaluate all models and generate the result report
    print("\nStart evaluating all models...")
    results = evaluate_all_models()
    
    print("\nModel evaluation completed!")
    print("Results saved to:", RESULTS_DIR)