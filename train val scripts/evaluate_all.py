import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from config import *
from models import get_model
from data_loader import get_data_loaders
from trainer import evaluate_model

def evaluate_all_models():
    """Evaluate results across multiple runs for all models."""
    # Get the test data loader
    _, test_loader = get_data_loaders()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    
    # Store evaluation results for all models
    all_results = {}
    
    for model_name in MODELS:
        print(f"\nEvaluating model: {model_name}")
        model_results = []
        
        for run_id in range(1, NUM_RUNS + 1):
            print(f"  Run {run_id}/{NUM_RUNS}")
            
            # Load the best model
            model = get_model(model_name)
            checkpoint_path = os.path.join(CHECKPOINT_DIR, model_name, f'best_model_run{run_id}.pth')
            
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                model = model.to(device)
                
                # Evaluate model
                _, acc, prec, rec, f1 = evaluate_model(model, test_loader, device, criterion)
                
                model_results.append({
                    'run_id': run_id,
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1_score': f1
                })
                print(f"    Accuracy: {acc:.4f}, F1 score: {f1:.4f}")
        
        # Compute mean and standard deviation
        if model_results:
            df = pd.DataFrame(model_results)
            avg_results = {
                'mean_accuracy': df['accuracy'].mean(),
                'std_accuracy': df['accuracy'].std(),
                'mean_precision': df['precision'].mean(),
                'std_precision': df['precision'].std(),
                'mean_recall': df['recall'].mean(),
                'std_recall': df['recall'].std(),
                'mean_f1': df['f1_score'].mean(),
                'std_f1': df['f1_score'].std(),
                'all_runs': df
            }
            all_results[model_name] = avg_results
    
    # Save results to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_df = []
    
    for model_name, results in all_results.items():
        summary_df.append({
            'model': model_name,
            'mean_accuracy': results['mean_accuracy'],
            'std_accuracy': results['std_accuracy'],
            'mean_f1': results['mean_f1'],
            'std_f1': results['std_f1']
        })
    
    summary_df = pd.DataFrame(summary_df)
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)
    
    # Generate visualization plots
    plot_results(summary_df)
    
    return all_results

def plot_results(summary_df):
    """Plot model performance comparison charts."""
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Sort by mean F1 score
    summary_df = summary_df.sort_values('mean_f1', ascending=False)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    axes[0].bar(range(len(summary_df)), summary_df['mean_accuracy'], 
               yerr=summary_df['std_accuracy'], capsize=5)
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Comparison Across Models')
    axes[0].set_xticks(range(len(summary_df)))
    axes[0].set_xticklabels(summary_df['model'], rotation=45, ha='right')
    axes[0].set_ylim(0.5, 1.0)
    
    # F1 score comparison
    axes[1].bar(range(len(summary_df)), summary_df['mean_f1'], 
               yerr=summary_df['std_f1'], capsize=5)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('F1 score')
    axes[1].set_title('F1 Score Comparison Across Models')
    axes[1].set_xticks(range(len(summary_df)))
    axes[1].set_xticklabels(summary_df['model'], rotation=45, ha='right')
    axes[1].set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=300)
    plt.close()