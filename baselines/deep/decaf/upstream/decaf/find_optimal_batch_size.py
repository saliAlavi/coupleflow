#!/usr/bin/env python
import os
import torch
import argparse
import numpy as np
import time
import gc
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda import max_memory_allocated, reset_peak_memory_stats

from src.models.model_factory import ModelFactory
from src.train.loss_factory import LossFactory
from src.data.data_utils import get_dataloaders
from src.models import registry  # This will register all models

def get_memory_usage(batch_size, model_name, config, num_iterations=10):
    """
    Test a specific batch size and return memory usage and time per batch.
    
    Args:
        batch_size: Batch size to test
        model_name: Name of the model to use
        config: Configuration object
        num_iterations: Number of batches to process for timing
        
    Returns:
        Dictionary with memory usage and timing information
    """
    # Update batch size in config
    config.batch_size = batch_size
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    try:
        # Create dataloaders with the current batch size
        train_loader, _, _ = get_dataloaders(config)
        
        # Initialize model
        model = ModelFactory.create(model_name, config)
        model = model.to(device)
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        criterion = LossFactory.create(config.loss_function)
        
        # Reset peak memory stats
        if device.type == 'cuda':
            reset_peak_memory_stats(device)
        
        # Time the forward and backward pass for a few batches
        model.train()
        batch_times = []
        
        # Process a few batches to warm up
        for i, batch in enumerate(train_loader):
            if i >= 2:  # Just 2 batches for warm-up
                break
                
            eeg, envelope = batch
            eeg = eeg.to(device)
            envelope = envelope.to(device)
            
            # Forward pass
            outputs = model(eeg)
            loss = criterion(outputs, envelope)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Reset memory stats after warm-up
        if device.type == 'cuda':
            reset_peak_memory_stats(device)
        
        # Measure time and memory for actual test iterations
        for i, batch in enumerate(train_loader):
            if i >= num_iterations:
                break
                
            eeg, envelope = batch
            eeg = eeg.to(device)
            envelope = envelope.to(device)
            
            # Time the operations
            start_time = time.time()
            
            # Forward pass
            outputs = model(eeg)
            loss = criterion(outputs, envelope)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            end_time = time.time()
            batch_times.append(end_time - start_time)
        
        # Get peak memory usage
        if device.type == 'cuda':
            peak_memory = max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
        else:
            peak_memory = 0  # Not available for CPU
        
        # Calculate average time per batch
        avg_time = np.mean(batch_times) if batch_times else 0
        
        # Clean up to free memory
        del model, optimizer, criterion, train_loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'batch_size': batch_size,
            'peak_memory_mb': peak_memory,
            'avg_time_per_batch': avg_time,
            'status': 'success'
        }
        
    except RuntimeError as e:
        # Catch out of memory errors
        if 'out of memory' in str(e).lower():
            # Clean up
            if 'model' in locals():
                del model
            if 'optimizer' in locals():
                del optimizer
            if 'criterion' in locals():
                del criterion
            if 'train_loader' in locals():
                del train_loader
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            return {
                'batch_size': batch_size,
                'peak_memory_mb': float('nan'),
                'avg_time_per_batch': float('nan'),
                'status': 'OOM error'
            }
        else:
            # Re-raise other errors
            raise

def find_optimal_batch_size(model_name, config, start_size=8, max_size=512, step_factor=2):
    """
    Find the optimal batch size by testing increasingly larger sizes.
    
    Args:
        model_name: Name of the model to use
        config: Configuration object
        start_size: Starting batch size
        max_size: Maximum batch size to test
        step_factor: Factor to increase batch size by in each step
        
    Returns:
        DataFrame with results and recommended batch size
    """
    print(f"Finding optimal batch size for model: {model_name}")
    print(f"Testing batch sizes from {start_size} to {max_size}")
    
    results = []
    batch_size = start_size
    
    # Test increasingly larger batch sizes
    while batch_size <= max_size:
        print(f"\nTesting batch size: {batch_size}")
        result = get_memory_usage(batch_size, model_name, config)
        results.append(result)
        
        print(f"  Status: {result['status']}")
        if result['status'] == 'success':
            print(f"  Peak memory: {result['peak_memory_mb']:.2f} MB")
            print(f"  Avg time per batch: {result['avg_time_per_batch']:.4f} seconds")
        
        # If we hit OOM, stop testing larger sizes
        if result['status'] == 'OOM error':
            print(f"  Out of memory error at batch size {batch_size}")
            break
            
        # Increase batch size for next iteration
        batch_size *= step_factor
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find the largest successful batch size
    successful_results = results_df[results_df['status'] == 'success']
    if not successful_results.empty:
        optimal_batch_size = successful_results['batch_size'].max()
        print(f"\nRecommended batch size: {optimal_batch_size}")
    else:
        optimal_batch_size = None
        print("\nNo successful batch sizes found. Try a smaller starting size.")
    
    return results_df, optimal_batch_size

def plot_results(results_df, save_path=None):
    """
    Plot the memory usage and timing results.
    
    Args:
        results_df: DataFrame with batch size test results
        save_path: Path to save the plot (optional)
    """
    # Filter for successful runs
    successful_df = results_df[results_df['status'] == 'success'].copy()
    
    if successful_df.empty:
        print("No successful runs to plot.")
        return
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot memory usage
    color = 'tab:blue'
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Memory Usage (MB)', color=color)
    ax1.plot(successful_df['batch_size'], successful_df['peak_memory_mb'], 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create second y-axis for timing
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Time per Batch (seconds)', color=color)
    ax2.plot(successful_df['batch_size'], successful_df['avg_time_per_batch'], 's-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title and adjust layout
    plt.title('Batch Size vs. Memory Usage and Training Time')
    fig.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Find optimal batch size for training")
    
    # Model selection
    parser.add_argument("--model", type=str, default="BaselineDecoder",
                      help=f"Model to use. Available models: {list(ModelFactory._models.keys())}")
    
    # Batch size search parameters
    parser.add_argument("--start_size", type=int, default=8,
                      help="Starting batch size")
    parser.add_argument("--max_size", type=int, default=512,
                      help="Maximum batch size to test")
    parser.add_argument("--step_factor", type=int, default=2,
                      help="Factor to increase batch size by in each step")
    
    # Other parameters
    parser.add_argument("--loss_function", type=str, default="mse",
                      help=f"Loss function to use. Options: {list(LossFactory._losses.keys())}")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to use (cuda or cpu)")
    parser.add_argument("--save_plot", type=str, default="batch_size_results.png",
                      help="Path to save the results plot")
    parser.add_argument("--save_csv", type=str, default="batch_size_results.csv",
                      help="Path to save the results CSV")
    
    args = parser.parse_args()
    
    # Get the appropriate config class and create instance
    print(f"Getting config class from model {args.model}")
    ConfigClass = ModelFactory.get_config_class(args.model)
    config = ConfigClass()
    
    # Update config with command line arguments
    config.model_name = args.model
    config.loss_function = args.loss_function
    config.device = args.device
    
    # Find optimal batch size
    results_df, optimal_batch_size = find_optimal_batch_size(
        args.model, 
        config,
        start_size=args.start_size,
        max_size=args.max_size,
        step_factor=args.step_factor
    )
    
    # Save results to CSV
    if args.save_csv:
        results_df.to_csv(args.save_csv, index=False)
        print(f"Results saved to {args.save_csv}")
    
    # Plot results
    if not results_df[results_df['status'] == 'success'].empty:
        plot_results(results_df, args.save_plot)
    
    # Print final recommendation
    if optimal_batch_size:
        print("\n" + "="*50)
        print(f"RECOMMENDATION: Use batch size of {optimal_batch_size} for training")
        print("="*50)
        
        # Calculate throughput (samples per second)
        best_result = results_df[results_df['batch_size'] == optimal_batch_size].iloc[0]
        throughput = optimal_batch_size / best_result['avg_time_per_batch']
        print(f"This will process approximately {throughput:.2f} samples per second")
        print(f"Peak memory usage: {best_result['peak_memory_mb']:.2f} MB")

if __name__ == "__main__":
    main() 