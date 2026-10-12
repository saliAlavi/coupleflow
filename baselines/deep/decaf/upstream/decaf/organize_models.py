#!/usr/bin/env python3
import re
import os
import glob
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def organize_model_paths(directory="checkpoints/model_comparison_analysis/visualizations_test_cropped_ending_at_0.5sec/"):
    """
    Identify and organize model paths by model type and context length.
    
    Args:
        directory: Directory containing model pickle files
        
    Returns:
        model_dict: Dictionary mapping model types to lists of (context_length, path) tuples
        df: DataFrame with organized model information
    """
    # Find all model data pickle files
    pattern = os.path.join(directory, "model_data_*.pkl")
    model_paths = glob.glob(pattern)
    
    # Define model type patterns to match
    model_types = {
        "BaselineDecoder": "BaselineDecoder",
        "Happyquoka": "Happyquoka",
        "TwoBranchContext": "TwoBranchContext",
        "TwoBranchFrozenEegContext_EnvelopeOnly": "TwoBranchFrozenEegContext.*EnvelopeOnly"
    }
    
    # Dictionary to store organized model paths
    model_dict = defaultdict(list)
    
    # List to store extracted information for dataframe
    model_data = []
    
    for path in model_paths:
        filename = os.path.basename(path)
        
        # Extract model type
        model_type = None
        for type_name, pattern in model_types.items():
            if re.search(pattern, filename):
                model_type = type_name
                break
        
        if model_type is None:
            # If no match, use the part after model_data_ and before first _
            match = re.search(r'model_data_([^_]+)', filename)
            if match:
                model_type = match.group(1)
            else:
                model_type = "Unknown"
        
        # Extract context length if applicable
        context_length = 0.0
        context_match = re.search(r'Context(\d+\.\d+)Sec', filename)
        if context_match:
            context_length = float(context_match.group(1))
        
        # Extract model variant (e.g., Dynamic, EnvelopeOnly)
        variant = "Base"
        if "Dynamic" in filename:
            variant = "Dynamic"
        elif "EnvelopeOnly" in filename:
            variant = "EnvelopeOnly"
            
        # Extract date for sorting by newest version if there are duplicates
        date_match = re.search(r'(\d{8}_\d{6})\.pkl$', filename)
        date_str = date_match.group(1) if date_match else ""
        
        # Store in dict for traditional access
        model_dict[model_type].append((context_length, path, variant, date_str))
        
        # Store in list for DataFrame
        model_data.append({
            "model_type": model_type,
            "context_length": context_length,
            "variant": variant,
            "date": date_str,
            "path": path,
            "filename": filename
        })
    
    # Convert to DataFrame for easier filtering/sorting
    df = pd.DataFrame(model_data)
    
    # Sort each model type by context length
    for model_type in model_dict:
        model_dict[model_type].sort(key=lambda x: (x[0], x[3]))
    
    # Sort the DataFrame
    df = df.sort_values(["model_type", "context_length", "variant", "date"])
    
    # Handle duplicates - keep the newest version based on date
    df = df.sort_values("date", ascending=False).drop_duplicates(
        subset=["model_type", "context_length", "variant"], 
        keep="first"
    ).sort_values(["model_type", "context_length", "variant"])
    
    return model_dict, df

def print_organized_models(model_dict):
    """
    Print organized model paths in a readable format.
    
    Args:
        model_dict: Dictionary mapping model types to lists of (context_length, path) tuples
    """
    for model_type, models in model_dict.items():
        print(f"\n{model_type}:")
        for context_length, path, variant, _ in models:
            context_str = f"{context_length:.2f} sec" if context_length > 0 else "N/A"
            print(f"  Context: {context_str}, Variant: {variant}")
            print(f"    {path}")

def visualize_model_organization(df):
    """
    Create a visual representation of available models by type and context length.
    
    Args:
        df: DataFrame with organized model information
    """
    # Get unique model types and context lengths
    model_types = df["model_type"].unique()
    
    # Prepare data for visualization
    model_counts = df.groupby(["model_type", "variant"]).size().reset_index(name="count")
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Define colors for variants
    variant_colors = {
        "Base": "gray",
        "Dynamic": "blue", 
        "EnvelopeOnly": "green"
    }
    
    # Plot model types and variants
    ax = plt.subplot(1, 2, 1)
    for i, variant in enumerate(["Base", "Dynamic", "EnvelopeOnly"]):
        variant_data = model_counts[model_counts["variant"] == variant]
        if not variant_data.empty:
            x = range(len(variant_data))
            plt.bar(
                [p + i*0.25 for p in x], 
                variant_data["count"], 
                width=0.25, 
                color=variant_colors[variant], 
                label=variant
            )
    
    plt.xlabel("Model Type")
    plt.ylabel("Number of Models")
    plt.title("Models by Type and Variant")
    plt.xticks(range(len(model_types)), model_types, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Plot context lengths by model type
    ax = plt.subplot(1, 2, 2)
    
    # Filter out models without context lengths
    context_df = df[df["context_length"] > 0]
    
    # Extract unique model types with context
    context_model_types = context_df["model_type"].unique()
    
    for i, model_type in enumerate(context_model_types):
        model_data = context_df[context_df["model_type"] == model_type]
        
        # Plot Dynamic variant
        dynamic_data = model_data[model_data["variant"] == "Dynamic"]
        if not dynamic_data.empty:
            plt.scatter(
                dynamic_data["context_length"], 
                [i] * len(dynamic_data), 
                color="blue", 
                marker="o", 
                s=100, 
                label="Dynamic" if i == 0 else ""
            )
        
        # Plot EnvelopeOnly variant
        envelope_data = model_data[model_data["variant"] == "EnvelopeOnly"]
        if not envelope_data.empty:
            plt.scatter(
                envelope_data["context_length"], 
                [i] * len(envelope_data), 
                color="green", 
                marker="s", 
                s=100, 
                label="EnvelopeOnly" if i == 0 else ""
            )
    
    plt.yticks(range(len(context_model_types)), context_model_types)
    plt.xlabel("Context Length (seconds)")
    plt.title("Available Context Lengths by Model Type")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("plots/model_organization.png", dpi=300)
    plt.close()

def get_model_paths_for_sub001_comparison(df, context_lengths=None):
    """
    Select model paths for sub-001 comparison with specified context lengths.
    
    Args:
        df: DataFrame with organized model information
        context_lengths: List of context lengths to include (default: all)
        
    Returns:
        comparison_paths: Dictionary of model paths for comparison
    """
    comparison_paths = {}
    
    # Get baseline model if available
    baseline = df[df["model_type"].str.contains("Baseline")]
    if not baseline.empty:
        comparison_paths["baseline"] = baseline.iloc[0]["path"]
    
    # Get Happyquoka model if available
    happyquoka = df[df["model_type"].str.contains("Happyquoka")]
    if not happyquoka.empty:
        comparison_paths["eeg_only"] = happyquoka.iloc[0]["path"]
    
    # Filter context lengths if specified
    if context_lengths is not None:
        filtered_df = df[df["context_length"].isin(context_lengths)]
    else:
        filtered_df = df.copy()
    
    # Get all TwoBranchFrozenEegContext_EnvelopeOnly models
    envelope_models = filtered_df[filtered_df["model_type"] == "TwoBranchFrozenEegContext_EnvelopeOnly"]
    for _, row in envelope_models.iterrows():
        key = f"envelope_{row['context_length']:.2f}"
        comparison_paths[key] = row["path"]
    
    # Get all TwoBranchFrozenEegContext_Dynamic models
    fusion_frozen_models = filtered_df[filtered_df["model_type"] == "TwoBranchFrozenEegContext_Dynamic"]
    for _, row in fusion_frozen_models.iterrows():
        key = f"fusion_frozen_{row['context_length']:.2f}"
        comparison_paths[key] = row["path"]
    
    # Get all TwoBranchContext models
    fusion_models = filtered_df[filtered_df["model_type"] == "TwoBranchContext"]
    for _, row in fusion_models.iterrows():
        key = f"fusion_{row['context_length']:.2f}"
        comparison_paths[key] = row["path"]
    
    return comparison_paths

if __name__ == "__main__":
    # Organize model paths
    model_dict, model_df = organize_model_paths()
    
    # Print organized models
    print_organized_models(model_dict)
    
    # Visualize model organization
    visualize_model_organization(model_df)
    
    # Get paths for sub-001 comparison
    # Example: For context lengths 0.1, 0.5, and 1.0
    comparison_paths = get_model_paths_for_sub001_comparison(
        model_df, 
        context_lengths=[0.1, 0.5, 1.0]
    )
    
    print("\n\nModel paths for sub-001 comparison:")
    for model_key, path in comparison_paths.items():
        print(f"{model_key}: {path}") 