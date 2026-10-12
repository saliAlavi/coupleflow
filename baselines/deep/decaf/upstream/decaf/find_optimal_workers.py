import time
import torch
import json
from src.data.eeg_dataloader import create_dataloader, StandardizeEEG, StandardizeEnvelope
from itertools import product
from configs.baseline_config import BaselineConfig

def benchmark_dataloader(num_workers, batch_size, pin_memory=True, prefetch_factor=2):
    """Benchmark data loading speed with different parameters."""
    
    # Initialize transforms
    transform_eeg = StandardizeEEG()
    transform_envelope = StandardizeEnvelope()
    
    config = BaselineConfig()
    
    # Create dataloader with specified parameters
    loader = create_dataloader(
        filelist_path=config.train_filelist,
        base_dir=config.data_base_dir,
        batch_size=batch_size,
        transform_eeg=transform_eeg,
        transform_envelope=transform_envelope,
        num_workers=num_workers,
        shuffle=True,
        cache_size=10,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        context_seconds=config.context_seconds,
        prediction_seconds=config.prediction_seconds,
        sample_rate=config.sampling_rate
    )
    
    start_time = time.time()
    
    # Load several batches and measure time
    num_batches = 5 # Test with multiple batches for more reliable results
    for idx, (metadata, eegs, envelopes, eeg_context, envelope_context) in enumerate(loader):
        if torch.cuda.is_available():  # If using GPU, include data transfer time
            print(f"Moving data to GPU: {eegs.shape}, {envelopes.shape}, {envelope_context.shape}")
            eegs = eegs.cuda()
            envelopes = envelopes.cuda()
            envelope_context = envelope_context.cuda()
            eeg_context = eeg_context.cuda()
        if idx >= num_batches:
            break
            
    end_time = time.time()
    total_time = end_time - start_time
    
    return total_time

def main():
    config = BaselineConfig()
    batch_size = config.batch_size
    
    # Test parameters
    worker_options = [0, 2, 4, 8, 12, 16]
    pin_memory_options = [True, False]
    prefetch_factors = [2, 4, 8]
    results = []
    
    print("\nTesting different DataLoader configurations...")
    print("============================================")
    print(f"Running with batch_size={batch_size} (from config)")
    print("This will take several minutes...")
    print("--------------------------------------------")
    
    # Test all combinations
    for num_workers in worker_options:
        if num_workers == 0:
            # For num_workers=0, only test pin_memory options
            for pin_memory in pin_memory_options:
                print(f"Testing workers={num_workers:2d}, pin_memory={pin_memory}, prefetch=N/A...", 
                      end=" ", flush=True)
                time_taken = benchmark_dataloader(
                    num_workers=num_workers,
                    batch_size=batch_size,
                    pin_memory=pin_memory,
                    prefetch_factor=None
                )
                results.append((num_workers, pin_memory, None, time_taken))
                print(f"Time: {time_taken:.2f}s")
        else:
            # Test all combinations for non-zero workers
            for pin_memory, prefetch in product(pin_memory_options, prefetch_factors):
                print(f"Testing workers={num_workers:2d}, pin_memory={pin_memory}, prefetch={prefetch}...", 
                      end=" ", flush=True)
                time_taken = benchmark_dataloader(
                    num_workers=num_workers,
                    batch_size=batch_size,
                    pin_memory=pin_memory,
                    prefetch_factor=prefetch
                )
                results.append((num_workers, pin_memory, prefetch, time_taken))
                print(f"Time: {time_taken:.2f}s")
    
    # Find optimal configuration
    optimal_config = min(results, key=lambda x: x[3])
    
    print("\nResults Summary (sorted by speed):")
    print("--------------------------------")
    print("Format: (workers, pin_memory, prefetch) -> time")
    for workers, pin_mem, prefetch, time_taken in sorted(results, key=lambda x: x[3]):
        prefetch_str = str(prefetch) if prefetch is not None else "N/A"
        print(f"Workers: {workers:2d}, Pin Memory: {pin_mem:5}, Prefetch: {prefetch_str:3} -> {time_taken:.2f}s")
    
    print("\nOptimal Configuration:")
    print(f"Number of workers: {optimal_config[0]}")
    print(f"Pin memory: {optimal_config[1]}")
    print(f"Prefetch factor: {optimal_config[2] if optimal_config[2] is not None else 'N/A'}")
    print(f"Time: {optimal_config[3]:.2f}s")
    
    if torch.cuda.is_available():
        print("\nNote: Times include GPU transfer since CUDA is available")
        print("Pin memory typically helps more with GPU transfers")
    
    # Save optimal configuration to file
    optimal_params = {
        'num_workers': optimal_config[0],
        'pin_memory': optimal_config[1],
        'prefetch_factor': optimal_config[2] if optimal_config[2] is not None else "N/A",
        'batch_size_used': batch_size
    }
    
    with open('optimal_dataloader_config_bs64.json', 'w') as f:
        json.dump(optimal_params, f, indent=4)
    print("\nOptimal configuration saved to 'optimal_dataloader_config.json'")

if __name__ == "__main__":
    main() 