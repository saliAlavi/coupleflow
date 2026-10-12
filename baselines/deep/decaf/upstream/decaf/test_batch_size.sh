#!/bin/bash

# Array of batch sizes to test
batch_sizes=(64 128 256 512 1024 2048 4096)

# Model to test
model="happyquoka4"

# Create a results file
echo "Batch Size,Status,Time (seconds)" > batch_size_results.txt

# Test each batch size
for bs in "${batch_sizes[@]}"; do
    echo "Testing batch size: $bs"
    
    # Run the training with timing
    { time python train_model.py --model $model --batch_size $bs --loss "pearson_l1" --test-mode; } 2> timing_output.txt
    
    # Check if the run was successful
    if [ $? -eq 0 ]; then
        status="Success"
    else
        status="Failed (OOM)"
    fi
    
    # Extract the real time in seconds
    time_taken=$(grep "real" timing_output.txt | awk '{print $2}' | sed 's/m/ * 60 + /g' | sed 's/s//g' | bc)
    
    # Append to results
    echo "$bs,$status,$time_taken" >> batch_size_results.txt
    
    echo "Completed batch size $bs: $status in $time_taken seconds"
    echo "-------------------------------------------"
    
    # Give the GPU a moment to cool down and clear memory
    sleep 5
done

echo "All tests completed. Results saved to batch_size_results.txt"