#!/bin/bash

# Increase file descriptor limits
increase_file_limits() {
    # Get current limits
    curr_soft=$(ulimit -Sn)
    curr_hard=$(ulimit -Hn)
    
    echo "Current file descriptor limits: soft=$curr_soft, hard=$curr_hard"
    
    # Try to increase soft limit
    if [ "$curr_soft" -lt 65535 ]; then
        ulimit -Sn 65535 2>/dev/null || ulimit -Sn "$curr_hard" 2>/dev/null || echo "Could not increase soft limit"
    fi
    
    # Print new limits
    echo "New file descriptor limits: soft=$(ulimit -Sn), hard=$(ulimit -Hn)"
}

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    # Run Python cleanup
    python3 -c "
from configs.baseline_config import BaselineConfig
BaselineConfig.cleanup_wandb()
"
    # Clear Python cache
    find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
    
    # Sleep to allow system to release resources
    sleep 2
    
    echo "Cleanup completed"
}

# Default values
TEST_MODE=false
TEST_SUBJECT="sub-001"  # Default test subject

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        --test-subject)
            TEST_SUBJECT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test] [--test-subject SUB_ID]"
            exit 1
            ;;
    esac
done

# Array of models and loss functions
models=("vlaaiquoka2")  # Added vlaaiquoka
losses=("mse" "pearson")

# Function to train a model with specific loss
train_model() {
    local model=$1
    local loss=$2
    
    echo "============================================================"
    echo "Training $model with $loss loss"
    if [ "$TEST_MODE" = true ]; then
        echo "TEST MODE: Using reduced epochs and single subject"
    fi
    echo "============================================================"
    
    # Increase file limits before each training (especially important for vlaaiquoka)
    increase_file_limits
    
    # Base command
    cmd="python train_and_finetune.py --model \"$model\" --loss-function \"$loss\""
    
    # Add test mode parameters if enabled
    if [ "$TEST_MODE" = true ]; then
        cmd="$cmd --test-mode --subjects \"$TEST_SUBJECT\""
    fi
    
    # Execute the command
    eval $cmd
    
    # Store the exit status
    local status=$?
    
    # Run cleanup regardless of success/failure
    cleanup
    
    # Check if the training was successful
    if [ $status -eq 0 ]; then
        echo "Successfully completed training $model with $loss loss"
        return 0
    else
        echo "Error training $model with $loss loss"
        echo "Moving on to next model..."
        return 1
    fi
}

# Main execution
echo "Starting training of all models..."

# Initial file descriptor increase
increase_file_limits

if [ "$TEST_MODE" = true ]; then
    echo "Running in TEST MODE"
    echo "Using subject: $TEST_SUBJECT"
fi
echo "Models to train: ${models[*]}"
echo "Loss functions: ${losses[*]}"
echo ""

# Initialize counters for summary
total_combinations=$((${#models[@]} * ${#losses[@]}))
successful_trainings=0
failed_trainings=0

# Train each combination sequentially
for model in "${models[@]}"; do
    for loss in "${losses[@]}"; do
        if train_model "$model" "$loss"; then
            ((successful_trainings++))
        else
            ((failed_trainings++))
        fi
    done
done

# Print summary
echo "Training Summary:"
echo "----------------"
if [ "$TEST_MODE" = true ]; then
    echo "TEST MODE Summary:"
fi
echo "Total combinations attempted: $total_combinations"
echo "Successful trainings: $successful_trainings"
echo "Failed trainings: $failed_trainings"

if [ $failed_trainings -eq 0 ]; then
    echo "All trainings completed successfully!"
    exit 0
else
    echo "Some trainings failed. Check the logs above for details."
    exit 1
fi 