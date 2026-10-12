#!/usr/bin/env python3

import wandb
import argparse
from tqdm import tqdm
import time
from requests.exceptions import HTTPError

def delete_with_retry(file, max_retries=3, retry_delay=5):
    """Try to delete a file with retries on network errors."""
    for attempt in range(max_retries):
        try:
            file.delete()
            return True
        except HTTPError as e:
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                print(f"Network error deleting {file.name}, attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to delete {file.name} after {max_retries} attempts: {str(e)}")
                return False
        except Exception as e:
            print(f"Error deleting {file.name}: {str(e)}")
            return False

def get_run_with_retry(api, path, max_retries=3, retry_delay=5):
    """Try to get a run with retries on network errors."""
    for attempt in range(max_retries):
        try:
            return api.run(path)
        except HTTPError as e:
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                print(f"Network error getting run {path}, attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to get run {path} after {max_retries} attempts: {str(e)}")
                raise
        except Exception as e:
            print(f"Error getting run {path}: {str(e)}")
            raise

def cleanup_checkpoints(project_name: str, dry_run: bool = True):
    """
    Clean up regular checkpoints from wandb runs while keeping best_model.pt files.
    
    Args:
        project_name: Name of the wandb project
        dry_run: If True, only print what would be deleted without actually deleting
    """
    # Initialize wandb API
    api = wandb.Api()
    entity = api.default_entity
    
    # Get all runs from the project
    try:
        runs = api.runs(f"{entity}/{project_name}")
    except Exception as e:
        print(f"Error getting runs from project: {str(e)}")
        return
    
    print(f"\nFound {len(runs)} runs in project {project_name}")
    
    # Track statistics
    total_files = 0
    deleted_files = 0
    saved_space = 0  # in bytes
    failed_deletions = 0
    
    # Process each run
    for run in tqdm(runs, desc="Processing runs"):
        try:
            # Get the run directly using the API with retry
            run = get_run_with_retry(api, f"{entity}/{project_name}/{run.id}")
            
            # Get all files in the run
            try:
                files = run.files()
            except Exception as e:
                print(f"Error getting files for run {run.name}: {str(e)}")
                continue
            
            # Process each file
            for file in files:
                total_files += 1
                
                # Check if it's a regular checkpoint (not best_model.pt)
                if file.name.endswith('.pt') and 'checkpoint_epoch_' in file.name:
                    try:
                        file_size = file.size
                    except Exception as e:
                        print(f"Error getting size for {file.name}: {str(e)}")
                        continue
                    
                    if dry_run:
                        print(f"Would delete: {run.name}/{file.name} (Size: {file_size/1024/1024:.2f} MB)")
                    else:
                        if delete_with_retry(file):
                            deleted_files += 1
                            saved_space += file_size
                            print(f"Deleted: {run.name}/{file.name} (Size: {file_size/1024/1024:.2f} MB)")
                        else:
                            failed_deletions += 1
                
        except Exception as e:
            print(f"Error processing run {run.name}: {str(e)}")
            continue
    
    # Print summary
    print("\nCleanup Summary:")
    print(f"Total files processed: {total_files}")
    print(f"Files {'to be deleted' if dry_run else 'deleted'}: {deleted_files}")
    print(f"Failed deletions: {failed_deletions}")
    print(f"Space {'to be saved' if dry_run else 'saved'}: {saved_space/1024/1024:.2f} MB")
    
    if dry_run:
        print("\nThis was a dry run. No files were actually deleted.")
        print("Run with --force to perform the actual deletion.")

def main():
    parser = argparse.ArgumentParser(description="Clean up regular checkpoints from wandb while keeping best_model.pt files")
    parser.add_argument("--project", type=str, default="ssl_eeg_envelope",
                      help="Name of the wandb project")
    parser.add_argument("--force", action="store_true",
                      help="Actually delete the files. Without this flag, will only show what would be deleted")
    
    args = parser.parse_args()
    
    # Run cleanup
    cleanup_checkpoints(args.project, dry_run=not args.force)

if __name__ == "__main__":
    main() 