import os
import torch
import wandb
import numpy as np
from typing import Optional, Dict, Any, Callable
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from .optimizer_schedule import OptimizerScheduler
from .weighted_losses import WeightedPearsonLoss
from .loss_factory import LossFactory
import time

class EnvelopeTrainer:
    """Trainer class for envelope decoding models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        experiment_name: str,
        checkpoint_dir: str = "checkpoints",
        subject_id: Optional[str] = None,
        wandb_project: str = "eeg_envelope_decoding",
        wandb_tags: Optional[list] = None,
        resume_checkpoint: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            config: Training configuration
            experiment_name: Name for the experiment
            checkpoint_dir: Directory to save checkpoints
            subject_id: Subject ID for subject-dependent training
            wandb_project: W&B project name
            wandb_tags: List of tags for wandb run
            resume_checkpoint: Path to checkpoint to resume from
        """
        # Set random seeds for reproducibility
        seed = config.get('seed', 42)  # Get seed from config or use default
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Set to False for better performance
        np.random.seed(seed)
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.mse_loss_per_sample = LossFactory.create('mse', reduction='none')
        self.corrcoef_per_sample = LossFactory.create('pearson', reduction='none', as_metric=True)
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.subject_id = subject_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_tags = wandb_tags if wandb_tags is not None else []
        
        # Initialize scheduler based on model type and configuration
        self.needs_transformer_schedule = self.config.get('needs_transformer_schedule', False)

        # Special handling for two-branch models
        if "two_branch" in experiment_name.lower():
            # Check if any branch is being trained from scratch
            trains_from_scratch = not any(x in experiment_name.lower() for x in ['frozen', 'finetune'])
            trains_env_from_scratch = 'frozen_eeg' in experiment_name.lower() or 'finetune_eeg' in experiment_name.lower()
            trains_eeg_from_scratch = 'frozen_env' in experiment_name.lower() or 'finetune_env' in experiment_name.lower()
            
            #self.needs_transformer_schedule = trains_from_scratch or trains_env_from_scratch or trains_eeg_from_scratch
            
        print(f"Using transformer schedule: {self.needs_transformer_schedule} for {experiment_name}")
        
        if self.needs_transformer_schedule:
            self.scheduler = OptimizerScheduler(
                optimizer=self.optimizer,
                scheduler_type='transformer',
                config=config,
                num_epochs=config.get('num_epochs', 100),
                num_training_steps=len(train_loader) * config.get('num_epochs', 100)
            )
        else:
            print("Using static learning rate: ", config.get('learning_rate', 1e-3))
            self.scheduler = None
        
        # Initialize wandb with tags
        self.run = wandb.init(
            project=wandb_project,
            name=f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
            tags=self.wandb_tags,
            resume=True if resume_checkpoint else False
        )
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_corr = float('-inf')  # Track best validation correlation
        self.patience_counter = 0
        self.train_mode = "subject_independent" if subject_id is None else "subject_dependent"
        
        # Add global step counter
        self.global_step = 0
        
        # Load checkpoint if provided
        if resume_checkpoint:
            self._load_checkpoint(resume_checkpoint)
            
        # Log model architecture
        wandb.watch(self.model)

    def save_gt_inputs_outputs(self, metadata, eegs, envelopes, envelope_context):
        """Save the ground truth inputs and outputs."""
        import matplotlib.pyplot as plt
        import numpy as np

        # Create a grid of 4 subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))

        print(f"Envelopes shape: {envelopes.shape}")
        print(f"Envelope context shape: {envelope_context.shape}")
        print(f"EEG shape: {eegs.shape}")
        
        # Plot EEG
        eeg_data = eegs[0].cpu().numpy()
        axes[0].plot(eeg_data[:, 0])
        axes[0].set_title('EEG channel 0 and sample 0')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Amplitude')
        
        # Plot envelope
        envelope_data = np.concatenate((envelope_context[0].cpu().numpy(), envelopes[0].cpu().numpy()), axis=0)
        axes[1].plot(envelope_data[:, 0])
        axes[1].plot(envelope_context[0].cpu().numpy()[:, 0], color='red')
        axes[1].set_title('Envelope channel 0 and sample 0')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Amplitude')
        
        # Save the plot
        plt.savefig(f'{self.checkpoint_dir}/gt_inputs_outputs.png')

    
    def save_model_outputs(self, envelopes, outputs, epoch=None):
        """Save the model outputs."""

        # Create a grid of 4 subplots for each of the first 5 samples
        fig, axes = plt.subplots(5, 4, figsize=(20, 25))

        for i in range(5):
            # plot the ground truth envelope
            envelope_data = envelopes[i].cpu().numpy()
            axes[i, 0].plot(envelope_data[:, 0])
            axes[i, 0].set_title(f'Envelope channel 0 and sample {i}')
            axes[i, 0].set_xlabel('Time')
            axes[i, 0].set_ylabel('Amplitude')

            # plot the model output
            envelope_pred = outputs['envelope'][i].cpu().numpy()
            axes[i, 1].plot(envelope_pred[:, 0])
            axes[i, 1].set_title(f'Model output envelope channel 0 and sample {i}')
            axes[i, 1].set_xlabel('Time')
            axes[i, 1].set_ylabel('Amplitude')

            # if the outputs have an envelope branch, plot it
            if 'envelope_branch' in outputs:
                envelope_pred_branch = outputs['envelope_branch'][i].cpu().numpy()
                axes[i, 2].plot(envelope_pred_branch[:, 0])
                axes[i, 2].set_title(f'Envelope branch output sample {i}')
                axes[i, 2].set_xlabel('Time')
                axes[i, 2].set_ylabel('Amplitude')

            # if the outputs have an eeg branch, plot it
            if 'eeg_branch' in outputs:
                eeg_pred_branch = outputs['eeg_branch'][i].cpu().numpy()
                axes[i, 3].plot(eeg_pred_branch[:, 0])
                axes[i, 3].set_title(f'EEG branch output sample {i}')
                axes[i, 3].set_xlabel('Time')
                axes[i, 3].set_ylabel('Amplitude')

        # save the plot
        plt.tight_layout()
        plt.savefig(f'{self.checkpoint_dir}/model_outputs_epoch_{epoch}.png')
        

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_corr = 0  # Pearson correlation
        total_envelope_branch_mse = 0
        total_envelope_branch_corr = 0
        total_eeg_branch_mse = 0
        total_eeg_branch_corr = 0
        num_batches = 0
        time_start = time.time()
        total_batch_time = 0  # Track total time spent on batches

        if self.config.get('is_recursive', False):
            # Use epoch-based teacher forcing schedule
            if self.epoch < 3:
                teacher_forcing_ratio = 1.0  # 90% for first 3 epochs
            else:
                teacher_forcing_ratio = 1.0  # 40% after epoch 3
        else:
            teacher_forcing_ratio = 0.0

        for batch_idx, (metadata, eegs, envelopes, eeg_context, envelope_context) in enumerate(self.train_loader):
            batch_start_time = time.time()  # Start timing this batch

            if not os.path.exists(f'{self.checkpoint_dir}/gt_inputs_outputs.png'):
                self.save_gt_inputs_outputs(metadata, eegs, envelopes, envelope_context)

            # Move data to device
            eegs = eegs.to(self.device)
            envelopes = envelopes.to(self.device)
            envelope_context = envelope_context.to(self.device)
            eeg_context = eeg_context.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            if self.config.get('is_contextual', False):
                if self.config.get('is_recursive', False):
                    outputs = self.model(eegs, eeg_context, envelope_context_gt=envelope_context, teacher_forcing_ratio=teacher_forcing_ratio)
                else:   
                    outputs = self.model(eegs, envelope_context)
                envelope_pred = outputs['envelope']
                loss = self.criterion(envelope_pred, envelopes)
                mse = self.mse_loss_per_sample(envelope_pred, envelopes).mean().item()
                corr = self.corrcoef_per_sample(envelope_pred, envelopes).mean().item()

                # track the performance of the two branches if they exist
                if 'envelope_branch' in outputs and 'eeg_branch' in outputs:
                    total_envelope_branch_mse += self.mse_loss_per_sample(outputs['envelope_branch'], envelopes).mean().item()
                    total_eeg_branch_mse += self.mse_loss_per_sample(outputs['eeg_branch'], envelopes).mean().item()

                    total_envelope_branch_corr += self.corrcoef_per_sample(outputs['envelope_branch'], envelopes).mean().item()
                    total_eeg_branch_corr += self.corrcoef_per_sample(outputs['eeg_branch'], envelopes).mean().item()
            else:
                outputs = self.model(eegs)
                envelope_pred = outputs['envelope']
                loss = self.criterion(envelope_pred, envelopes)
                mse = self.mse_loss_per_sample(envelope_pred, envelopes).mean().item()
                corr = self.corrcoef_per_sample(envelope_pred, envelopes).mean().item()

            # Backward pass
            loss.backward()
            
            # Optimizer step and scheduler step (if transformer)
            self.optimizer.step()
            if self.needs_transformer_schedule:
                self.scheduler.step()
            
            # Increment global step
            self.global_step += 1
            
            # Calculate batch time and iterations per second
            batch_time = time.time() - batch_start_time
            iterations_per_second = 1 / batch_time if batch_time > 0 else float('inf')
            total_batch_time += batch_time
            
            # Update metrics
            total_loss += loss.item()
            total_corr += corr
            total_mse += mse
            num_batches += 1
            
            # Log batch metrics
            if batch_idx % self.config['log_interval'] == 0:
                # Calculate learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Base metrics (maintain existing naming)
                log_dict = {
                    'train/batch_loss': loss.item(),
                    'train/batch_correlation': corr,
                    'train/batch_mse': mse,
                    'train/epoch': self.epoch,
                    'train/batch': batch_idx,
                    'train/learning_rate': current_lr,
                    'train/batch_subjects': metadata['sub'],
                    'train/batch_audio_files': metadata['audio_file'],
                    'train/running_avg_loss': total_loss / num_batches,
                    'train/running_avg_correlation': total_corr / num_batches,
                    'train/running_avg_mse': total_mse / num_batches,
                    'train/iterations_per_second': iterations_per_second,
                    'train/teacher_forcing_ratio': teacher_forcing_ratio,
                }

                if 'envelope_branch' in outputs and 'eeg_branch' in outputs:
                    log_dict['train/envelope_branch_mse'] = total_envelope_branch_mse / num_batches
                    log_dict['train/eeg_branch_mse'] = total_eeg_branch_mse / num_batches
                    log_dict['train/envelope_branch_corr'] = total_envelope_branch_corr / num_batches
                    log_dict['train/eeg_branch_corr'] = total_eeg_branch_corr / num_batches
                
                # Log to wandb
                wandb.log(log_dict, step=self.global_step)
                
                # Print progress
                progress_str = f"\rEp {self.epoch+1} | B {batch_idx}/{len(self.train_loader)} | "
                progress_str += f"Loss: {loss.item():.4f} | MSE: {mse:.4f} | Corr: {corr:.4f}"
                progress_str += f" | LR: {current_lr:.10f} | Iter/s: {iterations_per_second:.2f}" 
                if 'envelope_branch' in outputs and 'eeg_branch' in outputs:
                    progress_str += f" | Env MSE: {total_envelope_branch_mse / num_batches:.4f} | Env Corr: {total_envelope_branch_corr / num_batches:.4f}"
                    progress_str += f" | EEG MSE: {total_eeg_branch_mse / num_batches:.4f} | EEG Corr: {total_eeg_branch_corr / num_batches:.4f}"
                print(progress_str, end="")
        
        print()  # New line after progress
        
        # Compute epoch metrics
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_correlation': total_corr / num_batches,
            'train_mse': total_mse / num_batches,
            'train_avg_batch_time': total_batch_time / num_batches  # Add average batch time to metrics
        }

        if 'envelope_branch' in outputs and 'eeg_branch' in outputs:
            metrics['train_envelope_branch_mse'] = total_envelope_branch_mse / num_batches
            metrics['train_eeg_branch_mse'] = total_eeg_branch_mse / num_batches
            metrics['train_envelope_branch_corr'] = total_envelope_branch_corr / num_batches
            metrics['train_eeg_branch_corr'] = total_eeg_branch_corr / num_batches
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model on validation set."""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_corr = 0
        num_batches = 0
        total_envelope_branch_mse = 0
        total_envelope_branch_corr = 0
        total_eeg_branch_mse = 0
        total_eeg_branch_corr = 0
        
        with torch.no_grad():
            for batch_idx, (metadata, eegs, envelopes, eeg_context, envelope_context) in enumerate(self.val_loader):
                # Move data to device
                eegs = eegs.to(self.device)
                envelopes = envelopes.to(self.device)
                envelope_context = envelope_context.to(self.device)
                eeg_context = eeg_context.to(self.device)
                
                
                # Forward pass
                if self.config.get('is_contextual', False):
                    if self.config.get('is_recursive', False):
                        outputs = self.model.forward_teacher_forcing(eegs, envelope_context_gt=envelope_context)
                    else:
                        outputs = self.model(eegs, envelope_context)
                    envelope_pred = outputs['envelope']
                    loss = self.criterion(envelope_pred, envelopes)
                    mse = self.mse_loss_per_sample(envelope_pred, envelopes).mean().item()
                    corr = self.corrcoef_per_sample(envelope_pred, envelopes).mean().item()

                    # track the performance of the two branches if they exist
                    if 'envelope_branch' in outputs and 'eeg_branch' in outputs:
                        total_envelope_branch_mse += self.mse_loss_per_sample(outputs['envelope_branch'], envelopes).mean().item()
                        total_eeg_branch_mse += self.mse_loss_per_sample(outputs['eeg_branch'], envelopes).mean().item()

                        total_envelope_branch_corr += self.corrcoef_per_sample(outputs['envelope_branch'], envelopes).mean().item()
                        total_eeg_branch_corr += self.corrcoef_per_sample(outputs['eeg_branch'], envelopes).mean().item()
                        
                else:
                    outputs = self.model(eegs)
                    envelope_pred = outputs['envelope']
                    loss = self.criterion(envelope_pred, envelopes)
                    mse = self.mse_loss_per_sample(envelope_pred, envelopes).mean().item()
                    corr = self.corrcoef_per_sample(envelope_pred, envelopes).mean().item()
                
                # Update metrics
                total_loss += loss.item()
                total_mse += mse
                total_corr += corr
                num_batches += 1

                # progress bar
                progress_str = f"\rEpoch {self.epoch+1} | Batch {batch_idx}/{len(self.val_loader)} | "
                progress_str += f"Batch Loss: {loss.item():.4f} | Batch MSE: {mse:.4f} | Batch Corr: {corr:.4f}"
                print(progress_str, end="")

                if not os.path.exists(f'{self.checkpoint_dir}/model_outputs_epoch_{self.epoch}.png'):
                    self.save_model_outputs(envelopes, outputs, epoch=self.epoch)
        
        # Compute validation metrics
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_correlation': total_corr / num_batches,
            'val_mse': total_mse / num_batches
        }

        if 'envelope_branch' in outputs and 'eeg_branch' in outputs:
            metrics['val_envelope_branch_mse'] = total_envelope_branch_mse / num_batches
            metrics['val_eeg_branch_mse'] = total_eeg_branch_mse / num_batches
            metrics['val_envelope_branch_corr'] = total_envelope_branch_corr / num_batches
            metrics['val_eeg_branch_corr'] = total_eeg_branch_corr / num_batches
        
        return metrics
    
    def train(self, num_epochs: int):
        """Train the model for specified number of epochs."""
        best_epoch = 0
        start_time = datetime.now()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_start_time = datetime.now()

            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()

            metrics = {**train_metrics, **val_metrics}
            
            # Calculate epoch time
            epoch_time = datetime.now() - epoch_start_time
            total_time = datetime.now() - start_time
            
            # Log epoch-level metrics
            wandb.log({
                'epoch': epoch,
                'epoch_time_seconds': epoch_time.total_seconds(),
                'total_time_hours': total_time.total_seconds() / 3600,
                'best_val_corr': self.best_val_corr,
                'best_val_loss': self.best_val_loss,
                'patience_counter': self.patience_counter,
                'train/epoch_loss': train_metrics['train_loss'],
                'train/epoch_correlation': train_metrics['train_correlation'],
                'val/epoch_loss': val_metrics['val_loss'],
                'val/epoch_correlation': val_metrics['val_correlation']
            }, step=self.global_step)
            
            # Save checkpoint if best model (based on correlation)
            current_val_corr = val_metrics['val_correlation']
            if current_val_corr > self.best_val_corr:
                self.best_val_corr = current_val_corr
                self.best_val_loss = val_metrics['val_loss']  # Update best loss too
                self.patience_counter = 0
                best_epoch = epoch
                self._save_checkpoint(metrics, is_best=True)
            else:
                self.patience_counter += 1

            # Regular checkpoint
            #if epoch % self.config['checkpoint_interval'] == 0:
                #self._save_checkpoint(metrics, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation correlation {self.best_val_corr:.4f} achieved at epoch {best_epoch + 1}")
                break
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
            print(f"Time: {epoch_time.total_seconds():.2f}s | Total: {total_time.total_seconds()/3600:.2f}h")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")
            print("-" * 80)
    
    def _save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save a checkpoint of the model."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_corr': self.best_val_corr,  # Save best correlation
            'metrics': metrics,
            'config': self.config,
            'patience_counter': self.patience_counter,
            'global_step': self.global_step
        }
        
        if self.needs_transformer_schedule and self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / f"best_model.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch}_{self.train_mode}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        
        #if is_best:
        #    wandb.save(str(checkpoint_path))
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_corr = checkpoint.get('best_val_corr', float('-inf'))  # Load best correlation with default
        self.patience_counter = checkpoint.get('patience_counter', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        if self.needs_transformer_schedule and self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def fine_tune(self, num_epochs: int, subject_id: str):
        """Fine-tune a subject-independent model for a specific subject."""
        assert self.train_mode == "subject_independent", \
            "Can only fine-tune a subject-independent model"
        
        # Save original training state
        original_state = {
            'train_loader': self.train_loader,
            'val_loader': self.val_loader,
            'subject_id': self.subject_id,
            'best_val_loss': self.best_val_loss,
            'epoch': self.epoch,
            'train_mode': self.train_mode
        }
        
        # Create subject-specific dataloaders
        from src.data.eeg_dataloader import EEGDataset, StandardizeEEG, StandardizeEnvelope
        
        # Get the original dataset configurations
        train_dataset = self.train_loader.dataset
        val_dataset = self.val_loader.dataset
        
        # Create new subject-specific datasets
        train_dataset_subject = EEGDataset(
            filelist_path=train_dataset.df.iloc[0]['paths_eeg'].split('_-_')[0] + '.csv',  # Extract original filelist path
            base_dir=train_dataset.base_dir,
            subject_id=subject_id,
            transform_eeg=train_dataset.transform_eeg,
            transform_envelope=train_dataset.transform_envelope
        )
        
        val_dataset_subject = EEGDataset(
            filelist_path=val_dataset.df.iloc[0]['paths_eeg'].split('_-_')[0] + '.csv',  # Extract original filelist path
            base_dir=val_dataset.base_dir,
            subject_id=subject_id,
            transform_eeg=val_dataset.transform_eeg,
            transform_envelope=val_dataset.transform_envelope
        )
        
        # Create new dataloaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset_subject,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=self.train_loader.num_workers,
            pin_memory=True
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset_subject,
            batch_size=self.val_loader.batch_size,
            shuffle=False,
            num_workers=self.val_loader.num_workers,
            pin_memory=True
        )
        
        # Update trainer state for fine-tuning
        self.subject_id = subject_id
        self.train_mode = "subject_dependent"
        self.best_val_loss = float('inf')
        self.epoch = 0
        
        print(f"\nFine-tuning for subject {subject_id}")
        print(f"Training samples: {len(train_dataset_subject)}")
        print(f"Validation samples: {len(val_dataset_subject)}")
        
        # Train for specified epochs
        self.train(num_epochs)
        
        # Restore original state
        self.__dict__.update(original_state)
    
    def __del__(self):
        """Cleanup wandb run on deletion."""
        if hasattr(self, 'run'):
            self.run.finish()
    
    def test(self, teacher_forcing: bool = False) -> pd.DataFrame:
        """Test the model and return results DataFrame."""
        self.model.eval()
        results = []
        total_mse = 0
        total_corr = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (metadata, eegs, envelopes, eeg_context, envelope_context) in enumerate(self.test_loader):
                # Move data to device
                eegs = eegs.to(self.device)
                envelopes = envelopes.to(self.device)
                envelope_context = envelope_context.to(self.device)
                eeg_context = eeg_context.to(self.device)
                
                
                # Forward pass
                if self.config.get('is_contextual', False):
                    if self.config.get('is_recursive', False):
                        print("Recursive model")
                        if teacher_forcing:
                            print("Teacher forcing")
                            outputs = self.model.forward_teacher_forcing(eegs, envelope_context_gt=envelope_context)
                        else:
                            print("No teacher forcing")
                            outputs = self.model(eegs, eeg_context)
                    else:
                        print("Context model")
                        outputs = self.model(eegs, envelope_context)
                    envelope_pred = outputs['envelope']
                    envelope_branch_pred = outputs['envelope_branch']
                    eeg_branch_pred = outputs['eeg_branch']
                    batch_mse = self.mse_loss_per_sample(envelope_pred, envelopes)
                    batch_corr = self.corrcoef_per_sample(envelope_pred, envelopes)
                    batch_envelope_branch_corr = self.corrcoef_per_sample(envelope_branch_pred, envelopes)
                    batch_eeg_branch_corr = self.corrcoef_per_sample(eeg_branch_pred, envelopes)
                else:
                    outputs = self.model(eegs)
                    envelope_pred = outputs['envelope']
                    envelope_branch_pred = outputs['envelope_branch']
                    eeg_branch_pred = outputs['eeg_branch']
                    batch_mse = self.mse_loss_per_sample(envelope_pred, envelopes)
                    batch_corr = self.corrcoef_per_sample(envelope_pred, envelopes)
                    batch_envelope_branch_corr = self.corrcoef_per_sample(envelope_branch_pred, envelopes)
                    batch_eeg_branch_corr = self.corrcoef_per_sample(eeg_branch_pred, envelopes)

                # Store results
                for i in range(len(metadata['sub'])):
                    print(batch_envelope_branch_corr[i].item())
                    print(batch_eeg_branch_corr[i].item())
                    results.append({
                        'sub': metadata['sub'][i],
                        'audio_file': metadata['audio_file'][i],
                        'start': metadata['start'][i],
                        'end': metadata['end'][i],
                        'mse': batch_mse[i].item(),
                        'correlation': batch_corr[i].item(),
                        'envelope_branch_correlation': batch_envelope_branch_corr[i].item(),
                        'eeg_branch_correlation': batch_eeg_branch_corr[i].item()
                    })
                
                total_mse += batch_mse.mean().item()
                total_corr += batch_corr.mean().item()
                num_batches += 1
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Log final test metrics
        wandb.log({
            'test/final_correlation': results_df['correlation'].mean(),
            'test/final_correlation_std': results_df['correlation'].std(),
            'test/final_mse': results_df['mse'].mean(),
            'test/final_mse_std': results_df['mse'].std(),
            'test/subjects_count': results_df['sub'].nunique(),
            'test/audio_files_count': results_df['audio_file'].nunique(),
            'test/total_samples': len(results_df)
        })
        
        # Create and log correlation distribution plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=results_df, x='correlation', bins=30, ax=ax)
        ax.set_title('Distribution of Test Correlations')
        ax.set_xlabel('Correlation')
        ax.set_ylabel('Count')
        wandb.log({'test/correlation_distribution': wandb.Image(fig)})
        plt.close(fig)
        
        return results_df
    
    def test_recursive(self) -> pd.DataFrame:
        """Test the model and return results DataFrame."""
        self.model.eval()
        results = []
        total_mse = 0
        total_corr = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (metadata, eegs, envelopes, eeg_context, envelope_context) in enumerate(self.test_loader):
                # Move data to device
                eegs = eegs.to(self.device)
                envelopes = envelopes.to(self.device)
                envelope_context = envelope_context.to(self.device)
                eeg_context = eeg_context.to(self.device)
                
                # Forward pass
                if self.config.get('is_contextual', False):
                    if self.config.get('is_recursive', False):
                        outputs = self.model.forward_recursive(eegs, eeg_context)
                    else:
                        outputs = self.model.forward_recursive(eegs, envelope_context)
                    envelope_pred = outputs['envelope']
                    batch_mse = self.mse_loss_per_sample(envelope_pred, envelopes)
                    batch_corr = self.corrcoef_per_sample(envelope_pred, envelopes)
                    
                else:
                    outputs = self.model(eegs)
                    envelope_pred = outputs['envelope']
                    batch_mse = self.mse_loss_per_sample(envelope_pred, envelopes)
                    batch_corr = self.corrcoef_per_sample(envelope_pred, envelopes)

                # Store results
                for i in range(len(metadata['sub'])):
                    results.append({
                        'sub': metadata['sub'][i],
                        'audio_file': metadata['audio_file'][i],
                        'start': metadata['present_start'][i],
                        'end': metadata['present_end'][i],
                        'mse': batch_mse[i].item(),
                        'correlation': batch_corr[i].item()
                    })
                
                total_mse += batch_mse.mean().item()
                total_corr += batch_corr.mean().item()
                num_batches += 1
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Log final test metrics
        wandb.log({
            'test/final_correlation': results_df['correlation'].mean(),
            'test/final_correlation_std': results_df['correlation'].std(),
            'test/final_mse': results_df['mse'].mean(),
            'test/final_mse_std': results_df['mse'].std(),
            'test/subjects_count': results_df['sub'].nunique(),
            'test/audio_files_count': results_df['audio_file'].nunique(),
            'test/total_samples': len(results_df)
        })
        
        # Create and log correlation distribution plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=results_df, x='correlation', bins=30, ax=ax)
        ax.set_title('Distribution of Test Correlations')
        ax.set_xlabel('Correlation')
        ax.set_ylabel('Count')
        wandb.log({'test/correlation_distribution': wandb.Image(fig)})
        plt.close(fig)
        
        return results_df