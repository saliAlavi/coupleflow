# DECAF: Deep Learning for EEG-based Attention-Aware Decoder

Dynamic Envelope Context Aware Fusion for Envelope decoding from EEG.

A deep learning framework for attention-aware decoding using EEG and speech envelope data.

## 📊 Model Checkpoints & Data

Large model checkpoints and training data are hosted on Hugging Face:

**[→ DECAF Models & Data Dataset](https://huggingface.co/datasets/carankt/DECAF-models-and-data)**

### Quick Download

```bash
# Clone the HF dataset
git clone https://huggingface.co/datasets/carankt/DECAF-models-and-data checkpoints_hf

# Or use huggingface-hub
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='carankt/DECAF-models-and-data', repo_type='dataset', local_dir='./checkpoints')"
```

**Available on HF:**
- `checkpoints/paper/` - Models from the main paper
- `checkpoints/baseline_models/` - Baseline variants
- `checkpoints/dynamic_gru/` - Two-branch dynamic GRU
- `checkpoints/dynamic_lstm/` - Two-branch dynamic LSTM
- `checkpoints/recursive_gru/` - Recursive GRU architectures
- `checkpoints/recursive_lstm/` - Recursive LSTM architectures
- `wandb_logs/` - Training logs

## Training

### Train All Models
```bash
bash train_all_models.sh
```

### Train Specific Model
```bash
python train_model.py --config configs/two_branch_base_config.py
```

### Fine-tune
```bash
python fine_tune_subjects.py --model-checkpoint checkpoints/paper/best_model.pt
```

## Models

- **BaselineDecoder** - Baseline single-branch
- **HappyQuoka** - Two-branch attention decoder
- **TwoBranchDynamicLSTM/GRU** - Dynamic branch weighting
- **VLAAI** - Vision-Language-Audio Integrated

## Requirements

```bash
pip install torch numpy scipy scikit-learn wandb
```

## Citation

```bibtex
@article{DECAF2024,
  title={DECAF: Deep Learning for EEG-based Attention-Aware Decoding},
  author={Thakkar, Karan and ...},
  journal={...},
  year={2024}
}
```

## License

See LICENSE file.
