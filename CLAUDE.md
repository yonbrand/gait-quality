# CLAUDE.md - ElderNet Gait Quality Estimation

## Project Overview

ElderNet is a deep learning system for gait quality estimation from wrist-worn accelerometer data, optimized for older adults. It uses self-supervised learning (SSL) pre-trained on UK Biobank data and fine-tuned on RUSH Memory and Aging Project data.

**Gait metrics estimated:**
- Gait Speed (0-2 m/s)
- Stride Length (0-2 m)
- Cadence (0-160 steps/min)
- Stride Regularity (0-1)

## Project Structure

```
conf/                          # Hydra configuration files
├── augmentation/all.yaml      # Data augmentation settings
├── dataloader/                # Data loading configs (five_sec, ten_sec)
├── model/                     # Model configs (ElderNet, resnet, unet)
├── config_hyp.yaml            # Hyperparameter tuning config
├── config_final_training.yaml # Final training config
├── config_inference.yaml      # Inference/testing config
└── models_config.txt          # Best model configs & weight paths

data_parsing/                  # Data preprocessing
└── MOBILISE_D_parsing.py      # Mobilise-D dataset preparation

RUSH/                          # Daily living analysis
└── rush_pipeline.py           # RUSH dataset pipeline

weights/                       # Pre-trained model weights
├── ssl_ukb_weights.mdl        # UK Biobank SSL pre-trained
├── ssl_eldernet_weights.mdl   # ElderNet SSL pre-trained
├── gait_speed_weights.pt      # Fine-tuned models
├── gait_length_weights.pt
├── cadence_weights.pt
├── gait_detection_weights.pt
└── regularity_weights.pt

models.py                      # Neural network architectures
utils.py                       # Utilities, datasets, training helpers
regularity.py                  # Gait regularity calculation
hyperparameter_tuning.py       # Optuna-based hyperparameter optimization
final_training.py              # Model training with best hyperparameters
inference.py                   # Model testing/inference
```

## Commands

### Training Pipeline

```bash
# 1. Hyperparameter tuning
python hyperparameter_tuning.py

# 2. Final training with best config
python final_training.py

# 3. Inference/testing
python inference.py
```

### Data Preparation

```bash
# Prepare Mobilise-D dataset
python data_parsing/MOBILISE_D_parsing.py
```

### Daily Living Analysis

```bash
# RUSH dataset analysis
python RUSH/rush_pipeline.py
```

### Hydra Configuration Overrides

```bash
python hyperparameter_tuning.py data.labels=gait_speed.p data.cohort=TVS
python final_training.py model.batch_size=256 model.lr=0.0001
python inference.py model.weights_path=outputs/gait_speed/final_model.pt
```

## Tech Stack

- **PyTorch 2.0.1** - Deep learning framework
- **Hydra 1.3.2** - Configuration management
- **Optuna 3.6.1** - Hyperparameter optimization
- **NumPy, SciPy, Pandas** - Scientific computing
- **scikit-learn** - ML utilities and metrics
- **Python 3.10**

## Code Conventions

### Naming
- `mtl` = multi-task learning, `ssl` = self-supervised learning, `fc` = fully connected
- Boolean flags: `is_regression`, `is_classification`, `is_mtl`, `is_simclr`, `is_dense`
- Hyperparameters: `lr` (learning rate), `wd` (weight decay)
- Data: `acc` (acceleration), `gait_speed`, `cadence`, `gait_length`, `regularity`

### Key Classes (models.py)
- `Resnet` - ResNet-18 feature extractor with configurable heads
- `ElderNet` - Wraps ResNet with additional FC layers for older adults
- `Regressor` - Multi-layer regression head (sigmoid output scaled by `max_mu`)
- `Classifier` - Binary/multi-class classification head
- `LinearLayers` - 3-layer FC network for feature transformation
- `ContrastiveLoss` - SimCLR-style contrastive loss for SSL

### Key Functions (utils.py)
- `load_data()` - Load and preprocess dataset
- `train_epoch()` / `val_epoch()` - Training/validation loops
- `EarlyStopping` - Early stopping callback
- Dataset classes: `AccDataset`, `MTLAccDataset`

### Architecture Patterns
- Transfer learning: SSL pre-trained models frozen during fine-tuning
- Stratified sampling by cohort across train/val splits
- 10-second windows (300 samples at 30 Hz) with 9-second overlap
- Sigmoid activation scaled to metric-specific ranges via `max_mu`

### Data Processing
- 30 Hz sampling rate
- Optional bandpass filtering (0.2-14 Hz)
- Axis rotation and switching for augmentation

## Configuration

### Model Config Example (from models_config.txt)

```yaml
gait_speed:
  net: ElderNet
  feature_vector_size: 128
  is_regression: True
  num_layers_regressor: 0
  batch_norm: False
  pretrained: True
  max_mu: 2.0  # Output range [0, 2.0] m/s

cadence:
  max_mu: 160  # Output range [0, 160] steps/min
  num_layers_regressor: 1
  batch_norm: True
```

### Training Config Defaults
- Epochs: 100
- Warmup epochs: 5
- Early stopping patience: 5
- Window: 10 seconds (300 samples)
- Overlap: 9 seconds

## Workflow

1. **Data Preparation** - `MOBILISE_D_parsing.py` converts raw data to 10-sec windows
2. **Hyperparameter Tuning** - `hyperparameter_tuning.py` runs Optuna with k-fold CV
3. **Final Training** - `final_training.py` trains on full data with best params
4. **Inference** - `inference.py` evaluates on test data
5. **Daily Living** - `rush_pipeline.py` analyzes real-world accelerometer data

## License

University of Oxford Academic Use License - restricted to academic, non-commercial research. Based on [ssl-wearables](https://github.com/OxWearables/ssl-wearables).
