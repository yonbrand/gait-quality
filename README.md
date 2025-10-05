# ElderNet: Gait Quality Estimation for Older Adults
![pipeline](/imgs/figure 1.png)
## Overview
This repository implements **ElderNet**, a deep learning model for gait quality estimation optimized for older adults, including those with impaired gait. ElderNet leverages self-supervised learning (SSL) and fine-tuning to detect gait metrics from wrist-worn accelerometer data. It is fine-tuned for four gait metrics:
- Gait speed
- Stride length
- Cadence
- Stride regularity

The model uses a pre-trained SSL ResNet architecture (trained on UK Biobank data) with additional fully connected layers fine-tuned on the RUSH Memory and Aging Project data, tailoring it for older adults. This repository provides code to fine-tune ElderNet using labelled Mobilise-D data.

For more details, see the paper:  
["Continuous Assessment of Daily-Living Gait Using Self-Supervised Learning of Wrist-Worn Accelerometer Data"](https://www.medrxiv.org/content/10.1101/2025.05.21.25328061v1)

## Repository Structure

- **`data_parsing/`**: Scripts for parsing and preparing data.
- **`weights/`**: Pre-trained models and configurations.
- **`RUSH/`**: Scripts for daily living analysis using the RUSH dataset.
- **`hyperparameter_tuning.py`**: Finds optimal hyperparameters for each gait metric model.
- **`final_training.py`**: Trains the model with the best configuration on the full training set.
- **`inference.py`**: Infers gait quality using trained models.
- **`models_config.txt`**: Lists best configurations and paths to model weights for each gait metric.

## Preparing the Mobilise-D Data

1. Download participant data from [Zenodo](https://zenodo.org/records/13899386).
2. Refer to `participant_information.xlsx` for demographic details.
3. Included participant IDs (with complete sensor and reference data):  
   - **CHF**: 2014, 4023, 4025, 4028, 4029, 5003, 5008, 5012, 5015, 5019, 5020  
   - **COPD**: 1006, 1007, 1012, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 2010, 2012, 2013  
   - **HA**: 1001, 1002, 1008, 1010, 1014, 1016, 1017, 1018, 3002, 3003, 3006, 3007, 3008, 3009, 4009, 5000  
   - **MS**: 2001, 2003, 2005, 2007, 2008, 2011, 3004, 3012, 3014, 4005, 4006, 4008, 4013, 4019  
   - **PD**: 1003, 1004, 1005, 1009, 1013, 1015, 3010, 3011, 4002, 4003, 4012, 4015, 4016, 4017, 4020  
   - **PFF**: 5001, 5004, 5005, 5006, 5007, 5009, 5010, 5014, 5016, 5017, 5018  
4. Run `data_parsing/MOBILISE_D_parsing.py` to generate windowed acceleration data and gait metric labels.

## Fine-Tuning the Model

1. Run `hyperparameter_tuning.py` to optimize hyperparameters for each gait metric.
2. Use `final_training.py` to train the model with the best configuration on the full dataset.
3. Alternatively, skip training and use `inference.py` with pre-trained models. Best configurations and weights are in `models_config.txt`.

## Daily Living Analysis

Reproduce the daily living analysis by running `RUSH/rush_pipeline.py`.

## Citation

If you use this code or ElderNet in your research, please cite:  
"Continuous Assessment of Daily-Living Gait Using Self-Supervised Learning of Wrist-Worn Accelerometer Data," [medRxiv](https://www.medrxiv.org/content/10.1101/2025.05.21.25328061v1).
