import os
import pickle
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from torchvision import transforms

from dataset.dataloader import FT_Dataset
from dataset.transformations import RotationAxis, RandomSwitchAxis
from utils import (EarlyStopping, set_seed, stratified_group_train_val_split, setup_model,
                   create_cosine_decay_with_warmup,evaluate_model)

import hydra

# GLOBAL SEED
seed = 42

'''
Train ElderNet on the training dataset using the optimal configuration identified during hyperparameter tuning.
'''

@hydra.main(config_path="conf", config_name="config_final_training", version_base='1.1')
def main(cfg):

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed for reproducibility
    set_seed(seed)

    # Load the best hyperparameters (you need to have saved these during hyperparameter tuning)
    best_params = {'lr': cfg.model.lr,
                   'batch_size': cfg.model.batch_size,
                   'patience': cfg.model.patience,
                   'num_layers': cfg.model.num_layers_regressor,
                   'wd': cfg.model.wd}


    # Load your data
    data_path = cfg.data.data_root
    output_path = cfg.data.log_path
    os.makedirs(output_path, exist_ok=True)
    X = pickle.load(open(os.path.join(data_path, 'acc.p'), 'rb'))
    X = np.transpose(X, (0, 2, 1))
    Y = pickle.load(open(os.path.join(data_path, cfg.data.labels), 'rb'))
    subjects = pickle.load(open(os.path.join(data_path, 'subjects.p'), 'rb'))
    cohorts = np.asarray([sub.split('_')[0] for sub in subjects])

    # Prepare dataset and dataloader
    augmentations = []
    if cfg.augmentation.axis_switch:
        augmentations.append(RandomSwitchAxis())
    if cfg.augmentation.rotation:
        augmentations.append(RotationAxis())
    transform = transforms.Compose(augmentations) if augmentations else None

    # Split to train and val
    train_idx, val_idx = stratified_group_train_val_split(
        subjects, cohorts, test_size=0.125, random_state=42
    )

    train_dataset = FT_Dataset(X[train_idx], Y[train_idx], name="training", cfg=cfg, transform=transform)
    val_dataset = FT_Dataset(X[val_idx], Y[val_idx], name="validation", cfg=cfg, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'],  pin_memory=True)


    
    model = setup_model(
        net= cfg.model.net,
        head= cfg.model.head if cfg.model.net == 'ElderNet' else None,
        eldernet_linear_output=cfg.model.linear_output_size if cfg.model.net == 'ElderNet' else None,
        epoch_len=cfg.dataloader.epoch_len,
        is_regression=cfg.model.is_regression,
        max_mu=cfg.data.max_mu,
        num_layers_regressor=cfg.model.num_layers_regressor,
        batch_norm=cfg.model.batch_norm,
        pretrained=cfg.model.pretrained,
        trained_model_path=cfg.model.trained_model_path,
        device=device)

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=best_params['lr'],
                                 weight_decay=best_params['wd'],
                                  amsgrad=True)
    scheduler = create_cosine_decay_with_warmup(optimizer, cfg.model.warmup_epochs, max_epochs=100)

    # Define loss function
    loss_fn = nn.L1Loss()

    # Setup early stopping
    early_stopping = EarlyStopping(patience=best_params['patience'], verbose=True, path='best_model.pt')

    # Training loop
    num_epochs = cfg.model.num_epochs
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_maes = []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.float).unsqueeze(1)

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                loss = loss_fn(logits.float(), y.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_mae = mean_absolute_error(y.cpu().detach(), logits.cpu().detach())
            train_maes.append(train_mae)

        # Evaluate on validation set (if you have one)
        val_loss, val_mae = evaluate_model(model, val_loader, device, loss_fn)

        scheduler.step()

        logger.info(f"Epoch {epoch + 1}: train_loss={np.mean(train_losses):.3f}, train_mae={np.mean(train_maes):.3f}")

        # If you have a validation set:
        logger.info(f"Epoch {epoch + 1}: val_loss={val_loss:.3f}, val_mae={val_mae:.3f}")
        early_stopping(val_loss, model)

        # If you don't have a validation set, you might want to save the model periodically:
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pt')

        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    # Load the best model (if using early stopping) or the last model
    model.load_state_dict(torch.load('best_model.pt'))

    # Save the final model
    torch.save(model.state_dict(), os.path.join(output_path,'final_model.pt'))
    logger.info("Training completed. Final model saved as 'final_model.pt'")


if __name__ == "__main__":
    main()