import os
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pickle

import hydra
from omegaconf import OmegaConf
import optuna
from optuna.pruners import BasePruner

import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import SEED as seed
from dataset.transformations import RotationAxis, RandomSwitchAxis
from dataset.dataloader import FT_Dataset
from utils import (EarlyStopping, set_seed, cleanup_gpu, setup_model,
                   stratified_group_k_fold, stratified_group_train_val_split,
                   create_cosine_decay_with_warmup,plot_correlation_cohort, evaluate_model, evaluate_metrics, compute_icc)


class DuplicateIterationPruner(BasePruner):
    """
    DuplicatePruner

    Pruner to detect duplicate trials based on the parameters.

    This pruner is used to identify and prune trials that have the same set of parameters
    as a previously completed trial.
    """

    def prune(
        self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
    ) -> bool:
        completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])

        for completed_trial in completed_trials:
            if completed_trial.params == trial.params:
                return True

        return False


def configure_logger(run_path: Path, name: str) -> logging.Logger:
    """Set up a file-based logger at INFO level."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(run_path / "log_file.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger

def make_experiment_dirs(cfg) -> Path:
    """
    Build the experiment directory structure:
      <log_path>/<cohort>_<net>_pretrained_<...>/<timestamp>/
    Returns the path where all outputs for this run should go.
    """
    name = (
        f"{cfg.data.cohort}_{cfg.model.net}_pretrained_{cfg.model.pretrained}"
        f"_data_{cfg.dataloader.epoch_len}_overlap_{cfg.data.overlap}"
    )
    base = Path(cfg.data.log_path) / name
    base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_path = base / timestamp
    run_path.mkdir()
    return run_path

def prepare_data(x, y, name, cfg, transform, batch_size):
    dataset = FT_Dataset(x, y, name=name, cfg=cfg, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=name == 'training')
    return dataloader

def prepare_cached_data(x, y, groups, cohorts, cfg, transform, cache_dir):
    print("Preprocessing and caching data...")
    cache_file = Path(cache_dir) / f"preprocessed_data.p"
    preprocessed_data = []
    labels = []
    all_groups_test = []
    test_indices = []
    for fold, (train_idxs, test_idxs) in enumerate(stratified_group_k_fold(groups, cohorts, random_state=seed)):
        X_train, Y_train, groups_train = x[train_idxs], y[train_idxs], groups[train_idxs]
        X_test, Y_test, groups_test = x[test_idxs], y[test_idxs], groups[test_idxs]
        train_idx, val_idx = stratified_group_train_val_split(
            groups_train, cohorts[train_idxs], test_size=0.125, random_state=seed
        )

        train_dataset = FT_Dataset(X_train[train_idx], Y_train[train_idx], name="training", cfg=cfg,transform=transform)
        val_dataset = FT_Dataset(X_train[val_idx], Y_train[val_idx], name="validation", cfg=cfg, transform=transform)
        test_dataset = FT_Dataset(X_test, Y_test, name="prediction", cfg=cfg, transform=transform)

        preprocessed_data.append((train_dataset, val_dataset, test_dataset))
        labels.append(Y_test)
        all_groups_test.append(groups_test)
        test_indices.append(test_idxs)

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    with open(os.path.join(cache_dir, 'labels.p'), 'wb') as f:
        pickle.dump(np.concatenate(labels), f)
    with open(os.path.join(cache_dir, 'groups.p'), 'wb') as f:
        pickle.dump(np.concatenate(all_groups_test), f)
    with open(os.path.join(cache_dir, 'test_indices.p'), 'wb') as f:
        pickle.dump(np.concatenate(test_indices), f)

    return preprocessed_data

def predict(model, data_loader, device):
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Predicting"):
            x = x.to(device, dtype=torch.float)
            logits = model(x)
            predictions.append(logits.cpu().numpy())
            true_values.append(y.numpy())


    return np.concatenate(true_values), np.concatenate(predictions).squeeze()

def plot_best_trial(trial, cfg, logger, run_path):
    # Scatter plot
    if (hasattr(trial, 'user_attrs') and 'y_true' in trial.user_attrs and 'y_pred' in trial.user_attrs
            and 'mae' in trial.user_attrs):
        plot_correlation_cohort(trial.user_attrs['y_true'], trial.user_attrs['y_pred'],
                                cfg.data.measure, cfg.data.unit,
                                save_path= os.path.join(run_path, "Scatter plot.png"))
    logger.info("Plots for the best trial have been saved.")

def save_best_model(study, trial, cfg, logger, run_path):
    if study.best_trial.number == trial.number:
        file2load = os.path.join(run_path, f'checkpoint_trial{trial.number}.pt')
        file2save = os.path.join(run_path, 'best_model.pt')
        torch.save(torch.load(file2load), file2save)
        logger.info(f"New best model saved from trial {trial.number}")
        # Save labels, groups, predictions, and MAE for the best trial
        for name, data in [("labels", trial.user_attrs['y_true']),
                           ("predictions", trial.user_attrs['y_pred']),
                           ("CI", trial.user_attrs['CI']),
                           ("groups", trial.user_attrs['groups']),
                           ("mae", trial.user_attrs['mae'])]:
            with open(os.path.join(run_path, f"{name}.p"), 'wb') as f:
                pickle.dump(data, f)
        logger.info("Best trial data (labels, groups, predictions, MAE) saved.")
        # Trigger plotting for the best trial
        plot_best_trial(trial, cfg, logger, run_path)

@hydra.main(config_path="conf", config_name="config_hyp", version_base='1.1')
def main(cfg):
    # Ensure reproducibility
    set_seed(seed)
    # Create the experiment directory if it doesn't exist
    run_path = make_experiment_dirs(cfg)
    logger = configure_logger(run_path, f"hyp_tune_{datetime.now():%Y%m%d_%H%M%S}")
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Data loading and preprocessing
    cache_dir = os.path.join(cfg.data.log_path, 'cached_data')
    cache_file = Path(cache_dir) / f"preprocessed_data.p"
    if cache_file.exists():
        print("Loading cached preprocessed data...")
        with open(cache_file, 'rb') as f:
            preprocessed_data = pickle.load(f)
    else:
        X = pickle.load(open(os.path.join(cfg.data.data_root, 'acc.p'), 'rb'))
        X = np.transpose(X, (0, 2, 1))
        Y = pickle.load(open(os.path.join(cfg.data.data_root, cfg.data.labels), 'rb'))
        subjects = pickle.load(open(os.path.join(cfg.data.data_root, 'subjects.p'), 'rb'))
        cohorts = np.asarray([sub.split('_')[0] for sub in subjects])

        # Define your transform
        augmentations = []
        if cfg.augmentation.axis_switch:
            augmentations.append(RandomSwitchAxis())
        if cfg.augmentation.rotation:
            augmentations.append(RotationAxis())
        my_transform = transforms.Compose(augmentations) if augmentations else None
        preprocessed_data = prepare_cached_data(X, Y, subjects, cohorts, cfg, my_transform, cache_dir)

    labels = pickle.load(open(os.path.join(cache_dir, 'labels.p'), 'rb'))
    groups = pickle.load(open(os.path.join(cache_dir, 'groups.p'), 'rb'))

    def objective(trial, cfg, logger, run_path, preprocessed_data, labels, groups):
        try:
            # Suggest hyperparameters
            lr = trial.suggest_categorical('lr', [1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
            batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
            num_layers_regressor = trial.suggest_int('num_layers', 0, 2)
            cfg.model.num_layers_regressor = num_layers_regressor
            wd = trial.suggest_categorical('wd', [0, 0.01, 0.1])
            batch_norm = trial.suggest_categorical('batch_norm', [True, False])
            cfg.model.batch_norm = batch_norm
            num_epochs = cfg.model.num_epochs

            patience = 5
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Trial {trial.number} started with parameters: {trial.params}")

            predictions = []
            for fold, (train_dataset, val_dataset, test_dataset) in enumerate(preprocessed_data):
                try:
                    t0 = time.time()
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

                    # Setup model
                    model = setup_model(
                        net = cfg.model.net,
                        epoch_len=cfg.dataloader.epoch_len,
                        is_regression=cfg.model.is_regression,
                        max_mu=cfg.data.max_mu,
                        num_layers_regressor=cfg.model.num_layers_regressor,
                        batch_norm=cfg.model.batch_norm,
                        head=cfg.model.head if cfg.model.net=='ElderNet' else None,
                        pretrained=cfg.model.pretrained,
                        trained_model_path=cfg.model.trained_model_path,
                        name_start_idx=cfg.model.name_start_idx,
                        device=device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, amsgrad=True)
                    scheduler = create_cosine_decay_with_warmup(optimizer, cfg.model.warmup_epochs, max_epochs=100)

                    # Define the loss function
                    loss_fn =  nn.L1Loss()
                    early_stopping = EarlyStopping(patience=patience, verbose=True,
                                                   path=os.path.join(run_path, f'checkpoint_trial{trial.number}.pt'))

                    for epoch in range(num_epochs):
                        t_epoch_0 = time.time()
                        model.train()
                        train_losses = []
                        train_maes = []

                        for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                            x = x.to(device, dtype=torch.float).requires_grad_(True)
                            y = y.to(device, dtype=torch.float).unsqueeze(1)

                            optimizer.zero_grad()
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                logits = model(x)
                                loss = loss_fn(logits.float(), y.float())

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()

                            # Check for NaN values in model parameters
                            if any(torch.isnan(param).any() for param in model.parameters()):
                                logger.warning(
                                    f"Trial {trial.number}, Fold {fold + 1}, Epoch {epoch + 1}: NaN values detected!")
                                raise ValueError("NaN values detected in model parameters")

                            train_losses.append(loss.item())
                            train_maes.append(mean_absolute_error(y.cpu().detach(), logits.cpu().detach()))

                        val_loss, val_mae = evaluate_model(model, val_loader, device, loss_fn)
                        scheduler.step()
                        t_epoch_1 = time.time()
                        dt_epoch = t_epoch_1 - t_epoch_0
                        windows_per_sec = batch_size / dt_epoch
                        logger.info(f"Trial {trial.number}, Fold: {fold + 1},  Epoch {epoch + 1}: "
                                    f"train_loss={np.mean(train_losses):.3f}, train_mae={np.mean(train_maes):.3f} | "
                                    f"val_loss={val_loss:.3f}, val_mae={val_mae:.3f} |"
                                    f"dt_epoch: {dt_epoch / 60:.2f}min | windows/sec: {windows_per_sec:.2f}")

                        # Check if the trial should be pruned
                        trial.report(val_loss, epoch + (fold * num_epochs))
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()

                        early_stopping(val_loss, model)
                        if early_stopping.early_stop:
                            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                            break

                    t1 = time.time()
                    dt = t1 - t0
                    logger.info(f"fold dt: {dt / 60:.2f}min")

                    # Load best model and evaluate on test set
                    model.load_state_dict(torch.load(os.path.join(run_path, f'checkpoint_trial{trial.number}.pt')))
                    Y_test_true, Y_test_pred = predict(model, test_loader, device)
                    predictions.append(Y_test_pred)

                    # Cleanup after each fold
                    del model
                    cleanup_gpu()

                except (ValueError, RuntimeError) as fold_error:
                    logger.warning(f"Skipping fold {fold + 1} due to error: {str(fold_error)}")
                    # Continue to the next fold or exit if all folds fail
                    continue

            # If no predictions were made, raise an exception to skip the trial
            if not predictions:
                raise ValueError("No valid predictions could be made")

            predictions = np.concatenate(predictions)
            mae, rmse, mape, r2 = evaluate_metrics(labels, predictions)
            icc = compute_icc(labels, predictions)
            logger.info(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.3f}, R2: {r2:.3f}")

            # Store the labels, predictions and metrics
            trial.set_user_attr('y_true', labels)
            trial.set_user_attr('y_pred', predictions)
            trial.set_user_attr('groups', groups)
            trial.set_user_attr('mae', mae)
            trial.set_user_attr('mape', mape)
            trial.set_user_attr('r2', r2)
            trial.set_user_attr('icc', icc)

            # Final cleanup
            cleanup_gpu()
            return mae

        except Exception as trial_error:
            logger.warning(f"Trial {trial.number} failed: {str(trial_error)}")
            # Return a large value to indicate an unsuccessful trial
            return float('inf')
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=DuplicateIterationPruner())


    study.optimize(lambda trial: objective(trial, cfg, logger, run_path, preprocessed_data, labels, groups),
                   n_trials=cfg.model.num_trials,
                   callbacks=[lambda study, trial: save_best_model(study, trial, cfg, logger, run_path)])

    logger.info('Best trial:')
    trial = study.best_trial
    logger.info(f'  Value: {trial.value}')
    logger.info('  Params: ')
    for key, value in trial.params.items():
        logger.info(f'    {key}: {value}')

    # Analyze hyperparameter importance
    importance = optuna.importance.get_param_importances(study)
    logger.info("Hyperparameter importance:")
    for param, score in importance.items():
        logger.info(f"  {param}: {score:.3f}")

    # Visualizations
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(os.path.join(run_path, "optimization_history.png"))

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(os.path.join(run_path, "param_importances.png"))

        logger.info("Visualization plots saved as PNG files.")
    except Exception as e:
        logger.warning(f"Failed to generate visualizations: {str(e)}")


if __name__ == '__main__':
    main()