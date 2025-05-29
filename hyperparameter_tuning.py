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
from dataset.transformations import RotationAxis, RandomSwitchAxis
from dataset.dataloader import FT_Dataset
from utils import (EarlyStopping, set_seed, cleanup_gpu, setup_model,
                   stratified_group_k_fold, stratified_group_train_val_split,
                   create_cosine_decay_with_warmup, plot_correlation_cohort,
                   evaluate_model, evaluate_metrics, compute_icc)

# GLOBAL SEED
seed = 42

class DuplicateIterationPruner(BasePruner):
    """Pruner to detect duplicate trials based on parameters."""

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
        for completed_trial in completed_trials:
            if completed_trial.params == trial.params:
                return True
        return False


def configure_logger(run_path: Path, name: str) -> logging.Logger:
    """Set up a file-based logger at INFO level.

    Args:
        run_path (Path): Directory path for the log file.
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(run_path / "log_file.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def make_experiment_dirs(cfg) -> Path:
    """Create experiment directory structure: <log_path>/<cohort>_<net>_pretrained_<...>/<timestamp>/.

    Args:
        cfg: Configuration object from Hydra.

    Returns:
        Path: Path to the run-specific directory.
    """
    name = (f"{cfg.data.cohort}_{cfg.model.net}_pretrained_{cfg.model.pretrained}"
            f"_data_{cfg.dataloader.epoch_len}_overlap_{cfg.data.overlap}")
    base_dir = Path(cfg.data.log_path) / name
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_path = base_dir / timestamp
    run_path.mkdir()
    return run_path


def prepare_cached_data(cfg) -> list:
    """Prepare and cache preprocessed datasets, labels, groups, and test indices for each fold.

    Args:
        cfg: Configuration object from Hydra.

    Returns:
        list: List of tuples containing (train_dataset, val_dataset, test_dataset) for each fold.
    """
    cache_dir = Path(cfg.data.log_path) / 'cached_data'
    cache_file = cache_dir / "preprocessed_data.p"
    if cache_file.exists():
        print("Loading cached preprocessed data...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # Load raw data
    input_data = pickle.load(open(os.path.join(cfg.data.data_root, 'acc.p'), 'rb'))
    input_data = np.transpose(input_data, (0, 2, 1))
    labels = pickle.load(open(os.path.join(cfg.data.data_root, cfg.data.labels), 'rb'))
    subjects = pickle.load(open(os.path.join(cfg.data.data_root, 'subjects.p'), 'rb'))
    cohorts = np.asarray([sub.split('_')[0] for sub in subjects])

    # Define transformations
    augmentations = []
    if cfg.augmentation.axis_switch:
        augmentations.append(RandomSwitchAxis())
    if cfg.augmentation.rotation:
        augmentations.append(RotationAxis())
    transform = transforms.Compose(augmentations) if augmentations else None

    preprocessed_data = []
    all_test_labels = []
    all_test_groups = []
    all_test_indices = []

    for fold, (train_idxs, test_idxs) in enumerate(stratified_group_k_fold(subjects, cohorts, random_state=seed)):
        X_train, Y_train, groups_train = input_data[train_idxs], labels[train_idxs], subjects[train_idxs]
        X_test, Y_test, groups_test = input_data[test_idxs], labels[test_idxs], subjects[test_idxs]
        train_idx, val_idx = stratified_group_train_val_split(groups_train, cohorts[train_idxs], test_size=0.125,
                                                              random_state=seed)

        train_dataset = FT_Dataset(X_train[train_idx], Y_train[train_idx], name="training", cfg=cfg,
                                   transform=transform)
        val_dataset = FT_Dataset(X_train[val_idx], Y_train[val_idx], name="validation", cfg=cfg, transform=transform)
        test_dataset = FT_Dataset(X_test, Y_test, name="prediction", cfg=cfg, transform=transform)
        preprocessed_data.append((train_dataset, val_dataset, test_dataset))
        all_test_labels.append(Y_test)
        all_test_groups.append(groups_test)
        all_test_indices.append(test_idxs)

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    with open(cache_dir / 'labels.p', 'wb') as f:
        pickle.dump(np.concatenate(all_test_labels), f)
    with open(cache_dir / 'groups.p', 'wb') as f:
        pickle.dump(np.concatenate(all_test_groups), f)
    with open(cache_dir / 'test_indices.p', 'wb') as f:
        pickle.dump(np.concatenate(all_test_indices), f)
    return preprocessed_data

def setup_model_with_params(cfg, trial: optuna.Trial, device: torch.device) -> tuple:
    """Set up the model, optimizer, scheduler, and batch size based on trial hyperparameters.

    Args:
        cfg: Configuration object from Hydra.
        trial (optuna.Trial): Current Optuna trial.
        device (torch.device): Device to run the model on.

    Returns:
        tuple: (model, optimizer, scheduler, batch_size)
    """
    learning_rate = trial.suggest_categorical('lr', [1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    num_layers_regressor = trial.suggest_int('num_layers', 0, 2)
    weight_decay = trial.suggest_categorical('wd', [0, 0.01, 0.1])
    batch_norm = trial.suggest_categorical('batch_norm', [True, False])

    model = setup_model(
        net=cfg.model.net,
        epoch_len=cfg.dataloader.epoch_len,
        is_regression=cfg.model.is_regression,
        max_mu=cfg.data.max_mu,
        num_layers_regressor=num_layers_regressor,
        batch_norm=batch_norm,
        head=cfg.model.head if cfg.model.net == 'ElderNet' else None,
        pretrained=cfg.model.pretrained,
        trained_model_path=cfg.model.trained_model_path,
        name_start_idx=cfg.model.name_start_idx,
        device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)
    scheduler = create_cosine_decay_with_warmup(optimizer, cfg.model.warmup_epochs, max_epochs=100)
    return model, optimizer, scheduler, batch_size


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer,
                scheduler, loss_fn: nn.Module, device: torch.device, early_stopping: EarlyStopping,
                num_epochs: int, trial: optuna.Trial, fold: int, logger: logging.Logger) -> None:
    """Train the model for one fold with early stopping.

    Args:
        model (nn.Module): Neural network model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to train on.
        early_stopping (EarlyStopping): Early stopping instance.
        num_epochs (int): Number of epochs to train.
        trial (optuna.Trial): Current Optuna trial.
        fold (int): Current fold index.
        logger (logging.Logger): Logger instance.
    """
    for epoch in range(num_epochs):
        model.train()
        train_losses, train_maes = [], []
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float).unsqueeze(1)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = loss_fn(outputs.float(), targets.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            train_maes.append(mean_absolute_error(targets.cpu().detach(), outputs.cpu().detach()))

        val_loss, val_mae = evaluate_model(model, val_loader, device, loss_fn)
        scheduler.step()
        logger.info(f"Trial {trial.number}, Fold: {fold + 1}, Epoch {epoch + 1}: "
                    f"train_loss={np.mean(train_losses):.3f}, train_mae={np.mean(train_maes):.3f} | "
                    f"val_loss={val_loss:.3f}, val_mae={val_mae:.3f}")

        trial.report(val_loss, epoch + (fold * num_epochs))
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break


def predict(model: nn.Module, test_loader: DataLoader, device: torch.device) -> tuple:
    """Generate predictions on the test set.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to run predictions on.

    Returns:
        tuple: (true_values, predictions) as numpy arrays.
    """
    model.eval()
    predictions, true_values = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device, dtype=torch.float)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            true_values.append(targets.numpy())
    return np.concatenate(true_values), np.concatenate(predictions).squeeze()


def objective(trial: optuna.Trial, cfg, logger: logging.Logger, run_path: Path, preprocessed_data: list,
              labels: np.ndarray, groups: np.ndarray) -> float:
    """Objective function for Optuna to minimize MAE.

    Args:
        trial (optuna.Trial): Current Optuna trial.
        cfg: Configuration object from Hydra.
        logger (logging.Logger): Logger instance.
        run_path (Path): Directory for saving checkpoints.
        preprocessed_data (list): Preprocessed datasets for each fold.
        labels (np.ndarray): Ground truth labels.
        groups (np.ndarray): Subject groups.

    Returns:
        float: Mean Absolute Error (MAE) or infinity if trial fails.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_fn = nn.L1Loss()
        early_stopping = EarlyStopping(patience=cfg.model.patience, verbose=True,
                                       path=os.path.join(run_path, f'checkpoint_trial{trial.number}.pt'))
        logger.info(f"Trial {trial.number} started with parameters: {trial.params}")

        predictions = []
        for fold, (train_dataset, val_dataset, test_dataset) in enumerate(preprocessed_data):
            model, optimizer, scheduler, batch_size = setup_model_with_params(cfg, trial, device)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

            train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, early_stopping,
                        cfg.model.num_epochs, trial, fold, logger)

            model.load_state_dict(torch.load(os.path.join(run_path, f'checkpoint_trial{trial.number}.pt')))
            _, fold_preds = predict(model, test_loader, device)
            predictions.append(fold_preds)
            del model
            cleanup_gpu()

        if not predictions:
            raise ValueError("No valid predictions generated across folds")

        predictions = np.concatenate(predictions)
        mae, rmse, mape, r2 = evaluate_metrics(labels, predictions)
        icc = compute_icc(labels, predictions)
        logger.info(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.3f}, R2: {r2:.3f}, ICC: {icc:.3f}")

        trial.set_user_attr('y_true', labels)
        trial.set_user_attr('y_pred', predictions)
        trial.set_user_attr('groups', groups)
        trial.set_user_attr('mae', mae)
        trial.set_user_attr('mape', mape)
        trial.set_user_attr('r2', r2)
        trial.set_user_attr('icc', icc)
        cleanup_gpu()
        return mae
    except Exception as e:
        logger.warning(f"Trial {trial.number} failed: {str(e)}")
        return float('inf')


def save_best_model(study: optuna.Study, trial: optuna.Trial, cfg, logger: logging.Logger, run_path: Path) -> None:
    """Save the best model and its predictions if this trial is the best.

    Args:
        study (optuna.Study): Optuna study object.
        trial (optuna.Trial): Current Optuna trial.
        cfg: Configuration object from Hydra.
        logger (logging.Logger): Logger instance.
        run_path (Path): Directory for saving files.
    """
    if study.best_trial.number == trial.number:
        checkpoint_path = os.path.join(run_path, f'checkpoint_trial{trial.number}.pt')
        best_model_path = os.path.join(run_path, 'best_model.pt')
        torch.save(torch.load(checkpoint_path), best_model_path)
        logger.info(f"New best model saved from trial {trial.number}")
        for name, data in [("labels", trial.user_attrs['y_true']),
                           ("predictions", trial.user_attrs['y_pred']),
                           ("groups", trial.user_attrs['groups']),
                           ("mae", trial.user_attrs['mae'])]:
            with open(os.path.join(run_path, f"{name}.p"), 'wb') as f:
                pickle.dump(data, f)
        logger.info("Best trial data saved.")
        plot_correlation_cohort(trial.user_attrs['y_true'], trial.user_attrs['y_pred'],
                                cfg.data.measure, cfg.data.unit,
                                save_path=os.path.join(run_path, "Scatter_plot.png"))


@hydra.main(config_path="conf", config_name="config_hyp", version_base='1.1')
def main(cfg):
    """Main function to run hyperparameter tuning.

    Args:
        cfg: Configuration object from Hydra.
    """
    set_seed(seed)
    run_path = make_experiment_dirs(cfg)
    logger = configure_logger(run_path, f"hyp_tune_{datetime.now():%Y%m%d_%H%M%S}")
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    preprocessed_data = prepare_cached_data(cfg)
    cache_dir = Path(cfg.data.log_path) / 'cached_data'
    labels = pickle.load(open(os.path.join(cache_dir, 'labels.p'), 'rb'))
    groups = pickle.load(open(os.path.join(cache_dir, 'groups.p'), 'rb'))

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

    importance = optuna.importance.get_param_importances(study)
    logger.info("Hyperparameter importance:")
    for param, score in importance.items():
        logger.info(f"  {param}: {score:.3f}")

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