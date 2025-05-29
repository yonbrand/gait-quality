import random
import math
import pickle
import numpy as np
import pandas as pd

import scipy.io as sio
from scipy.io.matlab import mat_struct
from scipy import signal

from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import copy

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import pingouin as pg

import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 1000000000
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from models import Resnet, ElderNet

verbose = False
torch_cache_path = Path(__file__).parent / 'torch_hub_cache'
cuda = torch.cuda.is_available()


######################################################################################################################
# Loading
######################################################################################################################
def loadmat(filename):
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(data):
    for key in data:
        if isinstance(data[key], mat_struct):
            data[key] = _todict(data[key])
    return data


def _todict(matobj):
    return {strg: _todict(elem) if isinstance(elem, mat_struct) else elem for strg, elem in matobj.__dict__.items()}


######################################################################################################################
# Initialization
######################################################################################################################
def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###################################################################################################################
# Early Stopping
###################################################################################################################
"""
Taken from https://github.com/Bjarten/early-stopping-pytorch
"""


class EarlyStopping:
    """Early stops the training if validation loss
    doesn't improve after a given patience."""

    def __init__(
            self,
            patience=5,
            verbose=False,
            delta=0,
            path="checkpoint.pt",
            trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time v
                            alidation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each
                            validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity
                            to qualify as an improvement.
                            Default: 7
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            msg = "Validation loss decreased"
            msg = msg + f"({self.val_loss_min:.6f} --> {val_loss:.6f})"
            msg = msg + "Saving model ..."
            self.trace_func(msg)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


########################################################################################################################
## Metrics Evaluation
#######################################################################################################################
def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, mape, r2

def compute_icc(y_true, y_pred):
    # Create a DataFrame with labels and predictions
    labels_df = pd.DataFrame({
        'gait_speed': y_true,
        'subject': np.arange(len(y_true)),  # Ensure subject identifiers are meaningful
        'rater': 'labels'  # Add a constant np.uniqrater column for labels
    })
    predictions_df = pd.DataFrame({
        'gait_speed': y_pred,
        'subject': np.arange(len(y_pred)),  # Ensure subject identifiers are meaningful
        'rater': 'predictions'  # Add a constant rater column for predictions
    })

    # Concatenate the DataFrames
    data = pd.concat([labels_df, predictions_df], axis=0)
    # Calculate the ICC (2,1) for consistency (can switch to ICC(1) for absolute agreement)
    icc_result = pg.intraclass_corr(data=data, targets='subject', raters='rater', ratings='gait_speed')
    # Extract the ICC value for type 'ICC2'
    icc_value = icc_result.loc[icc_result['Type'] == 'ICC2', 'ICC'].values[0]

    return icc_value

def compute_subject_metrics(y_true, y_pred, subjects):
    maes = []
    rmses = []
    mapes = []
    r2s = []
    iccs = []
    unq_subs = np.unique(subjects)
    for sub in unq_subs:
        mask = subjects == sub
        cur_labels = y_true[mask]
        cur_preds = y_pred[mask]
        mae, rmse, mape, r2 = evaluate_metrics(cur_labels, cur_preds)
        icc = compute_icc(cur_labels, cur_preds)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        r2s.append(r2)
        iccs.append(icc)

    return maes, rmses, mapes, r2s, iccs


###################################################################################################################
# Bout Analysis
###################################################################################################################
def calculate_bout_durations_indicator(bout_id_path, test_idxs_path):
    """
    This function calculates the bout durations indicator from a given bout_id file.

    Arguments:
    bout_id_path : str
        The path to the .p file containing the bout_id data.

    test_idxs_path : str
        The path to the .p file containing the indexes mapping to the predictions array order.

    Returns:
    bout_durations_indicator : numpy array
        An array where each element corresponds to the duration of the bout in which it belongs.
    """

    # Load the bout_id and test indices data from the specified path
    bout_id = pickle.load(open(bout_id_path, 'rb'))
    test_idxs = pickle.load(open(test_idxs_path, 'rb'))

    # Re-order bout_id to align 5-fold validation predictions
    bout_id = bout_id[test_idxs]

    # Calculate the differences between consecutive bout IDs
    bout_diff = np.diff(bout_id)

    # Identify the start indices of each new bout (where the ID changes)
    bout_starts = np.where(bout_diff != 0)[0] + 1

    # Add the first bout start at index 0
    bout_starts = np.insert(bout_starts, 0, 0)

    # Calculate the durations of each bout
    bout_durations = np.diff(bout_starts)

    # Initialize an array of zeros with the same shape as bout_id
    bout_durations_indicator = np.zeros_like(bout_id)

    # Assign the duration of each bout to all elements within that bout
    for ind in range(len(bout_starts)):
        if ind == len(bout_durations):
            # If we're at the last bout, fill the remaining elements with the duration
            bout_durations_indicator[bout_starts[-1]:] = len(bout_id) - bout_starts[-1]
        else:
            # Otherwise, fill the elements between consecutive bout starts with the bout's duration
            bout_durations_indicator[bout_starts[ind]:bout_starts[ind + 1]] = bout_durations[ind]

    return bout_durations_indicator


def window2bout(labels, preds, subjects, bout_id):
    '''
    Convert window predictions and labels to bout-level
    :param subjects: obtained from the parsed data (test set)
    :param bout_id: obtained from the parsed data (test set)
    :return:
    '''
    bout_data = defaultdict(lambda: {"labels": [], "preds": []})
    for l, p, s, b in zip(labels, preds, subjects, bout_id):
        bout_key = f"{s}_{b}"
        bout_data[bout_key]["labels"].append(l)
        bout_data[bout_key]["preds"].append(p)

    bout_medians = {
        bout: {
            "label_median": np.median(data["labels"]),
            "pred_median": np.median(data["preds"])
        }
        for bout, data in bout_data.items()
    }

    bout_labels = [data["label_median"] for data in bout_medians.values()]
    bout_preds = [data["pred_median"] for data in bout_medians.values()]
    bout_subjects = [data.split('_')[0] + '_' + data.split('_')[1] for data in bout_medians.keys()]
    #
    # mae, rmse, r2 = evaluate_metrics(y_true, y_pred)
    # icc = compute_icc(y_true, y_pred)

    return np.array(bout_labels), np.array(bout_preds), np.array(bout_subjects)


def plot_results(bout_medians, mae):
    label_medians = [data["label_median"] for data in bout_medians.values()]
    pred_medians = [data["pred_median"] for data in bout_medians.values()]

    plt.figure(figsize=(10, 8))
    plt.scatter(label_medians, pred_medians, alpha=0.6)
    plt.plot([min(label_medians), max(label_medians)],
             [min(label_medians), max(label_medians)],
             'r--', lw=2)  # Add identity line

    plt.xlabel("Ground Truth Median Gait Speed")
    plt.ylabel("Predicted Median Gait Speed")
    plt.title(f"Bout-wise Median Gait Speed, MAE: {mae}")

    plt.tight_layout()
    plt.show()


####################################################################################################################
# Data splitting
####################################################################################################################
def stratified_group_k_fold(groups, cohorts, n_splits=5, random_state=None):
    np.random.seed(random_state)
    unique_cohorts = np.unique(cohorts)
    folds = [[] for _ in range(n_splits)]

    for cohort in unique_cohorts:
        cohort_indices = np.where(cohorts == cohort)[0]
        cohort_subjects = np.unique(groups[cohort_indices])

        subject_to_fold = {subject: i % n_splits for i, subject in enumerate(np.random.permutation(cohort_subjects))}

        for idx in cohort_indices:
            subject = groups[idx]
            fold = subject_to_fold[subject]
            folds[fold].append(idx)

    for i in range(n_splits):
        train_idx = np.hstack([folds[j] for j in range(n_splits) if j != i])
        test_idx = np.array(folds[i])
        yield train_idx, test_idx


def stratified_group_train_val_split(groups, cohorts, test_size=0.125, random_state=None):
    np.random.seed(random_state)

    unique_cohorts = np.unique(cohorts)
    train_indices = []
    val_indices = []
    for cohort in unique_cohorts:
        cohort_mask = cohorts == cohort
        cohort_groups = groups[cohort_mask]
        unique_groups = np.unique(cohort_groups)
        n_val = int(len(unique_groups) * test_size)
        # if there are less than 8 subjects in the training set n_val will be 0
        if n_val == 0:
            n_val = 1
        val_sub = np.random.choice(unique_groups, size=n_val, replace=False)
        train_sub = unique_groups[~np.isin(unique_groups, val_sub)]
        cohort_val_indices = np.where(np.isin(groups, val_sub))[0]
        cohort_train_indices = np.where(np.isin(groups, train_sub))[0]
        train_indices.extend(cohort_train_indices)
        val_indices.extend(cohort_val_indices)
    return np.array(train_indices), np.array((val_indices))


################################################################################################################
# Plots
################################################################################################################
def plot_correlation(y_true, y_pred, measure, unit, save_path):
    '''
    Scatter plot of a predicted continuous measure against ground truth
    :param y_true: np array; labels indicating the ground truth
    :param y_pred: np array; the predictions of the model
    :param save_path: string
    '''
    # # Convert from meter to cm
    # if unit == 'cm/s' or 'cm\s':
    #     y_true, y_pred = 100 * y_true, 100 * y_pred

    # Compute the metrics for model evaluation
    mae, rmse, mape, r2 = evaluate_metrics(y_true, y_pred)
    icc = compute_icc(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2)  # Reference line (y=x)

    plt.xlabel(f'Reference in {unit}', fontsize=18)
    plt.ylabel(f'Predictions in {unit}', fontsize=18)
    plt.title(f'{measure}', fontsize=20)

    # Add metrics as a text box
    metrics_text = (
        f"MAE: {mae:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"MAPE: {mape:.2f}\n"
        f"R²: {r2:.2f}\n"
        f"ICC(2,1): {icc:.2f}"
    )

    # Place the text box in the upper left corner
    plt.text(
        0.05, 0.95, metrics_text,
        transform=plt.gca().transAxes,  # Place relative to the plot axis
        fontsize=16,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', edgecolor='black', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_correlation_cohort(y_true, y_pred, measure, unit, title=None, save_path=None, cohorts=None):
    """
    Enhanced scatter plot for predicted vs. ground truth values with improvements.

    :param y_true: np.array; Ground truth values.
    :param y_pred: np.array; Predicted values.
    :param measure: str; Measure name to display on the plot title.
    :param unit: str; Unit for labels (e.g., cm/s).
    :param save_path: str; Path to save the plot.
    :param cohorts: list or np.array; Optional cohort labels for coloring points.
    """
    # Convert from meter to cm if needed
    if unit in ['cm/s', 'cm\\s', 'cm/stride', 'cm\\stride']:
        y_true, y_pred = 100 * y_true, 100 * y_pred

    # Compute the metrics for model evaluation
    mae, rmse, mape, r2 = evaluate_metrics(y_true, y_pred)
    icc = compute_icc(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    # Scatter plot with cohort coloring
    if cohorts is not None:
        unique_cohorts = np.unique(cohorts)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_cohorts)))
        for cohort, color in zip(unique_cohorts, colors):
            plt.scatter(
                y_true[cohorts == cohort], y_pred[cohorts == cohort],
                label=f'Cohort {cohort}', alpha=0.8, color=color, edgecolor='none', s=25
            )
        # Cohort legend
        plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.05, 0.95))
    else:
        plt.scatter(y_true, y_pred, alpha=0.7, color='steelblue', edgecolor='none', s=25)

    # Add diagonal reference line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([0, max_val], [0, max_val], 'k--', lw=2, label='y = x')

    # Add separate legend for the diagonal line in the bottom right
    plt.legend(fontsize=12, loc='lower right')

    # Enforce axes start at 0
    plt.xlim(0, 1.1 * max_val)
    plt.ylim(0, 1.1 * max_val)

    # Labels and title
    plt.xlabel(f'True {measure} in {unit}', fontsize=18)
    plt.ylabel(f'Predicted {measure} in {unit}', fontsize=18)
    plt.title(f'{title}', fontsize=20)

    # Add metrics text box
    metrics_text = (
        f"MAE: {mae:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"MAPE: {mape:.2f}\n"
        f"R²: {r2:.2f}\n"
        f"ICC(2,1): {icc:.2f}"
    )
    plt.text(
        0.05, 0.95, metrics_text,
        transform=plt.gca().transAxes,
        fontsize=16, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', edgecolor='black', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_model_comparison(y_true_list, y_pred_list, model_names, measure, unit, save_path, cohorts=None):
    """
    Create subplots for comparing models with a flexible grid layout.
    Each row uses a unique ground truth (y_true) dataset, and each column (model)
    is displayed with a distinct color if cohorts are not provided.
    :param y_true_list: list of np.arrays; Ground truth values for each dataset (one per row).
    :param y_pred_list: list of np.arrays; Predicted values for each model on each dataset
                        (length should be n_rows * n_cols).
    :param model_names: list of str; Names of models to display on subplots (length should be n_rows * n_cols).
    :param measure: str; Measure name to display on plot titles.
    :param unit: str; Unit for labels (e.g., cm/s).
    :param save_path: str; Path to save the comparison plot.
    :param cohorts: list or np.array; Optional cohort labels. If a list of arrays (length equals n_rows)
                    is provided, the corresponding array is used for that row.
    """
    n_rows = 2
    n_cols = len(model_names) // n_rows
    # Function to convert values based on unit if needed
    def convert(arr):
        return 100 * arr if unit in ['cm/s', 'cm\\s'] else arr
    # Convert each array for y_true and y_pred
    y_true_conv = [convert(y) for y in y_true_list]
    y_pred_conv = [convert(y) for y in y_pred_list]
    # Compute global min and max from all arrays for consistent scaling
    all_arrays = y_true_conv + y_pred_conv
    global_min = min(np.min(arr) for arr in all_arrays)
    global_max = max(np.max(arr) for arr in all_arrays)
    # Precompute column-specific colors if cohorts is not provided.
    # Define a custom palette of colors.
    if cohorts is None:
        custom_colors = ['steelblue', 'deepskyblue', 'darkturquoise']
        if n_cols <= len(custom_colors):
            column_colors = custom_colors[:n_cols]
        else:
            # Fallback to a colormap if number of columns exceeds predefined colors
            column_colors = plt.cm.tab10(np.linspace(0, 1, n_cols))

    plt.figure(figsize=(12, 10))
    # Loop over each subplot index; each row uses its corresponding y_true
    for i, (y_pred, model_name) in enumerate(zip(y_pred_conv, model_names), start=1):
        # Determine current row and column indices (0-indexed)
        row_idx = (i - 1) // n_cols
        col_idx = (i - 1) % n_cols
        # Select the corresponding ground truth for this row
        y_true = y_true_conv[row_idx]
        # Compute evaluation metrics for this subplot
        mae, rmse, mape, r2 = evaluate_metrics(y_true, y_pred)
        icc = compute_icc(y_true, y_pred)
        ax = plt.subplot(n_rows, n_cols, i)
        # Scatter plot: if cohorts is provided, use it for coloring;
        # otherwise, use the column-specific color.
        if cohorts is not None:
            # If cohorts is a list matching the number of rows, use the corresponding element
            current_cohorts = cohorts[row_idx] if isinstance(cohorts, list) and len(cohorts) == n_rows else cohorts
            unique_cohorts = np.unique(current_cohorts)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_cohorts)))
            for cohort, color in zip(unique_cohorts, colors):
                plt.scatter(
                    y_true[current_cohorts == cohort], y_pred[current_cohorts == cohort],
                    label=f'Cohort {cohort}', alpha=0.8, color=color, edgecolor='none', s=25
                )
            plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0.05, 0.95))
        else:
            color = column_colors[col_idx]
            plt.scatter(y_true, y_pred, alpha=0.7, color=color, edgecolor='none', s=25)
        # Add diagonal reference line
        plt.plot([global_min, global_max], [global_min, global_max], 'k--', lw=2, label='y = x')
        plt.legend(fontsize=12, loc='lower right')
        # Enforce consistent axes limits
        plt.xlim(global_min, 1.1 * global_max)
        plt.ylim(global_min, 1.1 * global_max)
        # Add labels only for the bottom row and first column respectively
        if row_idx == n_rows - 1:
            plt.xlabel(f'True {measure} in {unit}', fontsize=16)
        if col_idx == 0 and row_idx == 0:
            # plt.ylabel(f'Internal Validation (Dataset 3)\n\n Predicted {measure} in {unit}', fontsize=16)
            plt.ylabel(f'Predicted {measure} in {unit}', fontsize=16)
        elif col_idx == 0 and row_idx == 1:
                # plt.ylabel(f'External Validation (Dataset 4)\n\n Predicted {measure} in {unit}', fontsize=16)
                plt.ylabel(f'Predicted {measure} in {unit}', fontsize=16)
        plt.title(f'{model_name}', fontsize=14)
        # Display metrics as a text box
        mape = 100 * mape
        metrics_text = (
            f"MAE: {mae:.2f} (cm/s)\n"
            f"RMSE: {rmse:.2f} (cm/s)\n"
            f"MAPE: {mape:.2f} (%)\n"
            f"R²: {r2:.2f}\n"
            f"ICC: {icc:.2f}"
        )
        plt.text(
            0.05, 0.95, metrics_text,
            transform=plt.gca().transAxes,
            fontsize=13, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', edgecolor='black', facecolor='white', alpha=0.7)
        )
    plt.suptitle('Comparison of Model Performance in Predicting Gait Speed Across Validation Datasets', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_cohort_performance(labels, preds, groups, file_name):
    cohorts = [g.split('_')[0] for g in groups]
    unq_cohorts = np.unique(cohorts)
    plt.figure(figsize=(18, 12))  # Increased height for better spacing
    count = 1
    for cohort in unq_cohorts:
        cohort_idx = np.isin(cohorts, cohort)
        cohort_labels = labels[cohort_idx]
        cohort_preds = preds[cohort_idx]
        # Compute the metrics for model evaluation
        mae, rmse, mape, r2 = evaluate_metrics(cohort_labels, cohort_preds)
        icc = compute_icc(cohort_labels, cohort_preds)

        plt.subplot(2, 3, count)
        plt.scatter(cohort_labels, cohort_preds, alpha=0.6, edgecolors='w', s=100)  # Improved scatter plot
        plt.plot([0, max(cohort_labels)], [0, max(cohort_labels)], 'k--', lw=2)  # Reference line (y=x)
        plt.xlabel('Reference in m/s')
        plt.ylabel('Predictions in m/s')
        plt.title(f"Performance for cohort: {cohort}", fontsize=12)
        # Add metrics text box
        metrics_text = (
            f"MAE: {mae:.2f}\n"
            f"RMSE: {rmse:.2f}\n"
            f"MAPE: {mape:.2f}\n"
            f"R²: {r2:.2f}\n"
            f"ICC(2,1): {icc:.2f}"
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)
        count += 1
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close


def plot_subjects_performance(labels, groups, preds, file_name):
    unq_subjects = np.unique(groups)
    mae_per_sub = []
    rmse_per_sub = []
    r2_per_sub = []

    for sub in unq_subjects:
        sub_idx = np.isin(groups, sub)
        # Subject nust have more than 1 window for calculating r2
        if np.sum(sub_idx) < 2:
            continue
        subject_labels = labels[sub_idx]
        subject_preds = preds[sub_idx]
        mae, rmse, r2 = evaluate_metrics(subject_labels, subject_preds)
        mae_per_sub.append(mae)
        rmse_per_sub.append(rmse)
        r2_per_sub.append(r2)

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Performance Metrics Across Subjects')

    # Plot boxplots for each metric
    metrics = [mae_per_sub, rmse_per_sub, r2_per_sub]
    titles = ['MAE', 'RMSE', 'R2']
    axes = [ax1, ax2, ax3]

    for ax, metric, title in zip(axes, metrics, titles):
        bp = ax.boxplot(metric, patch_artist=True)
        ax.set_title(title)
        ax.set_ylabel('Value')

        # Calculate and display median and std
        median = np.median(metric)
        std = np.std(metric)
        ax.text(1.1, median, f'Median: {median:.3f}', verticalalignment='center')
        ax.text(1.1, median - 0.1, f'Std: {std:.3f}', verticalalignment='center')

        # Customize boxplot colors
        plt.setp(bp['boxes'], facecolor='lightblue', alpha=0.7)
        plt.setp(bp['medians'], color='red')

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

    # Print summary statistics
    print(f"MAE - Median: {np.median(mae_per_sub):.3f}, Std: {np.std(mae_per_sub):.3f}")
    print(f"RMSE - Median: {np.median(rmse_per_sub):.3f}, Std: {np.std(rmse_per_sub):.3f}")
    print(f"R2 - Median: {np.median(r2_per_sub):.3f}, Std: {np.std(r2_per_sub):.3f}")


def plot_random_windows(acc, gait_speed, turn_coverages, turn_counts, filename):
    fig, axs = plt.subplots(2, 3, figsize=(20, 15))
    axs = axs.flatten()

    for ax in axs:
        # Randomly select a subject
        sample_idx = np.random.choice(acc.shape[0])
        acc2plot = acc[sample_idx]
        gait_speed_sample = gait_speed[sample_idx]
        turn_coverages_sample = turn_coverages[sample_idx]
        turn_counts_sample = turn_counts[sample_idx]
        # Plot the acceleration data
        ax.plot(acc2plot)
        ax.set_title(
            f'Speed: {gait_speed_sample:.2f}, Turn Coverages: {turn_coverages_sample}, Turn Counts: {turn_counts_sample}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Acceleration')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def plot_bland_altman(labels, preds, groups, save_path=None):
    # Compute the differences and means
    diffs = preds - labels
    means = labels

    # Extract unique group categories from `groups`
    group_categories = [g.split('_')[0] for g in groups]
    unique_groups = list(set(group_categories))

    # Map groups to colors
    palette = sns.color_palette("colorblind", len(unique_groups))
    group_colors = {group: palette[i] for i, group in enumerate(unique_groups)}

    # Prepare the plot
    plt.figure(figsize=(10, 8))

    # Scatter plot for each group
    for group in unique_groups:
        mask = np.array(group_categories) == group
        plt.scatter(
            means[mask], diffs[mask],
            color=group_colors[group],
            label=group,
            edgecolors="black",
            linewidths=0.8,
            s=100,
            alpha=0.8
        )

    # Regression line
    reg = LinearRegression().fit(means.reshape(-1, 1), diffs)
    x_vals = np.linspace(min(means), max(means), 100)
    y_vals = reg.predict(x_vals.reshape(-1, 1))
    plt.plot(x_vals, y_vals, linestyle="--", color="black", label=f"Regression: R = {reg.coef_[0]:.2f}, p < 0.001")

    # Mean and ±1.96 SD lines
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    plt.axhline(mean_diff, color="black", linewidth=1.5, label=f"Mean {mean_diff:.2f}")
    plt.axhline(mean_diff + 1.96 * std_diff, color="black", linestyle="dotted",
                label=f"+1.96 SD {mean_diff + 1.96 * std_diff:.2f}")
    plt.axhline(mean_diff - 1.96 * std_diff, color="black", linestyle="dotted",
                label=f"-1.96 SD {mean_diff - 1.96 * std_diff:.2f}")

    # Customize the plot
    plt.ylim(-70, 70)
    plt.xlabel("Reference [cm/s]", fontsize=14)
    plt.ylabel("Predictions - Reference [cm/s]", fontsize=14)
    # plt.title("Bland-Altman Plot", fontsize=16, weight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Show the plot
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_bland_altman_subjects(labels, preds, groups):
    # Compute the differences and means
    diffs = preds - labels
    means = labels

    # Extract unique subjects and group categories from `groups`
    subjects = np.array([g.split('_')[1] for g in groups])
    cohorts = np.array([g.split('_')[0] for g in groups])

    unique_subjects = np.unique(subjects)

    # Compute mean differences and means for each subject
    subject_mean_diffs = []
    subject_mean_means = []
    subject_cohorts = []

    for subject in unique_subjects:
        mask = subjects == subject
        subject_mean_diffs.append(np.mean(diffs[mask]))
        subject_mean_means.append(np.mean(means[mask]))
        subject_cohorts.append(cohorts[mask][0])

    subject_mean_diffs = np.array(subject_mean_diffs)
    subject_mean_means = np.array(subject_mean_means)

    # Map cohorts to colors
    unique_cohorts = list(set(subject_cohorts))
    palette = sns.color_palette("tab10", len(unique_cohorts))
    cohort_colors = {cohort: palette[i] for i, cohort in enumerate(unique_cohorts)}

    # Prepare the plot
    plt.figure(figsize=(8, 6))

    # Scatter plot for each cohort
    for cohort in unique_cohorts:
        mask = np.array(subject_cohorts) == cohort
        plt.scatter(
            subject_mean_means[mask], subject_mean_diffs[mask], color=cohort_colors[cohort], label=cohort, alpha=0.7
        )

    # Regression line
    reg = LinearRegression().fit(subject_mean_means.reshape(-1, 1), subject_mean_diffs)
    x_vals = np.linspace(min(subject_mean_means), max(subject_mean_means), 100)
    y_vals = reg.predict(x_vals.reshape(-1, 1))
    plt.plot(x_vals, y_vals, linestyle="--", color="black", label=f"Regression: R = {reg.coef_[0]:.2f}, p < 0.001")

    # Mean and ±1.96 SD lines
    mean_diff = np.mean(subject_mean_diffs)
    std_diff = np.std(subject_mean_diffs)
    plt.axhline(mean_diff, color="black", linewidth=1.5, label=f"Mean {mean_diff:.2f}")
    plt.axhline(mean_diff + 1.96 * std_diff, color="black", linestyle="dotted",
                label=f"+1.96 SD {mean_diff + 1.96 * std_diff:.2f}")
    plt.axhline(mean_diff - 1.96 * std_diff, color="black", linestyle="dotted",
                label=f"-1.96 SD {mean_diff - 1.96 * std_diff:.2f}")

    # Customize the plot
    plt.ylim(-0.85, 0.85)
    plt.xlabel("Reference [m/s]")
    plt.ylabel("Difference [m/s]")
    plt.title("Bland-Altman Plot by Subject")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_bland_altman_with_comparison(labels, preds, groups, other_model_mean, other_model_name, save_path=None):
    '''

    :param labels: need to be in cm (in the paper we used the bout labels,predictions, and groups using the window2bout function)
    :param preds: need to be in cm
    :param groups: can be obtained from the parsed data test set
    :param other_model_mean: 6 for lower back
    :param other_model_name: Lower Back
    :return:
    '''
    # Compute the differences and means
    diffs = preds - labels
    means = labels

    # Extract unique group categories from `groups`
    group_categories = [g.split('_')[0] for g in groups]
    unique_groups = list(set(group_categories))

    # Map groups to colors
    # palette = sns.color_palette("colorblind", len(unique_groups))
    # evenly spaced hues around the color wheel
    palette = sns.color_palette("hls", n_colors=len(unique_groups))
    palette = sns.color_palette("Set2", n_colors=len(unique_groups))

    group_colors = {group: palette[i] for i, group in enumerate(unique_groups)}

    # Prepare the plot
    plt.figure(figsize=(10, 8))

    # Scatter plot for each group
    for group in unique_groups:
        mask = np.array(group_categories) == group
        plt.scatter(
            means[mask], diffs[mask],
            color=group_colors[group],
            label=group,
            edgecolors="black",
            linewidths=0.8,
            s=100,
            alpha=0.65
        )

    # Calculate plot limits
    x_min, x_max = plt.xlim()
    x_text_pos = x_max * 0.67  # Position text at 75% of x-axis

    # Mean and ±1.96 SD lines for the current model
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    # Plot lines without labels
    plt.axhline(mean_diff, color="black", linewidth=1.5)
    plt.axhline(mean_diff + 1.96 * std_diff, color="black", linestyle="dotted")
    plt.axhline(mean_diff - 1.96 * std_diff, color="black", linestyle="dotted")

    # Add text annotations directly on the lines
    plt.text(x_text_pos, mean_diff, f' ElderNet Mean Error\n{mean_diff:.2f}',
             verticalalignment='top', horizontalalignment='left', fontsize=16, fontweight='bold')
    plt.text(x_text_pos, mean_diff + 1.96 * std_diff, f'+1.96 SD\n{mean_diff + 1.96 * std_diff:.2f}',
             verticalalignment='bottom', horizontalalignment='left', fontsize=13, fontweight='bold')
    plt.text(x_text_pos, mean_diff - 1.96 * std_diff, f'-1.96 SD\n{mean_diff - 1.96 * std_diff:.2f}',
             verticalalignment='top', horizontalalignment='left', fontsize=13, fontweight='bold')

    # Add the other model's mean
    plt.axhline(other_model_mean, color="red", linewidth=1.5, alpha=0.5)
    plt.text(x_text_pos , other_model_mean, f'{other_model_name} Mean Error\n{other_model_mean:.2f}',
             color='red', verticalalignment='bottom', horizontalalignment='left', fontsize=16, fontweight='bold')

    # Customize the plot
    plt.ylim(-70, 70)
    plt.xlabel("Reference [cm/s]", fontsize=16, fontweight='bold')
    plt.ylabel("Predictions - Reference [cm/s]", fontsize=16, fontweight='bold')

    # Only keep legend for scatter points
    plt.legend(fontsize=14.3)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()



####################################################################################################################
# Setup Model
####################################################################################################################

def get_config(cfg, model_type):
    if model_type is None:
        return cfg.model
    try:
        return getattr(cfg, model_type)
    except AttributeError:
        raise ValueError(f"Configuration for model type '{model_type}' not found")


def setup_model(
        net='Resnet',
        output_size=1,
        epoch_len=10,
        is_mtl=False,
        is_simclr=False,
        is_classification=False,
        is_regression=False,
        max_mu=None,
        num_layers_regressor=None,
        batch_norm=False,
        head=None,
        eldernet_linear_output=128,
        pretrained=False,
        trained_model_path=None,
        name_start_idx=0,
        device='cpu'):

    model = Resnet(
        output_size=output_size,
        epoch_len=epoch_len,
        is_mtl=is_mtl,
        is_simclr=is_simclr,
        is_classification=is_classification,
        is_regression=is_regression,
        max_mu=max_mu,
        num_layers_regressor=num_layers_regressor,
        batch_norm=batch_norm
    )

    if net == 'ElderNet':
        feature_extractor = model.feature_extractor
        feature_vector_size = feature_extractor[-1][0].out_channels
        model = ElderNet(feature_extractor,
                         head=head,
                         non_linearity=True,
                         linear_model_input_size=feature_vector_size,
                         linear_model_output_size=eldernet_linear_output,
                         output_size=output_size,
                         is_mtl=is_mtl,
                         is_simclr=is_simclr,
                         is_classification=is_classification,
                         is_dense=False,
                         is_regression=is_regression,
                         max_mu=max_mu,
                         num_layers_regressor=num_layers_regressor,
                         batch_norm=batch_norm
                         )

    if pretrained or trained_model_path is not None:
        load_weights(trained_model_path, model, device, name_start_idx)

    return copy.deepcopy(model).to(device, dtype=torch.float)


def load_weights(weight_path, model, my_device="cpu", name_start_idx=0):
    pretrained_dict = torch.load(weight_path, map_location=my_device)
    pretrained_dict_v2 = copy.deepcopy(
        pretrained_dict
    )  # v2 has the right para names
    model_dict = model.state_dict()

    # change the name of the keys in the pretrained model to align with our convention
    for key in pretrained_dict:
        para_names = key.split(".")
        new_key = ".".join(para_names[name_start_idx:])
        pretrained_dict_v2[new_key] = pretrained_dict_v2.pop(key)

    # 1. Check if the model has a classifier module
    has_classifier = hasattr(model, 'classifier')
    # Filter out unnecessary keys
    if has_classifier:
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict_v2.items()
            if k in model_dict
        }
    else:
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict_v2.items()
            if k in model_dict and k.split(".")[0] != "classifier"
        }
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print("%d Weights loaded" % len(pretrained_dict))


def get_sslnet(harnet, tag='v1.0.0', pretrained=False, class_num=1, is_classification=False, is_regression=False):
    """
    Load and return the Self Supervised Learning (SSL) model from pytorch hub.
    :param str harnet: The pretrained network that correspond to the current input size (5/10/30 seconds)
    :param str tag: Tag on the ssl-wearables repo to check out
    :param bool pretrained: Initialise the model with UKB self-supervised pretrained weights.
    :return: pytorch SSL model
    :rtype: nn.Module
    """

    repo_name = 'ssl-wearables'
    repo = f'OxWearables/{repo_name}:{tag}'

    if not torch_cache_path.exists():
        Path.mkdir(torch_cache_path, parents=True, exist_ok=True)

    torch.hub.set_dir(str(torch_cache_path))

    # find repo cache dir that matches repo name and tag
    cache_dirs = [f for f in torch_cache_path.iterdir() if f.is_dir()]
    repo_path = next((f for f in cache_dirs if repo_name in f.name and tag in f.name), None)

    if repo_path is None:
        repo_path = repo
        source = 'github'
    else:
        repo_path = str(repo_path)
        source = 'local'
        if verbose:
            print(f'Using local {repo_path}')

    sslnet: nn.Module = torch.hub.load(repo_path, harnet, trust_repo=True, source=source, class_num=class_num,
                                       pretrained=pretrained, is_classification=is_classification,
                                       is_regression=is_regression, verbose=verbose)
    sslnet
    return sslnet


####################################################################################################################
# Training
####################################################################################################################
def create_cosine_decay_with_warmup(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def evaluate_model(model, val_loader, device, loss_fn):
    model.eval()
    losses = []
    maes = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.float).unsqueeze(1)
            logits = model(x)
            loss = loss_fn(logits.float(), y.float())
            losses.append(loss.item())
            val_mae = mean_absolute_error(y.cpu().detach(), logits.cpu().detach())
            maes.append(val_mae)
    return np.mean(losses), np.mean(maes)


####################################################################################################################
# Testing
####################################################################################################################
class TestDataset(Dataset):
    def __init__(self, X, y=None, groups=None):
        self.X = X.astype("f4")  # PyTorch defaults to float32
        self.y = y.astype("f4") if y is not None else None
        self.groups = groups
        print(f"test set sample count: {len(self.X)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx, :]

        if self.y is not None:
            label = self.y[idx]
        else:
            label = np.NaN

        if self.groups is not None:
            groups = self.groups[idx]
        else:
            groups = np.NaN

        return sample, label, groups


def test_model(model, test_loader, device, inference_mode=False):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device)
            predictions= model(inputs)

            if not inference_mode:
                all_labels.extend(labels.numpy())

            all_preds.extend(predictions.cpu().numpy().squeeze())

    if inference_mode:
        return np.array(all_preds)
    else:
        return (np.array(all_labels), np.array(all_preds))
