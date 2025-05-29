import os
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from dateutil import parser
import numpy as np
from scipy.io import loadmat
import mat73
from scipy import signal
import pandas as pd
import logging
import torch
from scipy.stats import kurtosis, skew
from tqdm import tqdm
import hydra

from utils import loadmat, get_config, setup_model
from regularity import calc_regularity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Constants
RESAMPLED_HZ = 30
WINDOW_SEC = 10
WINDOW_OVERLAP_SEC = 9
WINDOW_LEN = int(RESAMPLED_HZ * WINDOW_SEC)
WINDOW_STEP_LEN = WINDOW_LEN - int(RESAMPLED_HZ * WINDOW_OVERLAP_SEC)
MIN_DATA = 0
NUM_BATCHES = 500


def set_model_to_eval(*models):
    for model in models:
        model.eval()


def load_mat_file(file):
    try:
        data = loadmat(file)
    except NotImplementedError:
        data = mat73.loadmat(file)
    except Exception as e:
        logging.error(f"Error loading file {file}: {str(e)}", exc_info=True)
        return None
    return data


def detect_non_wear_time(acceleration_data: np.ndarray, device_fs,
                         threshold_std: float = 0.01) -> np.ndarray:
    window_size = int(device_fs * 60 * 60)
    n_samples = acceleration_data.shape[0]
    n_segments = n_samples // window_size
    acceleration_seg = acceleration_data[:n_segments * window_size]

    acceleration_win = acceleration_seg.reshape((n_segments, window_size, 3))
    std_in_windows = np.std(acceleration_win, axis=1)
    mask_all_axes = np.any(std_in_windows > threshold_std, axis=1)

    final_mask = np.zeros(acceleration_data.shape[0])
    mask_ind = np.repeat(mask_all_axes, window_size)
    final_mask[:len(mask_ind)] = mask_ind
    return final_mask


def time_synch(start_date, fs):
    """Calculate samples until midnight of the first recording day and return the midnight datetime."""
    midnight = datetime.combine(start_date.date() + timedelta(days=1), datetime.min.time())
    time_diff = midnight - start_date
    return int(time_diff.total_seconds() * fs), midnight


def resample_data(data, original_fs, target_fs):
    """Resample data to target frequency."""
    num_samples = int(len(data) * target_fs / original_fs)
    return signal.resample(data, num_samples)


# Function to process gait regularity using signal processing
def process_signal_regularity(walking_batch, fs):
    regularity_scores = np.array([calc_regularity(np.linalg.norm(segment, axis=1), fs) for segment in walking_batch])
    return regularity_scores.reshape(-1, 1)


def process_batch(batch, model, device):
    X = torch.tensor(batch, dtype=torch.float32).to(device)
    if X.shape[1] != 3:
        X = X.transpose(1, 2)

    with torch.no_grad():
        predictions = model(X)
        # Gait detection output is 2 columns of logits
        if predictions.shape[1] == 2:
            predictions = torch.argmax(predictions, dim=1)

    return predictions.cpu().numpy()


def calc_second_prediction(pred_walk, window_sec, second_predictions):
    '''
    Map window predictions to seconds based on the middle-second approach
    :param pred_walk: numpy array with the prediction for each window
    :param window_sec: number of seconds per window
    :param second_predictions: zeros nump array for the second-level predictions
    :return: second-level predictions
    '''
    for idx, window_prediction in enumerate(pred_walk):
        center_second = idx + window_sec // 2  # Calculate center second of the window
        if center_second < len(second_predictions):
            second_predictions[center_second] = window_prediction
    return second_predictions


def majority_vote(predictions, window_len, step_len, signal_len):
    """
    Assign predictions to each second using majority vote from overlapping windows.

    :param predictions: List or array of window predictions (0 or 1 for gait detection).
    :param window_len: Length of each window in samples.
    :param step_len: Step length between consecutive windows in samples.
    :param signal_len: Total length of the signal in samples.
    :return: Array of second-wise predictions.
    """
    seconds = signal_len // step_len  # Estimate number of seconds based on step length
    second_predictions = np.zeros(seconds)

    for sec in range(seconds):
        overlapping_window_indices = [
            i for i in range(len(predictions))
            if (sec * step_len) < (i * step_len + window_len) and (sec * step_len) >= (i * step_len)
        ]

        if overlapping_window_indices:
            # Majority vote among overlapping window predictions
            overlapping_preds = [predictions[i] for i in overlapping_window_indices]
            second_predictions[sec] = np.round(np.mean(overlapping_preds))
        else:
            second_predictions[sec] = 0  # Default to 0 if no overlapping windows

    return second_predictions


def merge_gait_bouts(pred_walk, sec_per_sample, min_bout_duration=5, merge_interval=3):
    """
    Post-process gait predictions to merge close bouts and filter short bouts.

    Args:
        pred_walk (np.ndarray): Binary array of walking predictions (1 = walking, 0 = not walking).
        fs (int): Sampling frequency of the predictions (Hz).
        min_bout_duration (int): Minimum duration for a gait bout (in seconds).
        merge_interval (int): Maximum gap duration to merge bouts (in seconds).

    Returns:
        np.ndarray: Post-processed binary array of walking predictions.
    """
    min_bout_samples = int(min_bout_duration * sec_per_sample)
    merge_gap_samples = int(merge_interval * sec_per_sample)

    # Detect start and end indices of gait bouts
    diff_preds = np.diff(np.concatenate([[0], pred_walk, [0]]))  # Add edges to detect transitions
    bout_starts = np.where(diff_preds == 1)[0]
    bout_ends = np.where(diff_preds == -1)[0]

    merged_pred_walk = pred_walk.copy()

    # Iterate through bouts and process gaps
    new_bouts = []
    for i in range(len(bout_starts) - 1):
        start, end = bout_starts[i], bout_ends[i]
        next_start = bout_starts[i + 1]

        # Check if the gap between this bout and the next is within the merge threshold
        if next_start - end <= merge_gap_samples:
            # Merge current bout with the next
            bout_ends[i] = bout_ends[i + 1]
            bout_starts[i + 1] = bout_starts[i]
        else:
            new_bouts.append((start, end))

    # Add the last bout
    if bout_starts[-1] != bout_ends[-1]:
        new_bouts.append((bout_starts[-1], bout_ends[-1]))

    # Apply the new bouts to the prediction array
    merged_pred_walk[:] = 0
    for start, end in new_bouts:
        # Filter out short bouts
        if end - start >= min_bout_samples:
            merged_pred_walk[start:end] = 1

    return merged_pred_walk


def detect_bouts(pred_walk):
    """Detect walking bouts from predicted walk data."""
    diff_preds = np.diff(pred_walk, prepend=0, append=0)
    where_bouts_start = np.where(diff_preds == 1)[0]
    where_bouts_end = np.where(diff_preds == -1)[0]

    bouts_id = np.zeros_like(pred_walk)
    for bout_idx, (start, end) in enumerate(zip(where_bouts_start, where_bouts_end), 1):
        bouts_id[start:end] = bout_idx

    return bouts_id


def calculate_statistics(result, fs, win_step_len):
    # Calculate time metrics once
    seconds_per_sample = win_step_len / fs
    samples_per_day = int(24 * 60 * 60 / seconds_per_sample)
    days = len(result['pred_walk']) // samples_per_day

    # Calculate daily walking statistics
    daily_walking = np.array_split(result['pred_walk'], days)
    daily_walk_amounts = [sum(day) * seconds_per_sample / 60 for day in daily_walking]

    # Pre-calculate bout masks for all unique bouts
    unique_bouts = np.unique(result['bouts_id'])
    bout_stats = {
        'speeds': [],
        'cadences': [],
        'gait_lengths': [],
        'gait_lengths_indirect': [],
        'regularity_eldernet': [],
        'regularity_sp': []
    }

    # Calculate bout statistics in a single loop
    for bout_id in unique_bouts:
        bout_mask = result['bouts_id'] == bout_id
        for metric, pred_key in [
            ('speeds', 'pred_speed'),
            ('cadences', 'pred_cadence'),
            ('gait_lengths', 'pred_gait_length'),
            ('gait_lengths_indirect', 'pred_gait_length_indirect'),
            ('regularity_eldernet', 'pred_regularity_eldernet'),
            ('regularity_sp', 'pred_regularity_sp')
        ]:
            bout_stats[metric].append(np.median(result[pred_key][bout_mask]))

    def calc_stats(data, prefix='', save_all=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        data = data.flatten()

        if len(data) == 0:
            return {f'{prefix}{stat}': np.nan for stat in
                    ['median', 'mean', 'std', 'p90', 'kurtosis', 'skewness',
                     'var', 'peak_to_peak', 'hist_values', 'hist_bins']}

        stats_dict = {
            f'{prefix}median': np.median(data),
            f'{prefix}mean': np.mean(data),
            f'{prefix}std': np.std(data),
            f'{prefix}p10': np.percentile(data, 10),
            f'{prefix}p20': np.percentile(data, 20),
            f'{prefix}p30': np.percentile(data, 30),
            f'{prefix}p40': np.percentile(data, 40),
            f'{prefix}p50': np.percentile(data, 50),
            f'{prefix}p60': np.percentile(data, 60),
            f'{prefix}p70': np.percentile(data, 70),
            f'{prefix}p80': np.percentile(data, 80),
            f'{prefix}p90': np.percentile(data, 90),
            f'{prefix}kurtosis': kurtosis(data),
            f'{prefix}skewness': skew(data),
            f'{prefix}var': np.var(data),
            f'{prefix}peak_to_peak': np.ptp(data)
        }
        if save_all:
            stats_dict[f'{prefix}all_values'] = json.dumps(data.tolist())
        return stats_dict

    return {
        'sub_id': result['subject_id'],
        'wear_days': result['wear_days'],
        **calc_stats(daily_walk_amounts, 'daily_walking_'),
        **calc_stats(result['bouts_durations'], 'bout_duration_', save_all=True),
        **calc_stats(result['pred_speed'], 'gait_speed_', save_all=True),
        **calc_stats(bout_stats['speeds'], 'bout_gait_speed_', save_all=True),
        **calc_stats(result['pred_cadence'], 'cadence_', save_all=True),
        **calc_stats(bout_stats['cadences'], 'bout_cadence_', save_all=True),
        **calc_stats(result['pred_gait_length'], 'gait_length_', save_all=True),
        **calc_stats(bout_stats['gait_lengths'], 'bout_gait_length_', save_all=True),
        **calc_stats(result['pred_gait_length_indirect'], 'gait_length_indirect_', save_all=True),
        **calc_stats(bout_stats['gait_lengths_indirect'], 'bout_gait_length_indirect_', save_all=True),
        **calc_stats(result['pred_regularity_eldernet'], 'regularity_eldernet_', save_all=True),
        **calc_stats(bout_stats['regularity_eldernet'], 'bout_regularity_eldernet_', save_all=True),
        **calc_stats(result['pred_regularity_sp'], 'regularity_sp_', save_all=True),
        **calc_stats(bout_stats['regularity_sp'], 'bout_regularity_sp_', save_all=True),
        # --- New PA features statistics ---
        **calc_stats(result['daily_pa_mean'], 'daily_pa_mean_', save_all=True),
        **calc_stats(result['daily_pa_std'], 'daily_pa_std_', save_all=True),
        **calc_stats(result['daily_pa_max'], 'daily_pa_max_', save_all=True),
        **calc_stats(result['daily_pa_min'], 'daily_pa_min_', save_all=True),
        **calc_stats(result['bout_pa_mean'], 'bout_pa_mean_', save_all=True),
        **calc_stats(result['bout_pa_std'], 'bout_pa_std_', save_all=True)
    }


def process_and_analyze_subject(file, gait_detection_model, gait_speed_model, cadence_model, gait_length_model,
                                regularity_model, device, sensor_device,
                                fs, win_step_len):
    try:
        data = load_mat_file(file)
        if data is None:
            return None

        if sensor_device == 'GENEActive':
            values = data['values']
            if isinstance(values, np.ndarray) and values.shape == (1, 1):
                values = values[0, 0]

            acc = values['acc'].astype(float)
            device_fs = values['sampFreq']
            header_string = values['header']
            # Find the start time string from the header
            start_time_match = re.search(r'Start Time:(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):\d{3}', header_string)
            if start_time_match:
                start_time_str = start_time_match.group(1)

        elif sensor_device == 'Axivity':
            acc = data['New_Data'].astype(float)
            device_fs = 50  # 50 hz is the default sampling rate for Axivity
            # Modify the file name from the acceleration file to the info file
            info_path = file.with_name(file.stem + "_info" + file.suffix)  # Adds "_info" before the .mat extension
            info_path = info_path.parent.parent / "info" / info_path.name
            info = load_mat_file(info_path)
            start_time_str = info['fileinfo']['start']['str']

        # Find the starting date and converting to datetime
        start_date = parser.parse(start_time_str)
        # Get the day of the week as an integer (0 = Monday, 6 = Sunday)
        day_of_week = start_date.weekday()
        # We take data starting from the second day (starting from midnight)
        day_of_week = 0 if day_of_week == 6 else day_of_week + 1

        wear_time_mask = detect_non_wear_time(acceleration_data=acc, device_fs=device_fs)
        sync_start_point, midnight_start = time_synch(start_date, device_fs)
        acc = acc[sync_start_point:]
        wear_time_mask = wear_time_mask[sync_start_point:]

        samples_per_day = int(device_fs * 60 * 60 * 24)
        days_in_data = acc.shape[0] // samples_per_day
        acc = acc[:samples_per_day * days_in_data]
        wear_time_mask = wear_time_mask[:samples_per_day * days_in_data]

        wear_time_by_day = wear_time_mask.reshape(days_in_data, samples_per_day)
        daily_wear_percentage = wear_time_by_day.mean(axis=1)
        # Find complete wear days (e.g., >80% wear time)
        complete_wear_days = np.where(daily_wear_percentage > 0.8)[0]
        # Analyze data only if there are 3+ days
        wear_days_in_data = len(complete_wear_days)
        acc_wear = np.empty((0, 3))
        if wear_days_in_data >= 3:
            for day in complete_wear_days:
                acc_day = acc[samples_per_day * day: samples_per_day * (day + 1)]
                acc_wear = np.append(acc_wear, acc_day, axis=0)
        else:
            return None

        # Resample data to target frequency (fs)
        resampled_acc = resample_data(acc_wear, device_fs, fs)

        # --- New: Compute day-level physical activity (PA) measures ---
        # Calculate the vector magnitude of acceleration for PA features
        acc_magnitude = np.linalg.norm(resampled_acc, axis=1)
        samples_per_day_resampled = int(24 * 60 * 60 * fs)
        num_days = resampled_acc.shape[0] // samples_per_day_resampled
        if num_days > 0:
            acc_magnitude = acc_magnitude[:num_days * samples_per_day_resampled]
            daily_pa = acc_magnitude.reshape(num_days, samples_per_day_resampled)
            daily_pa_mean = daily_pa.mean(axis=1)
            daily_pa_std = daily_pa.std(axis=1)
            daily_pa_max = daily_pa.max(axis=1)
            daily_pa_min = daily_pa.min(axis=1)
        else:
            daily_pa_mean = []
            daily_pa_std = []
            daily_pa_max = []
            daily_pa_min = []

        acc_win_all = np.array(
            [resampled_acc[i:i + WINDOW_LEN] for i in range(0, len(resampled_acc) - WINDOW_LEN + 1, WINDOW_STEP_LEN)]
        )

        # Process data in batches
        batch_size = acc_win_all.shape[0] // NUM_BATCHES
        pred_walk = []
        for i in range(0, len(acc_win_all), batch_size):
            batch = acc_win_all[i:i + batch_size]
            batch_pred_walk = process_batch(batch, gait_detection_model, device)
            pred_walk.extend(batch_pred_walk)

        pred_walk = np.array(pred_walk)

        # Map window predictions to seconds based on the middle-second approach
        second_predictions = np.zeros(len(resampled_acc) // fs, dtype=int)
        second_predictions_walk = calc_second_prediction(pred_walk, WINDOW_SEC, second_predictions)
        # Merge near bouts
        sec_per_sample = win_step_len / fs  # 1 for 90% overlap, 10 for no overlap
        merged_predictions_walk = merge_gait_bouts(second_predictions_walk, sec_per_sample, 10, 3)
        flat_predictions = np.repeat(merged_predictions_walk, fs)
        bouts_id = detect_bouts(flat_predictions)
        unq_bouts = np.unique(bouts_id[bouts_id != 0])
        bouts_win_all = []
        walking_bouts_id = []
        bouts_durations = []

        # --- New: Initialize lists for bout-level PA measures ---
        bout_pa_means = []
        bout_pa_stds = []

        for bout in unq_bouts:
            bout_predictions = np.where(bouts_id == bout)[0]
            acc_bout = resampled_acc[bout_predictions]
            bout_win = np.array(
                [acc_bout[i:i + WINDOW_LEN] for i in range(0, len(acc_bout) - WINDOW_LEN + 1, WINDOW_STEP_LEN)]
            )
            bout_duration = int(len(acc_bout) / fs)
            bouts_win_all.append(bout_win)
            current_bout_id = np.ones(bout_win.shape[0]) * bout
            walking_bouts_id.append(current_bout_id)
            bouts_durations.append(bout_duration)

            # --- New: Compute bout-level PA features ---
            bout_magnitude = np.linalg.norm(acc_bout, axis=1)
            bout_pa_means.append(np.mean(bout_magnitude))
            bout_pa_stds.append(np.std(bout_magnitude))

        if walking_bouts_id:
            walking_bouts_id = np.concatenate(walking_bouts_id)
        else:
            walking_bouts_id = np.array([])

        walking_batch = np.concatenate(bouts_win_all) if bouts_win_all else np.array([])

        if walking_batch.size > 0:
            # Process walking windows with gait quality models
            pred_speed = []
            pred_cadence = []
            pred_gait_length = []
            pred_gait_length_indirect = []
            pred_regularity_eldernet = []
            for i in range(0, len(walking_batch), batch_size):
                batch = walking_batch[i:i + batch_size]
                batch_pred_speed = process_batch(batch, gait_speed_model, device)
                batch_pred_cadence = process_batch(batch, cadence_model, device)
                batch_pred_gait_length = process_batch(batch, gait_length_model, device)
                batch_pred_regularity_eldernet = process_batch(batch, regularity_model, device)
                pred_speed.extend(batch_pred_speed)
                pred_cadence.extend(batch_pred_cadence)
                pred_gait_length.extend(batch_pred_gait_length)
                pred_regularity_eldernet.extend(batch_pred_regularity_eldernet)

            pred_speed = np.array(pred_speed)
            pred_cadence = np.array(pred_cadence)
            pred_gait_length = np.array(pred_gait_length)
            pred_gait_length_indirect = np.array(120 * pred_speed / pred_cadence)
            pred_regularity_eldernet = np.array(pred_regularity_eldernet)
            pred_regularity_sp = process_signal_regularity(walking_batch, fs)
        else:
            pred_speed = np.array([])
            pred_cadence = np.array([])
            pred_gait_length = np.array([])
            pred_gait_length_indirect = np.array([])
            pred_regularity_eldernet = np.array([])
            pred_regularity_sp = np.array([])

        file_path = Path(file)
        sub_id = '_'.join(file_path.stem.split('-')[:2])

        # Create result dictionary and include new PA features
        result = {
            'subject_id': sub_id,
            'wear_days': wear_days_in_data,
            'pred_walk': merged_predictions_walk,
            'pred_speed': pred_speed,
            'pred_cadence': pred_cadence,
            'pred_gait_length': pred_gait_length,
            'pred_gait_length_indirect': pred_gait_length_indirect,
            'pred_regularity_eldernet': pred_regularity_eldernet,
            'pred_regularity_sp': pred_regularity_sp,
            'bouts_id': walking_bouts_id,
            'bouts_durations': bouts_durations,
            # --- Add day-level PA features ---
            'daily_pa_mean': daily_pa_mean,
            'daily_pa_std': daily_pa_std,
            'daily_pa_max': daily_pa_max,
            'daily_pa_min': daily_pa_min,
            # --- Add bout-level PA features ---
            'bout_pa_mean': bout_pa_means,
            'bout_pa_std': bout_pa_stds
        }

        # Calculate statistics (including PA features)
        stats = calculate_statistics(result, fs, win_step_len)
        return stats

    except Exception as e:
        logging.error(f'Error processing file: {file}\nError message: {str(e)}')
    return None


def list_filtered_mat_files(directory):
    """List all .mat files excluding specific patterns."""
    return [
        file for file in Path(directory).rglob("*.mat")
        if not any(pattern in file.name for pattern in ["UpSideDown", "WearTime", "Temp.mat", "Time.mat", "info.mat"])
    ]


@hydra.main(config_path="conf", config_name="config_rush",
            version_base='1.1')
def main(cfg):
    data_files = Path(cfg.data.data_path)
    output_path = Path(cfg.data.log_path)
    output_file = os.path.join(output_path, cfg.data.output_filename)
    sensor_device = cfg.sensor_device  # Axivity or GENEActive
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the models
    gait_detection_cfg = get_config(cfg, model_type='gait_detection')
    gait_speed_cfg = get_config(cfg, model_type='gait_speed')
    cadence_model_cfg = get_config(cfg, model_type='cadence')
    gait_length_model_cfg = get_config(cfg, model_type='gait_length')
    regularity_model_cfg = get_config(cfg, model_type='regularity')

    gait_detection_model = setup_model(
        net=gait_detection_cfg.net,
        head=gait_detection_cfg.head if gait_detection_cfg.net == 'ElderNet' else None,
        eldernet_linear_output=gait_detection_cfg.feature_vector_size if gait_detection_cfg.net == 'ElderNet' else None,
        epoch_len=cfg.dataloader.epoch_len,
        is_classification=True,
        pretrained=gait_detection_cfg.pretrained,
        trained_model_path=gait_detection_cfg.trained_model_path,
        output_size=gait_detection_cfg.output_size,
        device=device)

    gait_speed_model = setup_model(
        net=gait_speed_cfg.net,
        head=gait_speed_cfg.head if gait_speed_cfg.net == 'ElderNet' else None,
        eldernet_linear_output=gait_speed_cfg.feature_vector_size if gait_speed_cfg.net == 'ElderNet' else None,
        epoch_len=cfg.dataloader.epoch_len,
        is_regression=gait_speed_cfg.is_regression,
        max_mu=gait_speed_cfg.max_mu,
        num_layers_regressor=gait_speed_cfg.num_layers_regressor,
        batch_norm=gait_speed_cfg.batch_norm,
        pretrained=gait_speed_cfg.pretrained,
        trained_model_path=gait_speed_cfg.trained_model_path,
        device=device)

    cadence_model = setup_model(
        net=cadence_model_cfg.net,
        head=cadence_model_cfg.head if cadence_model_cfg.net == 'ElderNet' else None,
        eldernet_linear_output=cadence_model_cfg.feature_vector_size if cadence_model_cfg.net == 'ElderNet' else None,
        epoch_len=cfg.dataloader.epoch_len,
        is_regression=cadence_model_cfg.is_regression,
        max_mu=cadence_model_cfg.max_mu,
        num_layers_regressor=cadence_model_cfg.num_layers_regressor,
        batch_norm=cadence_model_cfg.batch_norm,
        pretrained=cadence_model_cfg.pretrained,
        trained_model_path=cadence_model_cfg.trained_model_path,
        device=device)

    gait_length_model = setup_model(
        net=gait_length_model_cfg.net,
        head=gait_length_model_cfg.head if gait_length_model_cfg.net == 'ElderNet' else None,
        eldernet_linear_output=gait_length_model_cfg.feature_vector_size if gait_length_model_cfg.net == 'ElderNet' else None,
        epoch_len=cfg.dataloader.epoch_len,
        is_regression=gait_length_model_cfg.is_regression,
        max_mu=gait_length_model_cfg.max_mu,
        num_layers_regressor=gait_length_model_cfg.num_layers_regressor,
        batch_norm=gait_length_model_cfg.batch_norm,
        pretrained=gait_length_model_cfg.pretrained,
        trained_model_path=gait_length_model_cfg.trained_model_path,
        device=device)

    regularity_model = setup_model(
        net=regularity_model_cfg.net,
        head=regularity_model_cfg.head if regularity_model_cfg.net == 'ElderNet' else None,
        eldernet_linear_output=regularity_model_cfg.feature_vector_size if regularity_model_cfg.net == 'ElderNet' else None,
        epoch_len=cfg.dataloader.epoch_len,
        is_regression=regularity_model_cfg.is_regression,
        max_mu=regularity_model_cfg.max_mu,
        num_layers_regressor=regularity_model_cfg.num_layers_regressor,
        batch_norm=regularity_model_cfg.batch_norm,
        pretrained=regularity_model_cfg.pretrained,
        trained_model_path=regularity_model_cfg.trained_model_path,
        device=device)

    set_model_to_eval(gait_detection_model, gait_speed_model, cadence_model, gait_length_model, regularity_model)

    # Get list of files to process
    files_to_process = list_filtered_mat_files(data_files)

    # Process files
    results = []
    for file in tqdm(files_to_process, desc="Processing files"):
        result = process_and_analyze_subject(file, gait_detection_model, gait_speed_model, cadence_model,
                                             gait_length_model, regularity_model, device, sensor_device,
                                             RESAMPLED_HZ, WINDOW_STEP_LEN)
        if result is not None:
            results.append(result)
            df_results = pd.DataFrame(results)
            df_results.to_csv(output_file, index=False)
            for col in df_results.columns:
                if "all_values" in col:
                    df_results[col] = df_results[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (list, np.ndarray)) else x)
        # Clear GPU memory
        torch.cuda.empty_cache()

    # Save the DataFrame
    df_results = pd.DataFrame(results)
    for col in df_results.columns:
        if "all_values" in col:
            df_results[col] = df_results[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, np.ndarray)) else x)

    df_results.to_csv(output_file, index=False)

    logging.info(f"Processing complete. Results saved to {output_file}")


if __name__ == '__main__':
    main()
