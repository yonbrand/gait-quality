"""
Prepare Mobilise-D Dataset for ElderNet Fine-Tuning

Data hierarchy: <COHORT>/<PARTICIPANT_ID>/Free-living/data.mat

Cohorts: CHF, COPD, HA, PD, PFF, MS
    Original dataset: https://zenodo.org/records/13899386

This script extracts wrist and lower-back acceleration data and spatio-temporal gait annotations,
generating 10-second overlapping windows for downstream model training.

"""

import os
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.io.matlab import mat_struct
from scipy import signal
from regularity import calc_regularity

# -------------------- Configuration --------------------

INPUT_PATH = '<>'
OUTPUT_PATH = 'ten_seconds_windows_overlap_9sec_0.5nan_ratio'
OUTPUT_PATH_TRAIN = os.path.join(OUTPUT_PATH, 'Train')
OUTPUT_PATH_TEST = os.path.join(OUTPUT_PATH, 'Test')

COHORTS = ['CHF', 'COPD', 'HA', 'MS', 'PD', 'PFF']
WIN_LENGTH = 10          # seconds
OVERLAP = 9              # seconds
TARGET_FS = 30           # Hz
SEED = 42
TEST_SIZE = 0.25
NAN_RATIO = 0.5

# -------------------- Utility Functions --------------------

def loadmat(filename):
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(data):
    for key in data:
        if isinstance(data[key], mat_struct):
            data[key] = _todict(data[key])
    return data

def _todict(matobj):
    return {k: _todict(v) if isinstance(v, mat_struct) else v for k, v in matobj.__dict__.items()}

def turn_indices(turn_data, fs):
    return np.round(turn_data * fs).astype(int)

# -------------------- Gait Metrics --------------------

def strides_in_window(window, df):
    df_win = df[(df['end_time'] >= window.left) & (df['start_time'] < window.right)]

    if len(df_win) < 2:
        return df_win

    # Adjust start if second row is closer to window.left
    if abs(df_win.iloc[1]['start_time'] - window.left) < abs(df_win.iloc[0]['start_time'] - window.left):
        df_win = df_win.iloc[1:]

    if len(df_win) < 2:
        return df_win

    # Adjust end if second-to-last row is closer to window.right
    if abs(df_win.iloc[-2]['end_time'] - window.right) < abs(df_win.iloc[-1]['end_time'] - window.right):
        df_win = df_win.iloc[:-1]

    return df_win

def calculate_window_measure(window, df, field, nan_ratio=NAN_RATIO):
    df_window = strides_in_window(window, df)
    if (df_window[field].isna().sum() / df_window.shape[0]) > nan_ratio:
        return None
    return df_window[field].dropna().median()

def calculate_asymmetry(window, df):
    df_window = strides_in_window(window, df).copy()
    df_window['stride_time'] = df_window['end_time'] - df_window['start_time']
    right = df_window[df_window['LeftRight'] == 'Right']['stride_time'].dropna()
    left = df_window[df_window['LeftRight'] == 'Left']['stride_time'].dropna()
    if right.empty or left.empty:
        return None
    return 1 - min(right.mean(), left.mean()) / max(right.mean(), left.mean())

def calc_cadence(window, df):
    df_win = df[(df['end_time'] >= window.left) & (df['start_time'] < window.right)]
    step_times = np.unique(np.concatenate([df_win['start_time'].values, df_win['end_time'].values]))
    if len(step_times) < 2:
        return 0
    duration = (step_times[-1] - step_times[0]) / 100
    return ((len(step_times) - 1) / duration) * 60

# -------------------- Acceleration Metrics --------------------

def extract_acceleration_data(window, acc, win_length, original_fs=100, target_fs=30):
    start_idx, end_idx = int(window.left), int(window.right)
    window_data = acc[start_idx:end_idx]
    num_samples = int(win_length * target_fs / original_fs)
    return signal.resample(window_data, num_samples, axis=0)

def calculate_jerk(acc_window):
    return np.mean(np.abs(np.diff(acc_window, axis=0)))

def calculate_intensity(acc_window):
    sx, sz = acc_window[:, 1], acc_window[:, 2]
    return np.std(np.sqrt(sx ** 2 + sz ** 2))

def calculate_norm(acc_window):
    return np.mean(np.linalg.norm(acc_window, axis=1))

def count_turns_and_coverage(window, starts, ends):
    count, coverage = 0, 0
    for start, end in zip(np.atleast_1d(starts), np.atleast_1d(ends)):
        interval = pd.Interval(start, end, closed='left')
        if window.overlaps(interval):
            count += 1
            overlap = min(window.right, interval.right) - max(window.left, interval.left)
            coverage += overlap / (window.right - window.left)
    return count, coverage

# -------------------- Data Splitting --------------------

def stratified_group_train_val_split(groups, cohorts, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    train_idx, val_idx = [], []
    for cohort in np.unique(cohorts):
        mask = cohorts == cohort
        cohort_groups = groups[mask]
        unique_groups = np.unique(cohort_groups)
        n_val = int(len(unique_groups) * test_size)
        val_sub = np.random.choice(unique_groups, size=n_val, replace=False)
        train_sub = unique_groups[~np.isin(unique_groups, val_sub)]
        train_idx.extend(np.where(np.isin(groups, train_sub))[0])
        val_idx.extend(np.where(np.isin(groups, val_sub))[0])
    return np.array(train_idx), np.array(val_idx)

# -------------------- File Processing --------------------

def process_file(cohort, file, acc_all, avg_speeds, avg_lengths, asymmetry, turn_counts,
                 turn_coverages, subjects, bout_ids, cadence, jerk, intensity, norm, reg):

    try:
        data = loadmat(file)['data']
        wrist = data['TimeMeasure1']['Recording4']['SU_INDIP'].get('LeftWrist') \
                or data['TimeMeasure1']['Recording4']['SU_INDIP'].get('RightWrist')

        if wrist is None:
            return

        acc, fs = wrist['Acc'], wrist['Fs']['Acc']
        acc_lb = data['TimeMeasure1']['Recording4']['SU_INDIP']['LowerBack']['Acc']
        fs_lb = data['TimeMeasure1']['Recording4']['SU_INDIP']['LowerBack']['Fs']['Acc']

        if acc_lb is None:
            return

        subject_id = f"{cohort}_{os.path.basename(os.path.dirname(os.path.dirname(file)))}"
        win_samples = WIN_LENGTH * fs
        bouts = data['TimeMeasure1']['Recording4']['Standards']['INDIP']['MicroWB']

        for bout_num, bout in enumerate(bouts):
            try:
                strides = pd.DataFrame({
                    'start_time': bout.Stride_InitialContacts[:, 0],
                    'end_time': bout.Stride_InitialContacts[:, 1],
                    'speed': bout.Stride_Speed,
                    'length': bout.Stride_Length,
                    'LeftRight': bout.InitialContact_LeftRight[:len(bout.Stride_Speed)]
                })

                start, end = strides['start_time'].dropna().iloc[0], strides['end_time'].dropna().iloc[-1]
                windows = pd.interval_range(start=start, end=end - win_samples + (win_samples - OVERLAP * fs), freq=win_samples - OVERLAP * fs)

                for win in windows:
                    end = win.left + win_samples
                    current_window = pd.Interval(win.left, end)
                    speed = calculate_window_measure(current_window, strides, 'speed')
                    length = calculate_window_measure(current_window, strides, 'length')
                    asym = calculate_asymmetry(current_window, strides)

                    if speed is None or length is None:
                        continue

                    cadence_val = 120 * (speed / length)
                    acc_win = extract_acceleration_data(current_window, acc, win_samples, fs, TARGET_FS)
                    jerk_val = calculate_jerk(acc_win)
                    intensity_val = calculate_intensity(acc_win)
                    norm_val = calculate_norm(acc_win)

                    acc_lb_win = extract_acceleration_data(current_window, acc_lb, win_samples, fs_lb, TARGET_FS)
                    mag_lb = np.linalg.norm(acc_lb_win, axis=1)
                    regularity_val = calc_regularity(mag_lb, TARGET_FS)

                    turns, coverage = count_turns_and_coverage(current_window, turn_indices(bout.Turn_Start, fs), turn_indices(bout.Turn_End, fs))

                    acc_all = np.append(acc_all, acc_win[np.newaxis, :], axis=0)
                    avg_speeds, avg_lengths, asymmetry = np.append(avg_speeds, speed), np.append(avg_lengths, length), np.append(asymmetry, asym)
                    turn_counts, turn_coverages = np.append(turn_counts, turns), np.append(turn_coverages, coverage)
                    cadence, jerk, intensity, norm = np.append(cadence, cadence_val), np.append(jerk, jerk_val), np.append(intensity, intensity_val), np.append(norm, norm_val)
                    reg = np.append(reg, regularity_val)
                    subjects, bout_ids = np.append(subjects, subject_id), np.append(bout_ids, bout_num)

            except Exception as e:
                print(f"[ERROR] Bout {bout_num}: {e}")

        print(f"[INFO] Processed subject: {subject_id}")

    except Exception as e:
        print(f"[ERROR] File {file}: {e}")

    return acc_all, avg_speeds, avg_lengths, asymmetry, turn_counts, turn_coverages, subjects, bout_ids, cadence, jerk, intensity, norm, reg

# -------------------- Main Execution --------------------

def main():
    os.makedirs(OUTPUT_PATH_TRAIN, exist_ok=True)
    os.makedirs(OUTPUT_PATH_TEST, exist_ok=True)

    acc_all = np.empty((0, WIN_LENGTH * TARGET_FS, 3))
    avg_speeds = avg_lengths = asymmetry = turn_coverages = bout_ids = cadence = jerk = intensity = norm = reg = np.empty((0,))
    subjects = []
    turn_counts = np.empty((0,), dtype=int)

    for cohort in COHORTS:
        path = os.path.join(INPUT_PATH, cohort)
        files = [os.path.join(path, f, 'Free-living', 'data.mat') for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

        for file in files:
            if os.path.isfile(file):
                acc_all, avg_speeds, avg_lengths, asymmetry, turn_counts, turn_coverages, subjects, bout_ids, cadence, jerk, intensity, norm, reg = \
                    process_file(cohort, file, acc_all, avg_speeds, avg_lengths, asymmetry, turn_counts, turn_coverages, subjects, bout_ids, cadence, jerk, intensity, norm, reg)

    cohorts = np.asarray([s.split('_')[0] for s in subjects])
    train_idx, test_idx = stratified_group_train_val_split(np.array(subjects), cohorts, test_size=TEST_SIZE, random_state=SEED)

    # Save outputs
    data_dict = {
        "gait_speed": avg_speeds,
        "gait_length": avg_lengths,
        "acc": acc_all,
        "subjects": subjects,
        "turn_counts": turn_counts,
        "turn_coverages": turn_coverages,
        "bout_id": bout_ids,
        "cadence": cadence,
        "jerk": jerk,
        "intensity": intensity,
        "norm": norm,
        "regularity": reg,
        "asymmetry": asymmetry,
    }

    for name, data in data_dict.items():
        with open(os.path.join(OUTPUT_PATH_TRAIN, f"{name}.p"), 'wb') as f:
            pickle.dump(data[train_idx], f)
        with open(os.path.join(OUTPUT_PATH_TEST, f"{name}.p"), 'wb') as f:
            pickle.dump(data[test_idx], f)

if __name__ == "__main__":
    main()
