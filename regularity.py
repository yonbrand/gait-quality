import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def xcov(x, biased=False):
    """
    Computes the auto-covariance of an array (similar to MATLAB's xcov).
    Parameters:
        x (array-like): Input array.
        biased (bool): Whether to compute the biased estimate (default is unbiased).
    Returns:
        c (numpy.ndarray): Auto-covariance values.
        lags (numpy.ndarray): Lags corresponding to the auto-covariance values.
    """
    n = len(x)
    mean_x = np.mean(x)
    x_centered = x - mean_x
    # Full cross-correlation
    c_full = np.correlate(x_centered, x_centered, mode='full')
    # Normalize
    if biased:
        c = c_full / n
    else:
        lags = np.arange(-n + 1, n)
        normalization_factors = n - np.abs(lags)
        c = c_full / normalization_factors
    # Lags
    lags = np.arange(-n + 1, n)
    return c, lags

def smooth(x, fs):
    # Window size
    window_size = int(0.2 * fs)

    # Ensure the window size is odd (for symmetry)
    if window_size % 2 == 0:
        window_size += 1

    # Smooth using pandas (moving average)
    smoothed_x = pd.Series(x).rolling(window=window_size, center=True).mean().to_numpy()

    # Handle NaN values at the edges (if needed)
    # Example: fill with original values
    smoothed_x[np.isnan(smoothed_x)] = x[np.isnan(smoothed_x)]

    return smoothed_x


def correct_peaks(data, pks, locs):
    """
    Correct peaks found in a filtered signal to match the peaks in the raw signal.

    Parameters:
    data (array-like): Original data (raw data).
    pks (array-like): Values of the peaks in the filtered signal.
    locs (array-like): Indices of the peaks in the filtered signal.

    Returns:
    tuple: Corrected peak values and indices in the raw signal.
    """

    # If there are fewer than two peaks, skip correction.
    if len(locs) < 2:
        return pks, locs

    # Define search window size
    locale_win = int(np.ceil(0.2 * np.median(np.diff(locs))))

    # Remove peaks too close to the start/end of the recording
    valid_mask = (locs > locale_win) & (locs < (len(data) - locale_win))
    pks = pks[valid_mask]
    locs = locs[valid_mask]

    # Adjust peaks to align with the raw data
    for peak_ind in range(len(locs)):
        search_start = locs[peak_ind] - locale_win
        search_end = locs[peak_ind] + (locale_win // 2) + 1
        search_window = data[search_start:search_end]

        max_val = np.max(search_window)
        max_idx = np.argmax(search_window)

        pks[peak_ind] = max_val
        locs[peak_ind] = search_start + max_idx

    # Remove peaks that are too close due to the correction
    close_peaks = np.where(np.diff(locs) < locale_win)[0]
    for index in close_peaks[::-1]:
        if pks[index] > pks[index + 1]:
            pks = np.delete(pks, index + 1)
            locs = np.delete(locs, index + 1)
        else:
            pks = np.delete(pks, index)
            locs = np.delete(locs, index)

    return pks, locs

def calc_regularity(acc, sample_rate):
    """
    Calculate gait regularity from lower-back acceleration data.
    This is a python version of the MATLAB code, found here:
    "N:\Projects\Mobilise-D\WP2\Secondary Gait Metrics Codes\SO environment\SO_driver\Library\SO_functions\RegularitySymmetry.m"
    :param acc: array-like in the shape of (n,). Representing the acceleration magnitude.
    :param sample_rate:
    :return: Stride regularity, measure of the gait consistency
    """
    c, lags = xcov(acc, biased=True)
    normalized_c = c / c.max()
    normalized_c = normalized_c[lags >= 0]
    lags = lags[lags >= 0 ]
    Smoothed_normalized_c = smooth(normalized_c, sample_rate)
    # Assume Smoothed_normalized_c is a 1D numpy array, and Sample_rate is defined
    min_peak_distance = sample_rate / 4
    # Use find_peaks with the distance parameter
    locs, properties = find_peaks(Smoothed_normalized_c, distance=min_peak_distance)
    # Extract peak values using the indices (locs)
    pks = Smoothed_normalized_c[locs]
    mask = pks > 0
    pks = pks[mask]
    locs = locs[mask]
    pks, locs = correct_peaks(normalized_c, pks, locs)
    if pks.size > 1:
        #Autocorrelation coefficient for neighboring steps
        step_regularity = pks[0]
        # Autocorrelation coefficient for neighboring strides
        stride_regularity = pks[1]
    else:
        stride_regularity = 0

    return stride_regularity


