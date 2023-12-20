import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import classification_report  
from sklearn.linear_model import LogisticRegression
from scipy.signal import butter, filtfilt
from scipy.signal import detrend
from scipy.signal import find_peaks

def Vm_testplot (data, data_flt, title=''):
    """
    Plot raw and test data for comparison.

    Parameters:
    data (array-like): Sequence of raw data values.
    data_flt (array-like): Sequence of test data values.

    Returns:
    None: Displays the plot.
    """
    plt.figure(figsize=(25,10))
    plt.plot(data, label='Raw data')
    plt.plot(data_flt, label='Test data', color='blue')
    plt.xlabel('Time (ms)')  
    plt.ylabel('Membrane Potential (V)') 
    plt.title(title)
    plt.legend()

    plt.show() 

def spikelet_testplot (data, spikelets_index):
    """
    Plot data and highlight spikelets.

    Parameters:
    data (array-like): Data sequence to plot.
    spikelets_index (array-like): Indices of spikelets to highlight.

    Returns:
    None: 
    """
    plt.figure(figsize=(25,10))
    plt.plot(data, label='Raw data')
    plt.plot(spikelets_index, data[spikelets_index], 'o')
    plt.xlabel('Time (ms)')  
    plt.ylabel('Membrane Potential (V)') 
    plt.legend()

    plt.show()

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Design a bandpass filter.

    Parameters:
    lowcut (float): The lower cutoff frequency of the filter.
    highcut (float): The higher cutoff frequency of the filter.
    fs (float): The sampling rate of the data.
    order (int, optional): The order of the filter. Default is 5.

    Returns:
    b, a (tuple): Numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to the data.

    Parameters:
    data (array_like): The data to be filtered.
    lowcut (float): The lower cutoff frequency of the filter.
    highcut (float): The higher cutoff frequency of the filter.
    fs (float): The sampling rate of the data.
    order (int, optional): The order of the filter. Default is 5.

    Returns:
    y (array_like): The filtered data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def moving_avg_normalise(data, mvg_avg_window = 800):
    # Calculate moving average
    moving_avg = np.convolve(data, np.ones(mvg_avg_window), 'valid') / mvg_avg_window
    
    # Extend the moving average array to the same size as the original data by padding
    pad_start = mvg_avg_window // 2
    pad_end = (mvg_avg_window - 1) // 2
    moving_avg = np.concatenate((np.full(pad_start, moving_avg[0]), moving_avg, np.full(pad_end, moving_avg[-1])))
    
    # Subtract the moving average from the original data to locally normalize it
    normalized_Vm = data - moving_avg
    
    return normalized_Vm

def lower_bound(data, mvg_avg_window = 800, sub_window_size=10000):
    
    normalized_Vm = moving_avg_normalise(data, mvg_avg_window)
    min_std = np.inf  # Initialize the minimum standard deviation to infinity
    min_std_start = None  # Initialize the start index of the window with the minimum standard deviation
    
    # I had to introduce a step to make the function more efficient
    step = sub_window_size // 500
    
    # For each possible window in Vm
    for start in range(0, len(normalized_Vm) - sub_window_size + 1, step):
        # Calculate the standard deviation of the window
        std = np.std(normalized_Vm[start : start + sub_window_size])
        # If this standard deviation is lower than the current minimum
        if std < min_std:
            # Update the minimum standard deviation and the start index of the window
            min_std = std
            min_std_start = start

    # The sub-dataset of Vm with the lowest standard deviation is Vm[min_std_start : min_std_start + window_size]
    quiescence = normalized_Vm[min_std_start : min_std_start + sub_window_size]

    # Calculate the standard deviation of the membrane potential during this period
    std_dev = np.std(quiescence)

    # Calculate the average standard deviation of the entire signal
    avg_std_dev = np.std(normalized_Vm)

    # Calculate the coefficient
    bound = (0*avg_std_dev + 5*std_dev)

    # Set the lower threshold
    return bound

import numpy as np

def find_spikelet_moving_avg(Vm, Vm_thrs=-0.03, window_size=1000, lower_threshold=0.002, SR_Vm=20000):
    AP_Win = 0.0015  # time (s) to search for AP's peak
    AP_length = np.round(AP_Win * SR_Vm)

    normalized_Vm = moving_avg_normalise(Vm, window_size)
    All_Thrs_Onset = np.diff(np.divide(normalized_Vm - lower_threshold, np.abs(normalized_Vm - lower_threshold)))
    All_Thrs_Index, Peaks = find_peaks(All_Thrs_Onset, height=0.1, prominence=0.5, distance=SR_Vm * 0.001)

    spikelets_param_output = []
    spikelets_indices = []  # List to store the indices of the spikelets

    if (All_Thrs_Index.any()):
        for i in range(len(All_Thrs_Index)):
            pt1 = int(All_Thrs_Index[i])
            pt2 = int(All_Thrs_Index[i] + AP_length)

            if (pt2 < len(normalized_Vm) - 1):  
                AP_Seg = normalized_Vm[pt1:pt2]
                Ind = np.argmax(AP_Seg)
                
                if (Ind.any()):
                    AP_Index = pt1 + Ind - 1
                    if (AP_Index.any() and Vm[AP_Index] <= Vm_thrs):
                        # Calculate parameters
                        Thrs_Time = All_Thrs_Index[i] / SR_Vm
                        Peak_Time = AP_Index / SR_Vm
                        Peak_Vm = Vm[AP_Index]
                        Thrs_Vm = Vm[All_Thrs_Index[i]]
                        Spikelet_Amp = Peak_Vm - Thrs_Vm

                        # Add parameters to output and index to spikelets_indices
                        spikelets_param_output.append([Thrs_Time, Thrs_Vm, Peak_Time, Peak_Vm, Spikelet_Amp])
                        spikelets_indices.append(AP_Index)

    # Convert to numpy array for easy handling and remove NaNs
    spikelets_param_output = np.array(spikelets_param_output)
    spikelets_param_output = spikelets_param_output[~np.isnan(spikelets_param_output).all(axis=1), :]

    return spikelets_param_output, spikelets_indices

    

def local_lower_bound(data, mvg_avg_window = 800, sub_window_size=2000):
    # Calculate moving average
    normalized_Vm = moving_avg_normalise(data, mvg_avg_window)

    #calculate the std of each window of normalized Vm
    std_dev = []
    step = sub_window_size // 50
    for start in range(0, len(normalized_Vm) - sub_window_size + 1, step):
        std_dev.append(np.std(normalized_Vm[start : start + sub_window_size]))
    
    
    # Match the sizes of lower bound and Vm
    x = np.linspace(0, 1, len(std_dev))
    match = np.linspace(0, 1, len(Vm))
    std_dev = np.interp(match, x, std_dev)
    
    base_bound = lower_bound(data)
    lower_threshold = np.array(std_dev) + 1.5*base_bound

    return lower_threshold


def df_spikelet_converter(spikelet_params):
    """
    Converts spikelet parameters into a pandas DataFrame.

    Parameters:
    spikelet_params (numpy array): A 2D array where each row represents a spikelet and the columns represent the spikelet's parameters.

    Returns:
    DataFrame: A pandas DataFrame with the following columns:
        'spikelet_thresh_times': Threshold time of the spikelet
        'spikelet_thresh_vm': Membrane potential at the threshold
        'spikelet_peak_times': Peak time of the spikelet
        'spikelet_peak_vm': Membrane potential at the peak
        'spikelet_amp': Amplitude of the spikelet
    """
    return pd.DataFrame(spikelet_params, columns=['spikelet_thresh_times', 'spikelet_thresh_vm', 
                                                  'spikelet_peak_times', 'spikelet_peak_vm', 
                                                  'spikelet_amp'])


def calculate_spikelet_params(row):
    spikelet_params, _ = find_spikelet_moving_avg(row['Sweep_MembranePotential'], 
                                                  Vm_thrs=row['Cell_APThreshold_Slope'], 
                                                  SR_Vm=2000)

    if spikelet_params.size == 0:
        return pd.Series({
            'nbr_spikelets': 0,
            'avg_spikelet_duration': np.nan,
            'avg_spikelet_amp': np.nan,
            'avg_spikelet_thresh_vm': np.nan,
            'avg_consecutive_spikelet_interval': np.nan
        })

    df_spikelets = pd.DataFrame(spikelet_params, columns=['spikelet_thresh_times', 'spikelet_thresh_vm', 'spikelet_peak_times', 'spikelet_peak_vm', 'spikelet_amp'])

    return pd.Series({
        'nbr_spikelets': len(df_spikelets),
        'avg_spikelet_duration': df_spikelets['spikelet_peak_times'].mean() - df_spikelets['spikelet_thresh_times'].mean(),
        'avg_spikelet_amp': df_spikelets['spikelet_amp'].mean(),
        'avg_spikelet_thresh_vm': df_spikelets['spikelet_thresh_vm'].mean(),
        'avg_consecutive_spikelet_interval': df_spikelets['spikelet_thresh_times'].diff().mean()
    })

