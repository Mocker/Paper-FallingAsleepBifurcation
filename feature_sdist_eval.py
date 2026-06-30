import numpy as np
import scipy.io as sio
import pycatch22
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def evaluate_features_and_sdist():
    """
    Port of Feature_SDist_Eval.m
    Demonstrates computing EEG features, transforming to sleep distance,
    and visualizing.
    """
    # 1. Load the EEG data
    # (Assuming EEGProc.mat contains a variable named 'eeg1')
    try:
        mat_data = sio.loadmat('Core Algorithms and Examples/Feature and Sleep distance computation/EEGProc.mat')
        eeg1 = mat_data['eeg1'].flatten()
    except FileNotFoundError:
        print("Data files not found. Make sure to run this script from the root of the repository.")
        return

    # Parameters
    Fs = 256                           # sampling frequency
    overlap_min = 0.05                 # overlap between EEG epochs in min
    overlap_smp = int(overlap_min * 60 * Fs)   # number of datapoints for overlap samples (3 s)
    epc_size_smp = 2 * overlap_smp       # number of datapoints for a complete epoch (6 s)

    print(f"Total EEG samples: {len(eeg1)}")
    
    # 2. Extract Features
    # Note: Full feature extraction is slow. In MATLAB it was >10 minutes.
    # To replicate the skipping logic, we'll load the precomputed features
    # as the MATLAB script did to save time for demonstration.
    
    print("Loading precomputed features to demonstrate the sleep distance calculation...")
    try:
        # The MATLAB code loaded 'GrpFt_MeanStdnow.mat' for normalization constants
        grp_stats = sio.loadmat('Core Algorithms and Examples/Feature and Sleep distance computation/GrpFt_MeanStdnow.mat')
        mean_allft = grp_stats['mean_allft'].flatten()
        std_allft = grp_stats['std_allft'].flatten()
        
        # And it loaded the precomputed 'ft_mat_all' implicitly or from Features.mat
        # Actually the script says: "load Features.mat" if skipping.
        # But wait, it also evaluates `ft_mat_all = vertcat(ft_mat{:})`. We'll just load Features.mat.
        features_mat = sio.loadmat('Core Algorithms and Examples/Feature and Sleep distance computation/Features.mat')
        ft_mat = features_mat['ft_mat']
        # ft_mat is a 1xN object array of 1x50 feature arrays. Vertically stack them.
        ft_mat_all = np.vstack(ft_mat[0])
    except Exception as e:
        print(f"Error loading precomputed features: {e}")
        return

    # Calculate number of median epochs
    onset_mediantime = 10
    onsetepcs = int(onset_mediantime / overlap_min) - 1   
    
    # 3. Normalization
    # Z-score normalize all features to the group level mean and std
    ft_mat_all_norm = (ft_mat_all - mean_allft) / std_allft
    
    # Compute the centroid of the sleep by taking the median of 10 min post onset
    # In python, negative indexing counts from the end
    ft_onset_centroid = np.nanmedian(ft_mat_all_norm[-onsetepcs-1:, :], axis=0).reshape(1, -1)
    
    # Compute the sleep distance by taking the n-dimensional euclidean distance
    # pdist2 equivalent in scipy is cdist
    S_var = cdist(ft_mat_all_norm, ft_onset_centroid, metric='euclidean').flatten()
    
    # Smoothing the data: moving median over 20 points
    # MATLAB: smoothdata(S_var,1,"movmedian",[20,0]) -> looks back 20 points, 0 forward
    # pandas rolling median is equivalent
    s_var_series = pd.Series(S_var)
    S_var_smooth = s_var_series.rolling(window=21, min_periods=1).median().values
    
    xx_smooth = S_var_smooth - np.nanmin(S_var_smooth)
    
    # Time vector
    tvec_plot = np.arange(-29.9, 10 + overlap_min/2, overlap_min)
    
    # 4. Visualization
    try:
        hypno_data = sio.loadmat('Core Algorithms and Examples/Feature and Sleep distance computation/Hypnogram_This.mat')
        hyp_this = hypno_data['hyp_this'].flatten()
    except Exception as e:
        print(f"Could not load hypnogram: {e}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})
    
    # Top plot: Hypnogram
    ax1.plot(tvec_plot[:len(hyp_this)], hyp_this, color='k', linewidth=2)
    ax1.set_ylim([0, 3])
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['Awake', 'N1', 'N2', 'N3'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(direction='out', length=6, width=2)
    ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
    
    # Bottom plot: Sleep Distance
    ax2.plot(tvec_plot[:len(xx_smooth)], xx_smooth, color='b', linewidth=2)
    ax2.set_xlabel('Time (min)', fontsize=12)
    ax2.set_ylabel('Sleep distance', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(direction='out', length=6, width=2)
    
    plt.tight_layout()
    plt.savefig('Core Algorithms and Examples/Feature and Sleep distance computation/SleepDistance_Python_Plot.png')
    plt.close(fig)

if __name__ == "__main__":
    evaluate_features_and_sdist()
