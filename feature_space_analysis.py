import numpy as np
import scipy.spatial.distance as dist
import scipy.stats as stats
import pandas as pd

def rms(x):
    """Computes Root Mean Square of a signal."""
    return np.sqrt(np.nanmean(x**2))

def detect_artifacts(eeg_channels, Fs, epc_len=6, overlap=0.5, nrep=2, mul_rms=2.5):
    """
    Replicates the RMS-based artifact detection logic in FeatureDynam_Prep.m.
    
    Parameters:
    - eeg_channels: list of arrays, where each array is the EEG signal for a channel.
    - Fs: sampling frequency.
    - epc_len: epoch length in seconds.
    - overlap: overlap fraction.
    - nrep: number of recursive repetitions for thresholding.
    - mul_rms: multiplier for median RMS to define artifact threshold.
    
    Returns:
    - is_artifact: 1D boolean array indicating if an epoch contains artifacts.
    - steps_sbj: start indices of each epoch.
    """
    epc_lend = int(epc_len * Fs)
    jmp_lend = int(epc_lend - np.floor(epc_lend * overlap))
    
    all_ch_artifacts = []
    steps_sbj = None
    
    for eeg_chnow in eeg_channels:
        if eeg_chnow is None or len(eeg_chnow) == 0:
            continue
            
        len_eeg = len(eeg_chnow)
        num_epochs = int((len_eeg - epc_lend) // jmp_lend) + 1
        
        rms_ch = np.zeros(num_epochs)
        stps = np.zeros(num_epochs, dtype=int)
        
        for nep in range(num_epochs):
            stps_now = nep * jmp_lend
            eeg_epc = eeg_chnow[stps_now : stps_now + epc_lend]
            rms_ch[nep] = rms(eeg_epc)
            stps[nep] = stps_now
            
        if steps_sbj is None:
            steps_sbj = stps
            
        # Recursive artifact detection
        rms_ch_temp = rms_ch.copy()
        for _ in range(nrep):
            med_rms = np.nanmedian(rms_ch_temp)
            is_art_now = (rms_ch_temp > (mul_rms * med_rms))
            rms_ch_temp[is_art_now] = np.nan
            
        all_ch_artifacts.append(np.isnan(rms_ch_temp))
        
    if not all_ch_artifacts:
        return np.array([]), np.array([])
        
    # An epoch contains an artifact if any channel has an artifact
    is_artifact = np.any(np.vstack(all_ch_artifacts), axis=0)
    return is_artifact, steps_sbj

def bootstrap_indices(artifact_matrix, N_min=200, seed=42):
    """
    Replicates bootstrapping index generation in FeatureDynam_Prep.m.
    
    Parameters:
    - artifact_matrix: 2D boolean array (subjects x epochs), True if epoch is valid (no artifact).
    """
    num_sbjs, max_epochs = artifact_matrix.shape
    idx_bts = []
    
    # Calculate number of valid subjects per epoch
    num_valid_per_epoch = np.sum(artifact_matrix, axis=0)
    
    # Find when sample size is sufficient
    time_start_enough = np.where(num_valid_per_epoch >= N_min)[0]
    first_valid_idx = time_start_enough[0] if len(time_start_enough) > 0 else 0
    
    rng = np.random.default_rng(seed)
    
    for jj in range(max_epochs):
        if jj < first_valid_idx:
            idx_bts.append([])
            continue
            
        valid_sbj_idx = np.where(artifact_matrix[:, jj])[0]
        
        if len(valid_sbj_idx) < N_min:
            idx_bts.append(valid_sbj_idx)
        else:
            # Resample N_min subjects
            resampled = rng.choice(valid_sbj_idx, size=N_min, replace=False)
            idx_bts.append(resampled)
            
    return idx_bts, first_valid_idx

def compute_sleep_distance(features_zscored, onset_mark_indices):
    """
    Computes sleep distance s(t) for a subject.
    Replicates SleepDistanceS.m logic.
    
    Parameters:
    - features_zscored: 2D array of z-scored features (epochs x features).
    - onset_mark_indices: indices representing the sleep onset period.
    """
    # Calculate onset centroid (median of sleep onset period)
    onset_features = features_zscored[onset_mark_indices, :]
    onset_centroid = np.nanmedian(onset_features, axis=0).reshape(1, -1)
    
    num_epochs = features_zscored.shape[0]
    sleep_distance = np.full(num_epochs, np.nan)
    
    for jj in range(num_epochs):
        feat_vector = features_zscored[jj, :].reshape(1, -1)
        if np.any(np.isnan(feat_vector)):
            continue
        # Euclidean distance
        sleep_distance[jj] = dist.cdist(feat_vector, onset_centroid, metric='euclidean').flatten()[0]
        
    return sleep_distance

def compute_state_velocity(features_zscored):
    """
    Computes state velocity v(t) for a subject.
    Replicates StateVelocityVt.m logic.
    """
    num_epochs = features_zscored.shape[0]
    velocity = np.full(num_epochs, np.nan)
    
    for jj in range(1, num_epochs):
        feat_curr = features_zscored[jj-1, :].reshape(1, -1)
        feat_next = features_zscored[jj, :].reshape(1, -1)
        
        if np.any(np.isnan(feat_curr)) or np.any(np.isnan(feat_next)):
            continue
            
        # Euclidean distance between successive epochs
        velocity[jj] = dist.cdist(feat_curr, feat_next, metric='euclidean').flatten()[0]
        
    return velocity

def compute_sleep_velocity(features_zscored, onset_centroid, runwin=20):
    """
    Computes sleep velocity v_s(t) for a subject.
    Replicates SleepVelocityVs.m logic.
    """
    num_epochs = features_zscored.shape[0]
    sleep_velocity = np.full(num_epochs, np.nan)
    thetas = np.full(num_epochs, np.nan)
    projections = np.full(num_epochs, np.nan)
    
    for jj in range(num_epochs - runwin - 1):
        feat_curr_win = features_zscored[jj : jj + runwin + 1, :]
        feat_next_win = features_zscored[jj + 1 : jj + 1 + runwin + 1, :]
        
        # Average features in sliding window
        feat_curr = np.nanmean(feat_curr_win, axis=0)
        feat_next = np.nanmean(feat_next_win, axis=0)
        
        if np.any(np.isnan(feat_curr)) or np.any(np.isnan(feat_next)) or np.any(np.isnan(onset_centroid)):
            continue
            
        vec_next = feat_next - feat_curr
        vec_onset = onset_centroid - feat_curr
        vec_onset_next = onset_centroid - feat_next
        
        # Normalize vectors
        norm_next = np.linalg.norm(vec_next)
        norm_onset = np.linalg.norm(vec_onset)
        
        if norm_next == 0 or norm_onset == 0:
            continue
            
        vec_next_norm = vec_next / norm_next
        vec_onset_norm = vec_onset / norm_onset
        
        # Cosine angle (clamped to [-1, 1] to avoid float precision issues)
        cosine_angle = np.clip(np.dot(vec_next_norm, vec_onset_norm), -1.0, 1.0)
        theta = np.arccos(cosine_angle)
        
        d1 = np.linalg.norm(vec_onset)
        d2 = np.linalg.norm(vec_onset_next)
        
        sleep_velocity[jj] = d1 - d2
        thetas[jj] = theta
        projections[jj] = norm_next * np.cos(theta)
        
    return sleep_velocity, thetas, projections

def run_group_bootstrap(subject_metric_matrix, idx_bts):
    """
    Runs bootstrapping across subjects for a given metrics matrix (subjects x epochs).
    """
    num_sbjs, max_epochs = subject_metric_matrix.shape
    bootstrap_matrix = np.full((num_sbjs, max_epochs), np.nan)
    
    for jj in range(max_epochs):
        indices = idx_bts[jj]
        if len(indices) > 0:
            bootstrap_matrix[indices, jj] = subject_metric_matrix[indices, jj]
            
    # Calculate group average and standard error
    group_avg = np.nanmean(bootstrap_matrix, axis=0)
    num_valid = np.sum(~np.isnan(bootstrap_matrix), axis=0)
    group_ste = np.nanstd(bootstrap_matrix, axis=0) / np.sqrt(np.maximum(num_valid - 1, 1))
    
    return group_avg, group_ste, num_valid

def acf(X, alag=1):
    """
    Computes Autocorrelation at a given lag.
    """
    X = np.array(X)
    n = len(X)
    s = np.nanvar(X, ddof=1)
    mu = np.nanmean(X)
    if s == 0 or np.isnan(s):
        return np.nan
    Xt = X[:-alag]
    Xtk = X[alag:]
    res = 1 / (n - 1) / s * np.nansum((Xt - mu) * (Xtk - mu))
    return res

def dfa(y, smin=10, sstep=2):
    """
    Detrended Fluctuation Analysis (DFA).
    """
    y = np.array(y).flatten()
    if np.all(np.isnan(y)):
        return np.nan
        
    Yprof = np.nancumsum(y - np.nanmean(y)) # Subtract mean first as in typical DFA profile
    lenY = len(Yprof)
    smax = lenY // 2
    
    if lenY <= smin or smin >= smax:
        return np.nan
        
    svec = []
    F = []
    
    for s in range(smin, smax + 1, sstep):
        svec.append(s)
        t = np.arange(1, s + 1)
        Nseg = lenY // s
        Yerr_sum = 0
        
        for i in range(Nseg):
            Yseg = Yprof[i*s : (i+1)*s]
            # Linear fit (polyfit order 1)
            p = np.polyfit(t, Yseg, 1)
            Yfit = np.polyval(p, t)
            Yerr_sum += np.sum((Yseg - Yfit)**2)
            
        F.append(np.sqrt(Yerr_sum / (Nseg * s)))
        
    # Fit log10(svec) vs log10(F)
    try:
        pF = np.polyfit(np.log10(svec), np.log10(F), 1)
        res = pF[0]
    except Exception:
        res = np.nan
        
    return res

def ews_sleep_paper(ts, time_vec, win, ifdetrend=True):
    """
    Computes Early Warning Signals (AR1, StD, DFA) using rolling windows,
    and returns Kendall's tau correlation with time.
    """
    length = len(ts)
    if win >= length:
        raise ValueError("Too short input timeseries")
        
    acf1_ts = np.zeros(length - win + 1)
    std_ts = np.zeros(length - win + 1)
    dfa_ts = np.zeros(length - win + 1)
    
    for i in range(win, length + 1):
        ts_win = ts[i - win : i]
        
        if ifdetrend:
            # Linear detrending
            t = np.arange(len(ts_win))
            valid = ~np.isnan(ts_win)
            if np.sum(valid) > 1:
                slope, intercept, _, _, _ = stats.linregress(t[valid], ts_win[valid])
                ts_win_detrended = ts_win.copy()
                ts_win_detrended[valid] = ts_win[valid] - (slope * t[valid] + intercept)
                ts_win = ts_win_detrended
                
        if np.sum(~np.isnan(ts_win)) == 0:
            raise ValueError("Running window is entirely NaNs")
            
        idx = i - win
        acf1_ts[idx] = acf(ts_win, 1)
        std_ts[idx] = np.nanstd(ts_win, ddof=1)
        dfa_ts[idx] = dfa(ts_win)
        
    t_ts = time_vec[win - 1 :]
    
    # Calculate Kendall's tau
    tau_ar1, _ = stats.kendalltau(t_ts, acf1_ts, nan_policy='omit')
    tau_std, _ = stats.kendalltau(t_ts, std_ts, nan_policy='omit')
    tau_dfa, _ = stats.kendalltau(t_ts, dfa_ts, nan_policy='omit')
    
    ews_out = {
        'AR1': acf1_ts,
        'StD': std_ts,
        'DFA': dfa_ts
    }
    
    tau = {
        'AR1': tau_ar1,
        'StD': tau_std,
        'DFA': tau_dfa
    }
    
    return ews_out, tau

