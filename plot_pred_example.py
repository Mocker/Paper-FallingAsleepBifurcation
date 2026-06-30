import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_predictions():
    # 1. Load Data
    try:
        pred_data = sio.loadmat('Core Algorithms and Examples/Example Prediction Tipping point/PredictionExample.mat')
        fig4_data = sio.loadmat('All Final Data for plots/Figure4/Figure4di.mat')
        
        # Extract variables from PredictionExample.mat
        estimatedSVariables = pred_data['estimatedSVariables'].flatten()
        criticalSValue = pred_data['criticalSValue'].flatten()[0]
        globalMin = pred_data['globalMin'].flatten()[0]
        Tcross_raw = pred_data['Tcross_raw'].flatten()
        
        # Extract variables from Figure4di.mat
        tvec = fig4_data['tvec'].flatten()
        xx_smooth = fig4_data['xx_smooth'].flatten()
        # dd, params_optim are also in fig4_data but unused in this plotting script
    except FileNotFoundError:
        print("Data files not found.")
        return

    # Prepare plotting
    num_subplots = len(estimatedSVariables)
    num_rows = int(np.ceil(num_subplots / 2))
    num_cols = 2
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 3))
    axes = axes.flatten()
    
    smoothingWindow = 20
    
    for n_n in range(num_subplots):
        ax = axes[n_n]
        distPerNight = estimatedSVariables[n_n].flatten()
        
        # Smoothing
        s_series = pd.Series(distPerNight)
        uncut_xx_smoothed = s_series.rolling(window=smoothingWindow+1, min_periods=1, center=False).median().values
        uncut_xx_smoothed = uncut_xx_smoothed - globalMin
        
        # Trim last 9 points (equivalent to MATLAB 1:end-9)
        if len(uncut_xx_smoothed) > 9:
            uncut_xx_smoothed = uncut_xx_smoothed[:-9]
            
        # Time vector uncut
        uncut_tvec = np.arange(-30*60 + 6, 10*60 + 3, 3) / 60.0
        
        # Align lengths
        if len(uncut_tvec) > len(uncut_xx_smoothed):
            tvec_night = uncut_tvec[-len(uncut_xx_smoothed):]
            xx_smoothed = uncut_xx_smoothed
        else:
            tvec_night = uncut_tvec
            xx_smoothed = uncut_xx_smoothed[-len(uncut_tvec):]
            
        # Find crossing points (downward crossing criticalSValue)
        # We need xx_smoothed(i) > criticalSValue and xx_smoothed(i+1) <= criticalSValue
        downward_crossings = (xx_smoothed[:-1] > criticalSValue) & (xx_smoothed[1:] <= criticalSValue)
        cross_indices = np.where(downward_crossings)[0]
        
        real_crossings_t = []
        
        for idx in cross_indices:
            t1, t2 = tvec_night[idx], tvec_night[idx+1]
            x1, x2 = xx_smoothed[idx], xx_smoothed[idx+1]
            
            # Interpolate
            timepoint_int = t1 + (criticalSValue - x1) * (t2 - t1) / (x2 - x1)
            
            if timepoint_int > 0:
                continue
                
            # Find upward crossing
            future_t = tvec_night[idx+1:]
            future_x = xx_smoothed[idx+1:]
            upward_idx_arr = np.where(future_x >= criticalSValue)[0]
            
            if len(upward_idx_arr) > 0:
                upward_idx = upward_idx_arr[0]
                upward_crossing_time = future_t[upward_idx]
                time_spent_below = upward_crossing_time - timepoint_int
                is_real_crossing = (time_spent_below > 1.0) or (timepoint_int > -1.0)
            else:
                is_real_crossing = True # Stays below indefinitely
                
            if is_real_crossing:
                real_crossings_t.append(timepoint_int)
                
        # Plotting
        ax.plot(tvec_night, xx_smoothed, 'k-', linewidth=1.5)
        ax.axhline(criticalSValue, color='g', linestyle='--', linewidth=1.5)
        
        # Real crossings
        for rc_t in real_crossings_t:
            ax.plot(rc_t, criticalSValue, 'ro', markersize=6)
            
        # Post-hoc model tip
        if n_n < len(Tcross_raw):
            ax.axvline(Tcross_raw[n_n], color='b', linestyle='--', linewidth=1.5)
            
        ax.axvline(0, color='r', linestyle='--', linewidth=1.5) # Sleep onset
        
        ax.set_xlim([-30, 10])
        ax.set_ylim([0, 12])
        ax.set_ylabel('System State')
        ax.set_xlabel('Time (min)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.tight_layout()
    plt.savefig('Core Algorithms and Examples/Example Prediction Tipping point/Prediction_Python_Plot.png')
    plt.close(fig)
    print("Prediction plotting complete! Image saved to Prediction_Python_Plot.png")

if __name__ == "__main__":
    evaluate_predictions()
