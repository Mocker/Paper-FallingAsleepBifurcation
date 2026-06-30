import os
import numpy as np
import scipy.io as sio
import mat73 as mp
import matplotlib.pyplot as plt
import skfda
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.misc.regularization import L2Regularization
import pandas as pd

def plot_figure_3():
    data_dir_f3 = 'All Final Data for plots/Figure3'
    data_dir_f2 = 'All Final Data for plots/Figure2'
    output_dir = 'Final Plot codes/Figure3'
    os.makedirs(output_dir, exist_ok=True)
    
    # ------------------ Figure 3a & 3d (FPCA) ------------------
    try:
        data_dict = mp.loadmat(os.path.join(data_dir_f3, 'FPCADynamics.mat'))
        ftall_mat_allnorm_strm = data_dict["ftall_mat_allnorm_strm"]
        max_ck_real = int(data_dict["max_ck_real"])
        
        # Grid parameters (replicates FPCA_Features.py)
        preslp = int(np.floor((60*60 - 6) / 3))
        slp_onset_id = int(max_ck_real) - 1
        ftallmat_allnorm_nonan_preslp = ftall_mat_allnorm_strm[:, slp_onset_id-preslp:slp_onset_id]
        
        grid_p = np.linspace(-(((preslp-1)*3 + 6)/60), 0, preslp)
        
        fd = skfda.FDataGrid(
            data_matrix=ftallmat_allnorm_nonan_preslp,
            grid_points=grid_p,
        )
        
        basis = skfda.representation.basis.BSplineBasis(n_basis=10)
        fd_basis = fd.to_basis(basis)
        
        reg = L2Regularization()
        n_cmp = 2
        fpca_reg = FPCA(n_components=n_cmp, regularization=reg)
        pc_score_reg = fpca_reg.fit_transform(fd_basis)
        
        # Plotting FPCA components
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        
        # Component 1 (FPC1)
        mean_curve = fd_basis.mean()(grid_p).flatten()
        comp1 = fpca_reg.components_(grid_p)[0].flatten()
        # Replicate FPCAPlot styles (fd.mean() +/- factor * component)
        ax1.plot(grid_p, mean_curve, 'k-', linewidth=2)
        ax1.plot(grid_p, mean_curve + 5 * comp1, 'g+', markevery=5)
        ax1.plot(grid_p, mean_curve - 5 * comp1, 'r-', linestyle=':')
        ax1.set_title('FPC1')
        ax1.set_xlim([-60, 0])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Component 2 (FPC2)
        comp2 = fpca_reg.components_(grid_p)[1].flatten()
        ax2.plot(grid_p, mean_curve, 'k-', linewidth=2)
        ax2.plot(grid_p, mean_curve + 5 * comp2, 'g+', markevery=5)
        ax2.plot(grid_p, mean_curve - 5 * comp2, 'r-', linestyle=':')
        ax2.set_title('FPC2')
        ax2.set_xlim([-60, 0])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure3ad_Python.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting Fig 3ad: {e}")

    # ------------------ Figure 3b & 3e (Individual Features) ------------------
    try:
        tvec = - (max_ck_real - 1) / 20.0 + np.arange(ftall_mat_allnorm_strm.shape[1]) * 0.05
        
        # Features to plot: 15, 41, 6, 17 (0-based: 14, 40, 5, 16)
        features_to_plot = [14, 40, 5, 16]
        titles = ['Peak beta band frequency (15)', 'Total EEG power (41)', 
                  'Delta-to-Alpha ratio (6)', 'Theta temporal coherence (17)']
        ylims = [(-0.4, 1.6), (-2.0, 0.5), (-1.0, 0.4), (-0.2, 1.0)]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, ft_idx in enumerate(features_to_plot):
            ax = axes[idx]
            ax.plot(tvec, ftall_mat_allnorm_strm[ft_idx, :], 'k-', linewidth=2)
            ax.axhline(0, color='k', linestyle='--', linewidth=1.5)
            ax.axvline(0, color='r', linestyle='--', linewidth=1.5)
            ax.set_ylim(ylims[idx])
            ax.set_xlim([-60, 10])
            ax.set_title(titles[idx])
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('z-score')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure3be_Python.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting Fig 3be: {e}")

    # ------------------ Figure 3c & 3f (Sleep Distance vs Features) ------------------
    try:
        f2a = sio.loadmat(os.path.join(data_dir_f2, 'Figure2A.mat'))
        xx = f2a['xx_smooth'].flatten() # sleep distance
        idxstartfit = int(f2a['idxstartfit'].flatten()[0]) - 1
        
        # Features to plot: 6, 17 (0-based: 5, 16)
        features_to_plot = [5, 16]
        titles = ['Delta-to-Alpha ratio (6)', 'Theta temporal coherence (17)']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        tvec_full = - (max_ck_real - 1) / 20.0 + np.arange(ftall_mat_allnorm_strm.shape[1]) * 0.05
        tvec_fit = tvec_full[idxstartfit:]
        
        for idx, ft_idx in enumerate(features_to_plot):
            ax = axes[idx]
            
            # Smooth features and xx
            ft_series = pd.Series(ftall_mat_allnorm_strm[ft_idx, idxstartfit:])
            ft_smooth = ft_series.rolling(window=10, min_periods=1, center=True).median().values
            
            xx_series = pd.Series(xx)
            xx_smooth_metric = xx_series.rolling(window=10, min_periods=1, center=True).median().values
            
            # Highlight onset window (zero_idx to zero_idx+20)
            zero_idx = np.argmin(np.abs(tvec_fit - 0.0))
            
            # Color points by tvec_fit
            sc = ax.scatter(ft_smooth, xx_smooth_metric, c=tvec_fit, cmap='viridis', s=20)
            # Highlight onset in red
            ax.scatter(ft_smooth[zero_idx : zero_idx+21], xx_smooth_metric[zero_idx : zero_idx+21], color='red', s=20)
            
            ax.set_ylim([0, 5])
            ax.set_xlabel('z-score')
            ax.set_ylabel('Sleep Distance')
            ax.set_title(titles[idx])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.colorbar(sc, ax=ax, label='Time (min)')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure3cf_Python.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting Fig 3cf: {e}")

    print("All Figure 3 plots created successfully!")

if __name__ == "__main__":
    plot_figure_3()
