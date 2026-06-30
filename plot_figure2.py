import os
import numpy as np
import scipy.io as sio
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

def shaded_error_bar(ax, x, y, err, color='black', alpha=0.2, linewidth=2, label=None):
    ax.plot(x, y, color=color, linewidth=linewidth, label=label)
    ax.fill_between(x, y - err, y + err, color=color, alpha=alpha)

def plot_figure_2():
    data_dir = 'All Final Data for plots/Figure2'
    output_dir = 'Final Plot codes/Figure2'
    os.makedirs(output_dir, exist_ok=True)
    
    # ------------------ Figure 2a ------------------
    try:
        f2a = sio.loadmat(os.path.join(data_dir, 'Figure2A.mat'))
        tvec_now = f2a['tvec_now'].flatten()
        xx_smooth = f2a['xx_smooth'].flatten()
        distall_ste = f2a['distall_ste'].flatten()
        time_start_sampenough = int(f2a['time_start_sampenough'].flatten()[0]) - 1 # 1-based to 0-based
        idxstartfit = int(f2a['idxstartfit'].flatten()[0]) - 1
        dd = f2a['dd']
        pvec_dist = f2a['pvec_dist'].flatten()
        basetest_epc = int(f2a['basetest_epc'].flatten()[0]) - 1
        t_critc = f2a['t_critc'].flatten()[0]
        params_optim = f2a['params_optim'].flatten()
        c_3sol = f2a['c_3sol'].flatten()
        
        # In MATLAB: onsetmark = epcs_tofit;
        # Since epcs_tofit might be in f2a, let's extract it
        onsetmark = int(f2a['epcs_tofit'].flatten()[0]) - 1
        
        fig, ax = plt.subplots(figsize=(6, 5))
        xxste = distall_ste[time_start_sampenough:]
        shaded_error_bar(ax, tvec_now, xx_smooth, xxste[idxstartfit:], color='k')
        ax.plot(tvec_now, dd[:, 0], 'b-', linewidth=4, label='Bifurcation Fit')
        
        # Red dashed onset line
        ax.axvline(tvec_now[onsetmark], color='r', linestyle='--', linewidth=2)
        ax.set_ylim([0, 5])
        ax.set_xlim([-60, 10])
        ax.set_ylabel('s', fontsize=14)
        ax.set_xlabel('Time (min)', fontsize=14)
        
        # P-value bars
        p_th = 0.025 / len(pvec_dist)
        yl = ax.get_ylim()
        for iii in range(len(pvec_dist)):
            tidx = iii + basetest_epc - idxstartfit
            if tidx < len(tvec_now) - 1:
                if pvec_dist[iii] < p_th:
                    ax.plot([tvec_now[tidx], tvec_now[tidx+1]], [0.95*yl[1], 0.95*yl[1]], 'k-', linewidth=1)
                    
        # Tipping point scatter
        tc_idx = np.argmin(np.abs(tvec_now - t_critc))
        ax.scatter(t_critc, dd[tc_idx, 0], s=80, color='r', zorder=5)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure2a_Python.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting Fig 2a: {e}")

    # ------------------ Figure 2b ------------------
    try:
        f2b = sio.loadmat(os.path.join(data_dir, 'Figure2B.mat'))
        tvec = f2b['tvec'].flatten()
        distall_avg_smooth = f2b['distall_avg_smooth'].flatten()
        distall_ste = f2b['distall_ste'].flatten()
        time_start_sampenough = int(f2b['time_start_sampenough'].flatten()[0]) - 1
        max_ck_real = int(f2b['max_ck_real'].flatten()[0])
        pvec_dist = f2b['pvec_dist'].flatten()
        basetest_epc = int(f2b['basetest_epc'].flatten()[0]) - 1
        epc_len = float(f2b['epc_len'].flatten()[0])
        jmp_len = float(f2b['jmp_len'].flatten()[0])
        
        timeplot = 60.0 * 60.0
        epcs_toplot = int((timeplot - epc_len) // jmp_len) + 1
        idxstartplot = max_ck_real - epcs_toplot
        
        tvec_new = tvec[idxstartplot:]
        distvec_smooth = distall_avg_smooth[time_start_sampenough:]
        distvec_smooth_new = distvec_smooth[idxstartplot:]
        distste = distall_ste[time_start_sampenough:]
        distste_new = distste[idxstartplot:]
        
        onset_new = max_ck_real - idxstartplot
        
        fig, ax = plt.subplots(figsize=(6, 5))
        shaded_error_bar(ax, tvec_new, distvec_smooth_new, distste_new, color='k')
        ax.axvline(tvec_new[onset_new], color='r', linestyle='--', linewidth=2)
        ax.set_ylim([4.5, 6.0])
        ax.set_xlim([-60, 10])
        ax.set_ylabel('v(t)', fontsize=14)
        ax.set_xlabel('Time (min)', fontsize=14)
        
        p_th = 0.025 / len(pvec_dist)
        yl = ax.get_ylim()
        for iii in range(len(pvec_dist)):
            tidx = iii + basetest_epc - idxstartplot
            if tidx < len(tvec_new) - 1:
                if pvec_dist[iii] < p_th:
                    ax.plot([tvec_new[tidx], tvec_new[tidx+1]], [0.98*yl[1], 0.98*yl[1]], 'k-', linewidth=1)
                    
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure2b_Python.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting Fig 2b: {e}")

    # ------------------ Figure 2c ------------------
    try:
        f2c = sio.loadmat(os.path.join(data_dir, 'Figure2C.mat'))
        tvec = f2c['tvec'].flatten()
        dotp_avg = f2c['dotp_avg'].flatten()
        dotp_ste = f2c['dotp_ste'].flatten()
        time_start_sampenough = int(f2c['time_start_sampenough'].flatten()[0]) - 1
        max_ck_real = int(f2c['max_ck_real'].flatten()[0])
        pvec_dist = f2c['pvec_dist'].flatten()
        basetest_epc = int(f2c['basetest_epc'].flatten()[0]) - 1
        
        # Smooth data with rolling mean
        dotp_avg_series = pd.Series(dotp_avg)
        distall_avg_smooth = dotp_avg_series.rolling(window=20, min_periods=1, center=True).mean().values
        
        fig, ax = plt.subplots(figsize=(6, 5))
        shaded_error_bar(ax, tvec, distall_avg_smooth[time_start_sampenough:], dotp_ste[time_start_sampenough:], color='k')
        ax.axvline(tvec[max_ck_real], color='r', linestyle='--', linewidth=2)
        ax.set_ylim([-0.04, 0.1])
        ax.set_xlim([-60, 10])
        ax.set_ylabel('Alignment', fontsize=14)
        ax.set_xlabel('Time (min)', fontsize=14)
        
        p_th = 0.025 / (len(pvec_dist) - 20)
        yl = ax.get_ylim()
        for iii in range(len(pvec_dist)):
            tidx = iii + basetest_epc
            if tidx < len(tvec) - 1:
                if pvec_dist[iii] < p_th:
                    ax.plot([tvec[tidx], tvec[tidx+1]], [0.95*yl[1], 0.95*yl[1]], 'k-', linewidth=2)
                    
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure2c_Python.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting Fig 2c: {e}")

    # ------------------ Figure 2d ------------------
    try:
        f2d = sio.loadmat(os.path.join(data_dir, 'Figure2D.mat'))
        distall_avg = f2d['distall_avg'].flatten()
        
        # Using dotp_avg from Figure2C
        f2c_for_d = sio.loadmat(os.path.join(data_dir, 'Figure2C.mat'))
        dotp_avg = f2c_for_d['dotp_avg'].flatten()
        
        dotp_all = pd.Series(dotp_avg).rolling(window=10, min_periods=1, center=True).mean().values
        distsmooth = pd.Series(distall_avg).rolling(window=5, min_periods=1, center=True).mean().values
        distsmooth = distsmooth - np.nanmin(distsmooth)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(distsmooth[:-20], (dotp_all**2) / 2, s=20, color='k')
        ax.set_xlabel('s', fontsize=14)
        ax.set_ylabel('E (Potential)', fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure2d_Python.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting Fig 2d: {e}")

    # ------------------ Figure 2e (Theoretical Bifurcation) ------------------
    try:
        f2a = sio.loadmat(os.path.join(data_dir, 'Figure2A.mat'))
        dd = f2a['dd']
        params_optim = f2a['params_optim'].flatten()
        c_3sol = f2a['c_3sol'].flatten()
        t_critc = f2a['t_critc'].flatten()[0]
        tvec_now = f2a['tvec_now'].flatten()
        
        c = dd[:, 1]
        r, K, h, m = params_optim
        
        # Find critical indices
        c_3sol_sorted = np.sort(c_3sol)
        idx1 = np.where(c == c_3sol_sorted[0])[0][0]
        idx2 = np.where(c == c_3sol_sorted[-1])[0][0]
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Root finding function
        def solve_roots(c_val):
            # -(r/K)x^3 + rx^2 - (c + r*h^2/K)x + r*h^2 = 0
            roots = np.roots([-(r/K), r, -(c_val + r * h**2 / K), r * h**2])
            real_roots = np.real(roots[np.isreal(roots) & (np.abs(np.imag(roots)) < 1e-5)])
            real_roots = real_roots[real_roots > 0]
            return np.sort(real_roots)
            
        # Range 1 (before 3 solutions)
        cvec1, xvec1 = [], []
        for i in range(idx1):
            roots = solve_roots(c[i])
            if len(roots) >= 1:
                cvec1.append(c[i])
                xvec1.append(roots[0])
        ax.plot(cvec1, xvec1, 'k-', linewidth=2)
        
        # Range 2 (3 solutions)
        cvec2, xvec2 = [], []
        for i in range(idx1, idx2 + 1):
            roots = solve_roots(c[i])
            if len(roots) == 3:
                cvec2.append(c[i])
                xvec2.append(roots)
        if cvec2:
            xvec2 = np.array(xvec2)
            ax.plot(cvec2, xvec2[:, 0], 'k-', linewidth=2)
            ax.plot(cvec2, xvec2[:, 1], 'k--', linewidth=2) # unstable branch
            ax.plot(cvec2, xvec2[:, 2], 'k-', linewidth=2)
            
        # Range 3 (after 3 solutions)
        cvec3, xvec3 = [], []
        for i in range(idx2 + 1, len(c)):
            roots = solve_roots(c[i])
            if len(roots) >= 1:
                cvec3.append(c[i])
                xvec3.append(roots[-1])
        ax.plot(cvec3, xvec3, 'k-', linewidth=2)
        
        c_critc = c[np.argmin(np.abs(tvec_now - t_critc))]
        ax.axvline(c_critc, color='r', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Control parameter', fontsize=14)
        ax.set_ylabel('System state', fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure2e_Python.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting Fig 2e: {e}")

    # ------------------ Figure 2f ------------------
    try:
        f2f = sio.loadmat(os.path.join(data_dir, 'Figure2F.mat'))
        avg_score = f2f['avg_score']
        ews_all = f2f['ews_all'].flatten()
        pvec_ews = f2f['pvec_ews'].flatten()
        pvec_slp = f2f['pvec_slp'].flatten()
        tews = f2f['tews'].flatten()
        base_epcs = int(f2f['base_epcs'].flatten()[0]) - 1
        kk = int(f2f['kk'].flatten()[0])
        
        tews_plt = tews / 60.0
        idxplt = np.where(tews_plt < 10)[0]
        tews_plt = tews_plt[idxplt]
        
        onsetmark = np.argmin(np.abs(tews_plt - 0.0))
        lengap = len(tews) - len(tews_plt)
        
        # Calculate stats for average sleep stages
        avg_score1 = np.nanmean(avg_score, axis=0)
        std_score1 = np.nanstd(avg_score, axis=0) / np.sqrt(kk - 1)
        
        avg_score1 = avg_score1[-len(tews):][idxplt]
        std_score1 = std_score1[-len(tews):][idxplt]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8))
        
        # Subplot 1: Sleep stages
        shaded_error_bar(ax1, tews_plt, avg_score1, std_score1, color='k')
        ax1.set_title('Average sleep stages')
        ax1.set_ylabel('Sleep stages')
        ax1.set_ylim([-0.5, 3])
        ax1.set_xlim([-25, 10])
        ax1.axvline(tews_plt[onsetmark], color='r', linestyle='--', linewidth=2)
        
        p_th = 0.025 / len(pvec_slp)
        yl1 = ax1.get_ylim()
        for iii in range(len(pvec_slp)):
            tidx = iii + base_epcs - lengap
            if tidx < len(tews_plt) - 1:
                if pvec_slp[iii] < p_th:
                    ax1.plot([tews_plt[tidx], tews_plt[tidx+1]], [2.3, 2.3], 'k-', linewidth=1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Subplot 2: Autocorrelation
        allewsts1 = ews_all[0]
        ewsavg1 = np.nanmedian(allewsts1, axis=0)[idxplt]
        stdews1 = (np.nanstd(allewsts1, axis=0) / np.sqrt(kk - 1))[idxplt]
        
        shaded_error_bar(ax2, tews_plt, ewsavg1, stdews1, color='k')
        ax2.set_ylabel('Autocorrelation')
        ax2.set_ylim([0.4, 0.65])
        ax2.set_xlim([-25, 10])
        ax2.axvline(tews_plt[onsetmark], color='r', linestyle='--', linewidth=2)
        
        pvecnow1 = pvec_ews[0].flatten()
        yl2 = ax2.get_ylim()
        for iii in range(len(pvecnow1)):
            tidx = iii + base_epcs - lengap
            if tidx < len(tews_plt) - 1:
                if pvecnow1[iii] < p_th:
                    ax2.plot([tews_plt[tidx], tews_plt[tidx+1]], [0.98*yl2[1], 0.98*yl2[1]], 'k-', linewidth=1)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Subplot 3: Standard deviation
        allewsts2 = ews_all[1]
        ewsavg2 = np.nanmedian(allewsts2, axis=0)[idxplt]
        stdews2 = (np.nanstd(allewsts2, axis=0) / np.sqrt(kk - 1))[idxplt]
        
        shaded_error_bar(ax3, tews_plt, ewsavg2, stdews2, color='k')
        ax3.set_ylabel('Standard deviation')
        ax3.set_xlabel('Time (min)')
        ax3.set_ylim([1.2, 1.8])
        ax3.set_xlim([-25, 10])
        ax3.axvline(tews_plt[onsetmark], color='r', linestyle='--', linewidth=2)
        
        pvecnow2 = pvec_ews[1].flatten()
        yl3 = ax3.get_ylim()
        for iii in range(len(pvecnow2)):
            tidx = iii + base_epcs - lengap
            if tidx < len(tews_plt) - 1:
                if pvecnow2[iii] < p_th:
                    ax3.plot([tews_plt[tidx], tews_plt[tidx+1]], [0.99*yl3[1], 0.99*yl3[1]], 'k-', linewidth=1)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure2f_Python.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting Fig 2f: {e}")

    # ------------------ Figure 2g ------------------
    try:
        f2g = sio.loadmat(os.path.join(data_dir, 'Figure2G.mat'))
        tvec = f2g['tvec'].flatten()
        hyp_expand = f2g['hyp_expand'].flatten()
        s_dynam = f2g['s_dynam'].flatten()
        sfit = f2g['sfit'].flatten()
        t_crtc = f2g['t_crtc'].flatten()[0]
        stageNames = [str(el[0]) for el in f2g['stageNames'].flatten()]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={'height_ratios': [2, 1]})
        
        # Subplot 1 (Top): State dynamics
        ax1.plot(tvec, s_dynam, 'k-', alpha=0.5, label='Data')
        ax1.plot(tvec, sfit, 'b-', linewidth=3, label='Bifurcation Fit')
        ax1.axvline(0, color='r', linestyle='--', linewidth=2)
        ax1.set_ylabel('s')
        ax1.set_xlim([-30, 10])
        
        # Tipping point scatter
        tc_idx = np.argmin(np.abs(tvec - t_crtc))
        ax1.scatter(t_crtc, sfit[tc_idx], s=80, color='r', zorder=5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Subplot 2 (Bottom): Hypnogram
        ax2.plot(tvec, hyp_expand, 'k-', linewidth=2)
        ax2.set_ylim([0, 3])
        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_yticklabels(stageNames)
        ax2.axvline(0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Time (min)')
        ax2.set_xlim([-30, 10])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure2g_Python.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting Fig 2g: {e}")

    # ------------------ Figure 2h ------------------
    try:
        f2h = sio.loadmat(os.path.join(data_dir, 'Figure2H.mat'))
        r2all_new = f2h['r2all_new'].flatten()
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(r2all_new, bins=np.arange(0, 1.05, 0.05), color='gray', edgecolor='black', alpha=0.7)
        
        # Fit Kernel density estimate (KDE)
        kde = stats.gaussian_kde(r2all_new)
        pdfbin = np.linspace(0, 1, 100)
        ypdf = kde(pdfbin)
        # Scale to match histogram counts (bin width = 0.05)
        ax.plot(pdfbin, ypdf * (0.05 * len(r2all_new)), 'r-', linewidth=2)
        
        r2med = np.nanmedian(r2all_new)
        ax.axvline(r2med, color='r', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Fitting accuracy (R2)', fontsize=14)
        ax.set_ylabel('No. of participants', fontsize=14)
        ax.set_xlim([0, 1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure2h_Python.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting Fig 2h: {e}")

    # ------------------ Figure 2i ------------------
    try:
        f2i = sio.loadmat(os.path.join(data_dir, 'Figure2I.mat'))
        K_all_clean = f2i['K_all_clean'].flatten()
        t_crtic_clean = f2i['t_crtic_clean'].flatten()
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(K_all_clean, t_crtic_clean, color='blue', alpha=0.7)
        
        p = np.polyfit(K_all_clean, t_crtic_clean, 1)
        xl = np.array([0, 15])
        ax.plot(xl, xl * p[0] + p[1], 'r-', linewidth=2)
        
        ax.set_xlabel('Maximum Distance', fontsize=14)
        ax.set_ylabel('Tipping point (Relative to Onset)', fontsize=14)
        ax.set_ylim([-20, 10])
        ax.set_xlim([0, 15])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure2i_Python.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting Fig 2i: {e}")

    print("All Figure 2 plots created successfully!")

if __name__ == "__main__":
    plot_figure_2()
