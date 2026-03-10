#!/usr/bin/env python3
"""
Project 3 Step 3 (v2): Lead-Lag Analysis with Detrending

Key improvement: Remove long-term solar cycle trend before computing
cross-correlation. This isolates the short-term (days-to-weeks) 
lead/lag signal from the dominant ~11-year trend.

Usage:
  python3 leadlag_analysis_v2.py --freq-csv 3day_freq_shifts.csv --output-dir results_v2
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import signal, stats, interpolate
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import io
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================
# Data Loading
# =============================================================

def load_freq_shifts(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['start_date', 'mid_date'])
    df = df.sort_values('mdi_day').reset_index(drop=True)
    print(f"Loaded {len(df)} frequency shift measurements")
    print(f"  Date range: {df['mid_date'].min().date()} to {df['mid_date'].max().date()}")
    print(f"  Shift range: {df['freq_shift_uHz'].min():.4f} to {df['freq_shift_uHz'].max():.4f} uHz")
    return df


def download_daily_sunspot_numbers(cache_path='SN_d_tot_V2.0.csv'):
    if os.path.exists(cache_path):
        print(f"Using cached sunspot data: {cache_path}")
    else:
        url = "https://www.sidc.be/SILSO/INFO/sndtotcsv.php"
        print(f"Downloading daily sunspot numbers from SILSO...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(cache_path, 'wb') as f:
            f.write(resp.content)
        print(f"  Saved to {cache_path}")
    
    df = pd.read_csv(cache_path, sep=';', header=None,
                     names=['year', 'month', 'day', 'decimal_year',
                            'ssn', 'ssn_err', 'n_obs', 'flag'],
                     skipinitialspace=True)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df.loc[df['ssn'] < 0, 'ssn'] = np.nan
    print(f"Loaded {len(df)} daily sunspot records")
    return df[['date', 'ssn', 'ssn_err']].copy()


def download_f107_flux(cache_path='f107_daily.csv'):
    """Download F10.7 from LISIRD with correct date parsing."""
    if os.path.exists(cache_path):
        # Check if it's the broken version
        test = pd.read_csv(cache_path, nrows=2)
        if 'date' in test.columns:
            first_date = str(test['date'].iloc[0])
            if '1970' in first_date:
                print("F10.7 cache has bad dates, re-downloading...")
                os.remove(cache_path)
            else:
                print(f"Using cached F10.7 data: {cache_path}")
                df = pd.read_csv(cache_path, parse_dates=['date'])
                print(f"Loaded {len(df)} F10.7 records")
                return df
    
    url = ("https://lasp.colorado.edu/lisird/latis/dap/penticton_radio_flux.csv"
           "?time>=2010-01-01&time<=2025-12-31")
    print(f"Downloading F10.7 flux from LISIRD...")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.strip() for c in df.columns]
        
        # LISIRD returns time as Julian Date (days since some epoch)
        # or as a decimal timestamp. Try to detect format.
        time_col = df.columns[0]
        time_vals = pd.to_numeric(df[time_col], errors='coerce')
        
        if time_vals.median() > 2400000:
            # Julian Date
            jd_epoch = pd.Timestamp('1858-11-17')  # Modified Julian Date epoch
            df['date'] = pd.to_datetime(time_vals - 2400000.5, unit='D', origin=jd_epoch)
        elif time_vals.median() > 1e9:
            # Unix-like milliseconds
            df['date'] = pd.to_datetime(time_vals, unit='ms')
        else:
            # Try direct parsing
            df['date'] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Find flux column
        flux_cols = [c for c in df.columns if 'flux' in c.lower() or 'f107' in c.lower()
                     or c == df.columns[1]]
        if flux_cols:
            df['f107'] = pd.to_numeric(df[flux_cols[0]], errors='coerce')
        
        df = df[['date', 'f107']].dropna()
        df = df[(df['date'] >= '2009-01-01') & (df['date'] <= '2026-01-01')]
        df = df.sort_values('date').reset_index(drop=True)
        
        if len(df) > 100:
            df.to_csv(cache_path, index=False)
            print(f"  Saved {len(df)} records to {cache_path}")
            return df
        else:
            print(f"  Only got {len(df)} records, F10.7 may not work")
            return None
    except Exception as e:
        print(f"  F10.7 download failed: {e}")
        return None


# =============================================================
# Alignment and Preprocessing
# =============================================================

def align_to_3day(freq_df, activity_df, activity_col='ssn'):
    aligned = freq_df[['daterange', 'start_date', 'mid_date', 'mdi_day',
                        'freq_shift_uHz', 'freq_shift_err_uHz']].copy()
    
    activity_values = []
    for _, row in freq_df.iterrows():
        start = row['start_date']
        end = start + timedelta(days=2)
        mask = (activity_df['date'] >= start) & (activity_df['date'] <= end)
        vals = activity_df.loc[mask, activity_col]
        if len(vals) > 0 and vals.notna().sum() > 0:
            activity_values.append(vals.mean())
        else:
            activity_values.append(np.nan)
    
    aligned[activity_col] = activity_values
    n_before = len(aligned)
    aligned = aligned.dropna(subset=[activity_col]).reset_index(drop=True)
    print(f"Aligned: {len(aligned)}/{n_before} points with {activity_col}")
    return aligned


def detrend_series(x, method='lowpass', cutoff_days=81):
    """Remove long-term trend from a time series.
    
    Methods:
    - 'lowpass': subtract a running mean (cutoff in 3-day units)
    - 'highpass': Butterworth highpass filter
    
    Returns (residual, trend).
    """
    if method == 'lowpass':
        window = max(3, cutoff_days // 3)  # convert days to 3-day units
        trend = pd.Series(x).rolling(window, center=True, min_periods=window//2).mean().values
        residual = x - trend
    elif method == 'highpass':
        # Butterworth highpass
        fs = 1.0 / 3.0  # sampling freq: 1 per 3 days
        fc = 1.0 / cutoff_days  # cutoff freq
        b, a = signal.butter(3, fc / (fs / 2), btype='high')
        residual = signal.filtfilt(b, a, x)
        trend = x - residual
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return residual, trend


def smooth_series(x, window=7):
    return pd.Series(x).rolling(window, center=True, min_periods=1).mean().values


# =============================================================
# Cross-Correlation
# =============================================================

def compute_crosscorr(x, y, max_lag=100):
    """Normalized cross-correlation.
    Positive lag = x leads y.
    """
    x = (x - np.nanmean(x)) / np.nanstd(x)
    y = (y - np.nanmean(y)) / np.nanstd(y)
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.full(len(lags), np.nan)
    
    for i, lag in enumerate(lags):
        if lag == 0:
            cc[i] = np.corrcoef(x, y)[0, 1]
        elif lag > 0 and lag < n:
            cc[i] = np.corrcoef(x[:n-lag], y[lag:])[0, 1]
        elif lag < 0 and -lag < n:
            cc[i] = np.corrcoef(x[-lag:], y[:n+lag])[0, 1]
    
    return lags, cc


def block_bootstrap_peak_lag(x, y, max_lag=50, n_bootstrap=2000, 
                              block_size=10, ci=0.95):
    """Block bootstrap for peak lag CI.
    
    Block bootstrap preserves temporal autocorrelation better than iid.
    """
    n = len(x)
    n_blocks = n // block_size
    peak_lags = []
    
    for _ in range(n_bootstrap):
        # Draw random blocks
        block_starts = np.random.randint(0, n - block_size, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) for s in block_starts])
        indices = indices[:n]
        
        xb = x[indices]
        yb = y[indices]
        
        lags, cc = compute_crosscorr(xb, yb, max_lag)
        valid = ~np.isnan(cc)
        if valid.sum() > 0:
            peak_lags.append(lags[valid][np.argmax(cc[valid])])
    
    peak_lags = np.array(peak_lags)
    
    # Full-data result
    lags, cc = compute_crosscorr(x, y, max_lag)
    valid = ~np.isnan(cc)
    peak_idx = np.argmax(cc[valid])
    peak_lag = lags[valid][peak_idx]
    peak_corr = cc[valid][peak_idx]
    
    alpha = 1 - ci
    ci_low = np.percentile(peak_lags, 100 * alpha / 2)
    ci_high = np.percentile(peak_lags, 100 * (1 - alpha / 2))
    
    return peak_lag, peak_corr, ci_low, ci_high, peak_lags


# =============================================================
# Visualization
# =============================================================

def plot_detrending(dates, raw, trend, residual, ylabel, title, output):
    """Show the detrending process."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={'hspace': 0.1})
    
    axes[0].plot(dates, raw, color='C0', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(dates, raw, color='C0', linewidth=0.5, alpha=0.3)
    axes[1].plot(dates, trend, color='C1', linewidth=2, label='81-day trend')
    axes[1].set_ylabel(ylabel)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(dates, residual, color='C2', linewidth=0.5)
    axes[2].axhline(0, color='gray', linewidth=0.5)
    axes[2].set_ylabel(f'{ylabel}\n(detrended)')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_full_timeseries(aligned, activity_col, smooth_window=9, output='fig_ts.png'):
    """Time series plot."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10),
                                         sharex=True, gridspec_kw={'hspace': 0.08})
    dates = aligned['mid_date']
    dnu = aligned['freq_shift_uHz'].values
    act = aligned[activity_col].values
    
    dnu_s = smooth_series(dnu, smooth_window)
    act_s = smooth_series(act, smooth_window)
    
    ax1.scatter(dates, dnu, s=1, alpha=0.3, color='C0')
    ax1.plot(dates, dnu_s, color='C0', lw=1.5, label=f'{smooth_window*3}-day smooth')
    ax1.set_ylabel('Freq shift (uHz)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_title('3-Day P-Mode Frequency Shifts vs. Solar Activity (HMI)')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(dates, act, s=1, alpha=0.3, color='C1')
    ax2.plot(dates, act_s, color='C1', lw=1.5)
    labels = {'ssn': 'Daily Sunspot Number', 'f107': 'F10.7 (sfu)'}
    ax2.set_ylabel(labels.get(activity_col, activity_col))
    ax2.grid(True, alpha=0.3)
    
    # Normalized overlay
    dnu_n = (dnu_s - np.nanmean(dnu_s)) / np.nanstd(dnu_s)
    act_n = (act_s - np.nanmean(act_s)) / np.nanstd(act_s)
    ax3.plot(dates, dnu_n, color='C0', lw=1.5, label='Freq shift (norm)')
    ax3.plot(dates, act_n, color='C1', lw=1.5, label=f'{activity_col.upper()} (norm)')
    ax3.set_ylabel('Normalized')
    ax3.set_xlabel('Date')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_xcorr_comparison(results_dict, output='fig_xcorr.png'):
    """Plot cross-correlation for raw vs detrended side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, (title, data) in zip(axes, results_dict.items()):
        lags, cc, peak_lag, peak_corr, sig = data
        lag_days = lags * 3
        
        ax.plot(lag_days, cc, 'C0-', lw=1.5)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5, ls='--')
        ax.axhline(sig, color='red', lw=0.8, ls='--', alpha=0.7)
        ax.axhline(-sig, color='red', lw=0.8, ls='--', alpha=0.7)
        ax.axvline(peak_lag * 3, color='C2', lw=2,
                   label=f'Peak: {peak_lag*3:+d} days (r={peak_corr:.3f})')
        
        ax.set_title(title)
        ax.set_xlabel('Lag (days)\n[positive = freq shift leads]')
        ax.set_ylabel('Correlation')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-200, 200)
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_xcorr_single(lags, cc, peak_lag, peak_corr, ci_low, ci_high,
                      sig, title, output):
    """Single cross-correlation plot with CI."""
    fig, ax = plt.subplots(figsize=(10, 6))
    lag_days = lags * 3
    
    ax.plot(lag_days, cc, 'C0-', lw=1.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5, ls='--')
    ax.axhline(sig, color='red', lw=0.8, ls='--', alpha=0.7, label=f'95% sig ({sig:.3f})')
    ax.axhline(-sig, color='red', lw=0.8, ls='--', alpha=0.7)
    
    ax.axvline(peak_lag * 3, color='C2', lw=2,
               label=f'Peak: {peak_lag*3:+d} days (r={peak_corr:.3f})')
    ax.axvspan(ci_low * 3, ci_high * 3, alpha=0.2, color='C2',
               label=f'95% CI: [{ci_low*3:.0f}, {ci_high*3:.0f}] days')
    
    if peak_lag > 0:
        txt = f"Freq shift LEADS SSN by {peak_lag*3} days"
    elif peak_lag < 0:
        txt = f"SSN LEADS freq shift by {-peak_lag*3} days"
    else:
        txt = "No lead/lag detected"
    
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(title)
    ax.set_xlabel('Lag (days) [positive = freq shift leads]')
    ax.set_ylabel('Correlation coefficient')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-200, 200)
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_bootstrap_hist(peak_lags, peak_lag, ci_low, ci_high, title, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(peak_lags * 3, bins=50, density=True, alpha=0.7, color='C0',
            edgecolor='white', lw=0.5)
    ax.axvline(peak_lag * 3, color='C2', lw=2, label=f'Peak: {peak_lag*3:+d} days')
    ax.axvline(ci_low * 3, color='C3', lw=1.5, ls='--',
               label=f'95% CI: [{ci_low*3:.0f}, {ci_high*3:.0f}] days')
    ax.axvline(ci_high * 3, color='C3', lw=1.5, ls='--')
    ax.axvline(0, color='gray', lw=1, ls=':')
    ax.set_xlabel('Peak lag (days)')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_scatter(x, y, lag, xlabel, ylabel, output):
    n = len(x)
    if lag > 0:
        xp, yp = x[:n-lag], y[lag:]
    elif lag < 0:
        xp, yp = x[-lag:], y[:n+lag]
    else:
        xp, yp = x, y
    
    if len(xp) < 3:
        return
    
    r, p = stats.pearsonr(xp, yp)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xp, yp, s=5, alpha=0.3, color='C0')
    slope, intercept = np.polyfit(xp, yp, 1)
    xfit = np.linspace(np.min(xp), np.max(xp), 100)
    ax.plot(xfit, slope * xfit + intercept, 'C1-', lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'lag = {lag*3:+d} days | r = {r:.3f} | p = {p:.2e}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()
    return r, p


# =============================================================
# Main Analysis
# =============================================================

def analyze_one(dnu, act, label, max_lag, n_bootstrap, odir, tag):
    """Run full analysis on one pair of series."""
    sig = 1.96 / np.sqrt(len(dnu))
    
    # Cross-correlation
    lags, cc = compute_crosscorr(dnu, act, max_lag)
    valid = ~np.isnan(cc)
    peak_idx = np.argmax(cc[valid])
    peak_lag = lags[valid][peak_idx]
    peak_corr = cc[valid][peak_idx]
    zero_corr = cc[lags == 0][0] if 0 in lags else np.nan
    
    print(f"  Zero-lag r = {zero_corr:.4f}")
    print(f"  Peak: lag = {peak_lag*3:+d} days, r = {peak_corr:.4f}")
    
    # Bootstrap
    peak_lag_bs, peak_corr_bs, ci_low, ci_high, peak_lags_bs = \
        block_bootstrap_peak_lag(dnu, act, max_lag, n_bootstrap)
    
    print(f"  Bootstrap 95% CI: [{ci_low*3:.0f}, {ci_high*3:.0f}] days")
    
    if ci_low > 0:
        conclusion = "FREQ SHIFT LEADS"
    elif ci_high < 0:
        conclusion = f"{label} LEADS"
    else:
        conclusion = "NO SIGNIFICANT LEAD/LAG"
    print(f"  --> {conclusion}")
    
    # Plots
    plot_xcorr_single(lags, cc, peak_lag, peak_corr, ci_low, ci_high, sig,
                      f'Cross-Correlation: Freq Shift vs {label} ({tag})',
                      os.path.join(odir, f'xcorr_{tag}.png'))
    
    plot_bootstrap_hist(peak_lags_bs, peak_lag, ci_low, ci_high,
                        f'Bootstrap Peak Lag ({tag})',
                        os.path.join(odir, f'bootstrap_{tag}.png'))
    
    plot_scatter(dnu, act, peak_lag, 'Freq shift (uHz, detrended)',
                 f'{label} (detrended)',
                 os.path.join(odir, f'scatter_{tag}.png'))
    
    return {
        'analysis': tag,
        'n_points': len(dnu),
        'zero_lag_r': round(zero_corr, 4),
        'peak_lag_days': peak_lag * 3,
        'peak_r': round(peak_corr, 4),
        'ci95_low_days': round(ci_low * 3, 0),
        'ci95_high_days': round(ci_high * 3, 0),
        'conclusion': conclusion,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq-csv', default='3day_freq_shifts.csv')
    parser.add_argument('--output-dir', default='results_v2')
    parser.add_argument('--max-lag', type=int, default=60,
                        help='Max lag in 3-day units (default 60 = 180 days)')
    parser.add_argument('--n-bootstrap', type=int, default=2000)
    args = parser.parse_args()
    
    odir = args.output_dir
    os.makedirs(odir, exist_ok=True)
    
    # =========================================================
    # Load data
    # =========================================================
    print("=" * 60)
    print("Loading data")
    print("=" * 60)
    freq_df = load_freq_shifts(args.freq_csv)
    ssn_df = download_daily_sunspot_numbers(os.path.join(odir, 'SN_d_tot_V2.0.csv'))
    f107_df = download_f107_flux(os.path.join(odir, 'f107_daily.csv'))
    
    # Align
    print("\n" + "=" * 60)
    print("Aligning data")
    print("=" * 60)
    aligned = align_to_3day(freq_df, ssn_df, 'ssn')
    
    if f107_df is not None and len(f107_df) > 100:
        aligned_f107 = align_to_3day(freq_df, f107_df, 'f107')
        if len(aligned_f107) < 100:
            aligned_f107 = None
            print("  F10.7 alignment yielded too few points, skipping")
    else:
        aligned_f107 = None
    
    # =========================================================
    # Time series plot
    # =========================================================
    print("\n" + "=" * 60)
    print("Plotting time series")
    print("=" * 60)
    plot_full_timeseries(aligned, 'ssn', smooth_window=9,
                         output=os.path.join(odir, 'fig1_timeseries.png'))
    
    # =========================================================
    # Detrending
    # =========================================================
    print("\n" + "=" * 60)
    print("Detrending (81-day running mean removal)")
    print("=" * 60)
    
    dnu_raw = aligned['freq_shift_uHz'].values
    ssn_raw = aligned['ssn'].values
    dates = aligned['mid_date'].values
    
    # Detrend with multiple cutoffs
    for cutoff in [81, 162, 365]:
        dnu_res, dnu_trend = detrend_series(dnu_raw, 'lowpass', cutoff)
        ssn_res, ssn_trend = detrend_series(ssn_raw, 'lowpass', cutoff)
        
        # Remove NaN edges from detrending
        valid = ~(np.isnan(dnu_res) | np.isnan(ssn_res))
        print(f"\n  Cutoff {cutoff} days: {valid.sum()} valid points")
        
        if cutoff == 81:
            # Save for main analysis
            dnu_detrended = dnu_res[valid]
            ssn_detrended = ssn_res[valid]
            dates_valid = dates[valid]
            
            plot_detrending(pd.to_datetime(dates), dnu_raw, dnu_trend, dnu_res,
                           'Freq shift (uHz)', 'Frequency Shift Detrending (81-day)',
                           os.path.join(odir, 'fig2_detrend_dnu.png'))
            plot_detrending(pd.to_datetime(dates), ssn_raw, ssn_trend, ssn_res,
                           'SSN', 'Sunspot Number Detrending (81-day)',
                           os.path.join(odir, 'fig2_detrend_ssn.png'))
    
    # =========================================================
    # Cross-correlation analysis
    # =========================================================
    print("\n" + "=" * 60)
    print("Cross-correlation analysis")
    print("=" * 60)
    
    summary_rows = []
    
    # A) Raw (no detrending) - for comparison
    print("\n--- A) Raw (no detrending) ---")
    row = analyze_one(dnu_raw, ssn_raw, 'SSN', args.max_lag, args.n_bootstrap,
                      odir, 'raw_ssn')
    summary_rows.append(row)
    
    # B) 81-day detrended (main result)
    print("\n--- B) 81-day detrended ---")
    row = analyze_one(dnu_detrended, ssn_detrended, 'SSN', args.max_lag,
                      args.n_bootstrap, odir, 'detrend81_ssn')
    summary_rows.append(row)
    
    # C) 162-day detrended
    print("\n--- C) 162-day detrended ---")
    dnu_r162, _ = detrend_series(dnu_raw, 'lowpass', 162)
    ssn_r162, _ = detrend_series(ssn_raw, 'lowpass', 162)
    v = ~(np.isnan(dnu_r162) | np.isnan(ssn_r162))
    row = analyze_one(dnu_r162[v], ssn_r162[v], 'SSN', args.max_lag,
                      args.n_bootstrap, odir, 'detrend162_ssn')
    summary_rows.append(row)
    
    # D) 365-day detrended
    print("\n--- D) 365-day detrended ---")
    dnu_r365, _ = detrend_series(dnu_raw, 'lowpass', 365)
    ssn_r365, _ = detrend_series(ssn_raw, 'lowpass', 365)
    v = ~(np.isnan(dnu_r365) | np.isnan(ssn_r365))
    row = analyze_one(dnu_r365[v], ssn_r365[v], 'SSN', args.max_lag,
                      args.n_bootstrap, odir, 'detrend365_ssn')
    summary_rows.append(row)
    
    # E) Detrended + smoothed (to reduce noise, see short-term lead/lag)
    print("\n--- E) 81-day detrended + 9-point (27-day) smooth ---")
    dnu_ds = smooth_series(dnu_detrended, 9)
    ssn_ds = smooth_series(ssn_detrended, 9)
    v = ~(np.isnan(dnu_ds) | np.isnan(ssn_ds))
    row = analyze_one(dnu_ds[v], ssn_ds[v], 'SSN', args.max_lag,
                      args.n_bootstrap, odir, 'detrend81_smooth27_ssn')
    summary_rows.append(row)
    
    # F) F10.7 if available
    if aligned_f107 is not None and len(aligned_f107) > 100:
        print("\n--- F) F10.7 (81-day detrended) ---")
        f107_raw = aligned_f107['f107'].values
        dnu_f = aligned_f107['freq_shift_uHz'].values
        dnu_fr, _ = detrend_series(dnu_f, 'lowpass', 81)
        f107_r, _ = detrend_series(f107_raw, 'lowpass', 81)
        v = ~(np.isnan(dnu_fr) | np.isnan(f107_r))
        if v.sum() > 100:
            row = analyze_one(dnu_fr[v], f107_r[v], 'F10.7', args.max_lag,
                              args.n_bootstrap, odir, 'detrend81_f107')
            summary_rows.append(row)
    
    # =========================================================
    # Comparison: raw vs detrended cross-corr
    # =========================================================
    print("\n" + "=" * 60)
    print("Generating comparison plot")
    print("=" * 60)
    
    lags_raw, cc_raw = compute_crosscorr(dnu_raw, ssn_raw, args.max_lag)
    sig_raw = 1.96 / np.sqrt(len(dnu_raw))
    vr = ~np.isnan(cc_raw)
    pr_idx = np.argmax(cc_raw[vr])
    
    lags_det, cc_det = compute_crosscorr(dnu_detrended, ssn_detrended, args.max_lag)
    sig_det = 1.96 / np.sqrt(len(dnu_detrended))
    vd = ~np.isnan(cc_det)
    pd_idx = np.argmax(cc_det[vd])
    
    plot_xcorr_comparison(
        {
            'Raw (with solar cycle trend)': (
                lags_raw, cc_raw, lags_raw[vr][pr_idx],
                cc_raw[vr][pr_idx], sig_raw
            ),
            '81-day Detrended (short-term only)': (
                lags_det, cc_det, lags_det[vd][pd_idx],
                cc_det[vd][pd_idx], sig_det
            ),
        },
        os.path.join(odir, 'fig3_xcorr_comparison.png')
    )
    
    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(odir, 'leadlag_summary.csv'), index=False)
    
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    for _, row in summary_df.iterrows():
        print(f"  {row['analysis']:30s}: peak at {row['peak_lag_days']:+4.0f} days "
              f"(r={row['peak_r']:.3f}), CI=[{row['ci95_low_days']:+.0f}, "
              f"{row['ci95_high_days']:+.0f}] => {row['conclusion']}")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
  The raw cross-correlation shows r=0.90 at lag=0, dominated by the
  shared solar-cycle trend. After removing the 81-day trend, we isolate
  short-term (weeks-scale) covariations.
  
  Key question: Does the detrended peak shift away from lag=0?
  - If peak remains at ~0: frequency shifts and SSN vary synchronously
  - If peak is positive: frequency shifts lead (precursor signal)  
  - If peak is negative: SSN leads (surface activity drives p-modes)
  
  See the summary table above for quantitative results.
""")


if __name__ == '__main__':
    main()