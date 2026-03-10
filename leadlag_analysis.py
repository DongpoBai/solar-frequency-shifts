#!/usr/bin/env python3
"""
Project 3 Step 3: Lead-Lag Analysis
3-day p-mode frequency shifts vs. solar activity indices

Requirements: pip install pandas numpy scipy matplotlib requests

Usage: python3 leadlag_analysis.py --freq-csv 3day_freq_shifts.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import signal, stats
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import io


# =============================================================
# Data Loading
# =============================================================

def load_freq_shifts(csv_path):
    """Load 3-day frequency shift CSV."""
    df = pd.read_csv(csv_path, parse_dates=['start_date', 'mid_date'])
    df = df.sort_values('mdi_day').reset_index(drop=True)
    print(f"Loaded {len(df)} frequency shift measurements")
    print(f"  Date range: {df['mid_date'].min()} to {df['mid_date'].max()}")
    print(f"  Shift range: {df['freq_shift_uHz'].min():.4f} to {df['freq_shift_uHz'].max():.4f} uHz")
    return df


def download_daily_sunspot_numbers(cache_path='SN_d_tot_V2.0.csv'):
    """Download daily total sunspot number from SILSO/WDC-SILSO.
    
    Source: https://www.sidc.be/SILSO/datafiles
    """
    if os.path.exists(cache_path):
        print(f"Using cached sunspot data: {cache_path}")
    else:
        url = "https://www.sidc.be/SILSO/INFO/sndtotcsv.php"
        print(f"Downloading daily sunspot numbers from SILSO...")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(cache_path, 'wb') as f:
                f.write(resp.content)
            print(f"  Saved to {cache_path}")
        except Exception as e:
            print(f"  Download failed: {e}")
            print("  Please download manually from https://www.sidc.be/SILSO/datafiles")
            print(f"  Save as {cache_path}")
            raise
    
    # Parse SILSO daily CSV format:
    # year;month;day;decimal_year;SNvalue;SNerror;Nb_observations;definitive/provisional
    df = pd.read_csv(cache_path, sep=';', header=None,
                     names=['year', 'month', 'day', 'decimal_year', 
                            'ssn', 'ssn_err', 'n_obs', 'flag'],
                     skipinitialspace=True)
    
    # Build date column
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    # Replace -1 (missing) with NaN
    df.loc[df['ssn'] < 0, 'ssn'] = np.nan
    
    print(f"Loaded {len(df)} daily sunspot records")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df[['date', 'ssn', 'ssn_err']].copy()


def download_f107_flux(cache_path='f107_daily.csv'):
    """Download F10.7 solar radio flux from LISIRD/LASP.
    
    F10.7 is another common solar activity proxy, sometimes better
    correlated with p-mode shifts than sunspot number.
    """
    if os.path.exists(cache_path):
        print(f"Using cached F10.7 data: {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=['date'])
        print(f"Loaded {len(df)} F10.7 records")
        return df
    
    # LISIRD API
    url = ("https://lasp.colorado.edu/lisird/latis/dap/penticton_radio_flux.csv"
           "?time>=2010-01-01&time<=2025-12-31")
    print(f"Downloading F10.7 flux from LISIRD...")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        # Rename columns
        df.columns = [c.strip() for c in df.columns]
        if 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'])
        elif df.columns[0]:
            df['date'] = pd.to_datetime(df.iloc[:, 0])
        
        # Find the flux column
        flux_col = [c for c in df.columns if 'flux' in c.lower() or 'f107' in c.lower()]
        if flux_col:
            df['f107'] = pd.to_numeric(df[flux_col[0]], errors='coerce')
        else:
            df['f107'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        
        df = df[['date', 'f107']].dropna()
        df.to_csv(cache_path, index=False)
        print(f"  Saved {len(df)} records to {cache_path}")
        return df
    except Exception as e:
        print(f"  F10.7 download failed: {e}")
        print("  Continuing without F10.7 data")
        return None


# =============================================================
# Alignment and Preprocessing
# =============================================================

def align_to_3day(freq_df, activity_df, activity_col='ssn'):
    """Align daily activity data to 3-day frequency shift epochs.
    
    For each 3-day interval, compute the mean activity over those 3 days.
    """
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
    
    # Drop NaN
    n_before = len(aligned)
    aligned = aligned.dropna(subset=[activity_col]).reset_index(drop=True)
    print(f"Aligned: {len(aligned)}/{n_before} points have both freq shift and {activity_col}")
    
    return aligned


def smooth_series(x, window=7):
    """Apply running mean smoothing."""
    return pd.Series(x).rolling(window, center=True, min_periods=1).mean().values


# =============================================================
# Cross-Correlation Analysis
# =============================================================

def compute_crosscorr(x, y, max_lag=100):
    """Compute normalized cross-correlation between x and y.
    
    Positive lag means x leads y (x is shifted to the right).
    For our case: positive lag means freq shift leads activity.
    
    Returns (lags, correlation_coefficients).
    """
    # Standardize
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        if lag >= 0:
            # x leads: compare x[:-lag] with y[lag:]
            if lag == 0:
                cc[i] = np.corrcoef(x, y)[0, 1]
            elif lag < n:
                cc[i] = np.corrcoef(x[:n-lag], y[lag:])[0, 1]
        else:
            # y leads: compare x[-lag:] with y[:n+lag]
            alag = -lag
            if alag < n:
                cc[i] = np.corrcoef(x[alag:], y[:n-alag])[0, 1]
    
    return lags, cc


def bootstrap_peak_lag(x, y, max_lag=100, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval for the peak lag.
    
    Returns (peak_lag, peak_corr, ci_low, ci_high).
    """
    n = len(x)
    peak_lags = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        idx = np.sort(idx)  # preserve temporal order (block bootstrap better, but start simple)
        lags, cc = compute_crosscorr(x[idx], y[idx], max_lag)
        peak_lags.append(lags[np.argmax(cc)])
    
    peak_lags = np.array(peak_lags)
    
    # Full data peak
    lags, cc = compute_crosscorr(x, y, max_lag)
    peak_idx = np.argmax(cc)
    peak_lag = lags[peak_idx]
    peak_corr = cc[peak_idx]
    
    alpha = 1 - ci
    ci_low = np.percentile(peak_lags, 100 * alpha / 2)
    ci_high = np.percentile(peak_lags, 100 * (1 - alpha / 2))
    
    return peak_lag, peak_corr, ci_low, ci_high, peak_lags


def significance_threshold(n, alpha=0.05):
    """Approximate significance threshold for cross-correlation.
    
    Under H0 (no correlation), r ~ N(0, 1/sqrt(n)).
    """
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    return z / np.sqrt(n)


# =============================================================
# Visualization
# =============================================================

def plot_time_series(aligned, activity_col='ssn', smooth_window=9, 
                     output='fig_timeseries.png'):
    """Plot frequency shift and activity index time series."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), 
                                         sharex=True, gridspec_kw={'hspace': 0.08})
    
    dates = aligned['mid_date']
    dnu = aligned['freq_shift_uHz'].values
    act = aligned[activity_col].values
    
    dnu_smooth = smooth_series(dnu, smooth_window)
    act_smooth = smooth_series(act, smooth_window)
    
    # Panel 1: Raw frequency shift
    ax1.scatter(dates, dnu, s=1, alpha=0.3, color='C0', label='3-day raw')
    ax1.plot(dates, dnu_smooth, color='C0', linewidth=1.5, 
             label=f'{smooth_window*3}-day smooth')
    ax1.set_ylabel('Frequency shift (uHz)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_title('3-Day P-Mode Frequency Shifts vs. Solar Activity (HMI)')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Activity index
    ax2.scatter(dates, act, s=1, alpha=0.3, color='C1')
    ax2.plot(dates, act_smooth, color='C1', linewidth=1.5)
    label_map = {'ssn': 'Daily Sunspot Number (ISN)', 'f107': 'F10.7 Flux (sfu)'}
    ax2.set_ylabel(label_map.get(activity_col, activity_col))
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Overlay (normalized)
    dnu_norm = (dnu_smooth - np.nanmean(dnu_smooth)) / np.nanstd(dnu_smooth)
    act_norm = (act_smooth - np.nanmean(act_smooth)) / np.nanstd(act_smooth)
    ax3.plot(dates, dnu_norm, color='C0', linewidth=1.5, label='Freq shift (norm)')
    ax3.plot(dates, act_norm, color='C1', linewidth=1.5, label=f'{activity_col.upper()} (norm)')
    ax3.set_ylabel('Normalized value')
    ax3.set_xlabel('Date')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_crosscorr(lags, cc, peak_lag, peak_corr, ci_low, ci_high, 
                   sig_thresh, activity_label='SSN', smooth_label='',
                   output='fig_crosscorr.png'):
    """Plot cross-correlation function."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert lags from 3-day units to days
    lag_days = lags * 3
    peak_days = peak_lag * 3
    ci_low_days = ci_low * 3
    ci_high_days = ci_high * 3
    
    ax.plot(lag_days, cc, 'C0-', linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    
    # Significance threshold
    ax.axhline(sig_thresh, color='red', linewidth=0.8, linestyle='--', 
               alpha=0.7, label=f'95% significance ({sig_thresh:.3f})')
    ax.axhline(-sig_thresh, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
    
    # Peak marker
    ax.axvline(peak_days, color='C2', linewidth=1.5, linestyle='-',
               label=f'Peak: lag = {peak_days:+.0f} days (r = {peak_corr:.3f})')
    
    # CI shading
    ax.axvspan(ci_low_days, ci_high_days, alpha=0.2, color='C2',
               label=f'95% CI: [{ci_low_days:.0f}, {ci_high_days:.0f}] days')
    
    # Annotation
    if peak_days > 0:
        lead_text = f"Freq shift LEADS {activity_label} by {abs(peak_days):.0f} days"
    elif peak_days < 0:
        lead_text = f"{activity_label} LEADS freq shift by {abs(peak_days):.0f} days"
    else:
        lead_text = "No lead/lag detected"
    
    ax.text(0.02, 0.98, lead_text, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    title = f'Cross-Correlation: Frequency Shift vs. {activity_label}'
    if smooth_label:
        title += f' ({smooth_label})'
    ax.set_title(title)
    ax.set_xlabel('Lag (days)\n[positive = freq shift leads]')
    ax.set_ylabel('Correlation coefficient')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-300, 300)
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_bootstrap_hist(peak_lags, peak_lag, ci_low, ci_high,
                        activity_label='SSN', output='fig_bootstrap.png'):
    """Plot bootstrap distribution of peak lags."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    peak_lags_days = peak_lags * 3
    
    ax.hist(peak_lags_days, bins=50, density=True, alpha=0.7, color='C0',
            edgecolor='white', linewidth=0.5)
    ax.axvline(peak_lag * 3, color='C2', linewidth=2, linestyle='-',
               label=f'Peak: {peak_lag*3:+.0f} days')
    ax.axvline(ci_low * 3, color='C3', linewidth=1.5, linestyle='--',
               label=f'95% CI: [{ci_low*3:.0f}, {ci_high*3:.0f}] days')
    ax.axvline(ci_high * 3, color='C3', linewidth=1.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=1, linestyle=':')
    
    ax.set_xlabel('Peak lag (days)')
    ax.set_ylabel('Density')
    ax.set_title(f'Bootstrap Distribution of Peak Lag (Freq Shift vs. {activity_label})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_scatter_at_lag(dnu, act, lag, activity_label='SSN', 
                        output='fig_scatter.png'):
    """Scatter plot at optimal lag."""
    n = len(dnu)
    if lag >= 0:
        x = dnu[:n-lag] if lag > 0 else dnu
        y = act[lag:] if lag > 0 else act
    else:
        alag = -lag
        x = dnu[alag:]
        y = act[:n-alag]
    
    r, p = stats.pearsonr(x, y)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=5, alpha=0.3, color='C0')
    
    # Regression line
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    ax.plot(x_fit, slope * x_fit + intercept, 'C1-', linewidth=2)
    
    ax.set_xlabel('Frequency shift (uHz)')
    ax.set_ylabel(activity_label)
    ax.set_title(f'Freq Shift vs. {activity_label} at lag = {lag*3:+d} days\n'
                 f'r = {r:.3f}, p = {p:.2e}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()
    
    return r, p


# =============================================================
# Main
# =============================================================

def main():
    parser = argparse.ArgumentParser(description='Lead-lag analysis')
    parser.add_argument('--freq-csv', default='3day_freq_shifts.csv')
    parser.add_argument('--output-dir', default='project3_results')
    parser.add_argument('--max-lag', type=int, default=100,
                        help='Max lag in 3-day units (default 100 = 300 days)')
    parser.add_argument('--smooth', type=int, nargs='+', default=[1, 3, 9, 27],
                        help='Smoothing windows in 3-day units')
    parser.add_argument('--n-bootstrap', type=int, default=2000)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("=" * 60)
    print("Loading data")
    print("=" * 60)
    freq_df = load_freq_shifts(args.freq_csv)
    ssn_df = download_daily_sunspot_numbers(
        os.path.join(args.output_dir, 'SN_d_tot_V2.0.csv'))
    
    # Try F10.7
    f107_df = download_f107_flux(os.path.join(args.output_dir, 'f107_daily.csv'))
    
    # Align
    print("\n" + "=" * 60)
    print("Aligning data")
    print("=" * 60)
    aligned_ssn = align_to_3day(freq_df, ssn_df, 'ssn')
    
    aligned_f107 = None
    if f107_df is not None:
        try:
            aligned_f107 = align_to_3day(freq_df, f107_df, 'f107')
        except Exception as e:
            print(f"F10.7 alignment failed: {e}")
    
    # Time series plots
    print("\n" + "=" * 60)
    print("Plotting time series")
    print("=" * 60)
    plot_time_series(aligned_ssn, 'ssn', smooth_window=9,
                     output=os.path.join(args.output_dir, 'fig1_timeseries_ssn.png'))
    if aligned_f107 is not None:
        plot_time_series(aligned_f107, 'f107', smooth_window=9,
                         output=os.path.join(args.output_dir, 'fig1_timeseries_f107.png'))
    
    # Cross-correlation analysis at multiple smoothing levels
    print("\n" + "=" * 60)
    print("Cross-correlation analysis")
    print("=" * 60)
    
    datasets = [('SSN', aligned_ssn, 'ssn')]
    if aligned_f107 is not None:
        datasets.append(('F10.7', aligned_f107, 'f107'))
    
    summary_rows = []
    
    for label, aligned, col in datasets:
        print(f"\n--- {label} ---")
        
        for sw in args.smooth:
            if sw == 1:
                dnu = aligned['freq_shift_uHz'].values
                act = aligned[col].values
                smooth_label = 'raw (3-day)'
            else:
                dnu = smooth_series(aligned['freq_shift_uHz'].values, sw)
                act = smooth_series(aligned[col].values, sw)
                smooth_label = f'{sw*3}-day smooth'
            
            # Remove NaN from smoothing edges
            valid = ~(np.isnan(dnu) | np.isnan(act))
            dnu = dnu[valid]
            act = act[valid]
            
            print(f"\n  Smoothing: {smooth_label} ({len(dnu)} points)")
            
            # Cross-correlation
            lags, cc = compute_crosscorr(dnu, act, args.max_lag)
            
            # Significance
            sig = significance_threshold(len(dnu))
            
            # Peak
            peak_idx = np.argmax(cc)
            peak_lag = lags[peak_idx]
            peak_corr = cc[peak_idx]
            
            # Zero-lag correlation
            zero_idx = np.where(lags == 0)[0][0]
            zero_corr = cc[zero_idx]
            
            print(f"  Zero-lag r = {zero_corr:.4f}")
            print(f"  Peak: lag = {peak_lag} ({peak_lag*3:+d} days), r = {peak_corr:.4f}")
            print(f"  95% significance threshold: {sig:.4f}")
            
            # Bootstrap CI for peak lag
            peak_lag_bs, peak_corr_bs, ci_low, ci_high, peak_lags_bs = \
                bootstrap_peak_lag(dnu, act, args.max_lag, args.n_bootstrap)
            
            print(f"  Bootstrap peak: {peak_lag_bs*3:+d} days, "
                  f"95% CI: [{ci_low*3:.0f}, {ci_high*3:.0f}] days")
            
            # Is peak significantly different from zero?
            if ci_low > 0:
                conclusion = f"FREQ SHIFT LEADS {label}"
            elif ci_high < 0:
                conclusion = f"{label} LEADS FREQ SHIFT"
            else:
                conclusion = "NO SIGNIFICANT LEAD/LAG"
            print(f"  --> {conclusion}")
            
            summary_rows.append({
                'activity': label,
                'smoothing': smooth_label,
                'n_points': len(dnu),
                'zero_lag_r': round(zero_corr, 4),
                'peak_lag_3day': peak_lag,
                'peak_lag_days': peak_lag * 3,
                'peak_r': round(peak_corr, 4),
                'ci95_low_days': round(ci_low * 3, 0),
                'ci95_high_days': round(ci_high * 3, 0),
                'conclusion': conclusion,
            })
            
            # Plots
            sw_tag = f'{sw*3}d'
            col_tag = col
            
            plot_crosscorr(lags, cc, peak_lag, peak_corr, ci_low, ci_high,
                          sig, label, smooth_label,
                          os.path.join(args.output_dir, 
                                      f'fig2_xcorr_{col_tag}_{sw_tag}.png'))
            
            if sw in [1, 9]:  # only for raw and 27-day
                plot_bootstrap_hist(peak_lags_bs, peak_lag, ci_low, ci_high, label,
                                   os.path.join(args.output_dir,
                                               f'fig3_bootstrap_{col_tag}_{sw_tag}.png'))
                
                plot_scatter_at_lag(dnu, act, peak_lag, label,
                                   os.path.join(args.output_dir,
                                               f'fig4_scatter_{col_tag}_{sw_tag}.png'))
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(os.path.join(args.output_dir, 'leadlag_summary.csv'), index=False)
    print(f"\nSaved summary: {os.path.join(args.output_dir, 'leadlag_summary.csv')}")
    
    # Final conclusion
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    for _, row in summary_df.iterrows():
        print(f"  {row['activity']:6s} {row['smoothing']:20s}: "
              f"peak at {row['peak_lag_days']:+4.0f} days (r={row['peak_r']:.3f}), "
              f"CI=[{row['ci95_low_days']:+.0f}, {row['ci95_high_days']:+.0f}] days "
              f"=> {row['conclusion']}")


if __name__ == '__main__':
    main()