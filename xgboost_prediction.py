#!/usr/bin/env python3
"""
Project 3 Step 4: ML Prediction with XGBoost

Question: Can 3-day p-mode frequency shifts improve short-term 
solar activity forecasting beyond what SSN's own history provides?

Approach:
- Model A (baseline): Predict SSN(t+h) from SSN history only
- Model B (helioseismic): Predict SSN(t+h) from SSN history + freq shift features
- Compare RMSE and R^2 at multiple forecast horizons h

Requirements: pip install pandas numpy scikit-learn xgboost matplotlib

Usage:
  python3 xgboost_prediction.py --freq-csv 3day_freq_shifts.csv --output-dir results_ml
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed, using sklearn GradientBoosting instead")
    from sklearn.ensemble import GradientBoostingRegressor

import requests


# =============================================================
# Data Loading (same as lead-lag script)
# =============================================================

def load_freq_shifts(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['start_date', 'mid_date'])
    df = df.sort_values('mdi_day').reset_index(drop=True)
    print(f"Loaded {len(df)} frequency shift measurements")
    return df


def download_daily_sunspot_numbers(cache_path='SN_d_tot_V2.0.csv'):
    if os.path.exists(cache_path):
        pass
    else:
        url = "https://www.sidc.be/SILSO/INFO/sndtotcsv.php"
        print(f"Downloading daily sunspot numbers...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(cache_path, 'wb') as f:
            f.write(resp.content)
    
    df = pd.read_csv(cache_path, sep=';', header=None,
                     names=['year', 'month', 'day', 'decimal_year',
                            'ssn', 'ssn_err', 'n_obs', 'flag'],
                     skipinitialspace=True)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df.loc[df['ssn'] < 0, 'ssn'] = np.nan
    return df[['date', 'ssn']].copy()


def align_data(freq_df, ssn_df):
    """Align SSN to 3-day frequency shift epochs."""
    aligned = freq_df[['daterange', 'start_date', 'mid_date', 'mdi_day',
                        'freq_shift_uHz', 'freq_shift_err_uHz']].copy()
    
    ssn_values = []
    for _, row in freq_df.iterrows():
        start = row['start_date']
        end = start + timedelta(days=2)
        mask = (ssn_df['date'] >= start) & (ssn_df['date'] <= end)
        vals = ssn_df.loc[mask, 'ssn']
        ssn_values.append(vals.mean() if len(vals) > 0 and vals.notna().sum() > 0 else np.nan)
    
    aligned['ssn'] = ssn_values
    aligned = aligned.dropna(subset=['ssn']).reset_index(drop=True)
    print(f"Aligned dataset: {len(aligned)} points")
    return aligned


# =============================================================
# Feature Engineering
# =============================================================

def create_features(df, horizon=1, max_lag=10):
    """Create features for prediction.
    
    Parameters:
        horizon: forecast horizon in 3-day steps (1=3 days, 3=9 days, etc.)
        max_lag: how many past 3-day steps to use as features
    
    Target: SSN at time t+horizon
    
    Features (Model A - SSN only):
        - SSN(t), SSN(t-1), ..., SSN(t-max_lag)
        - SSN rolling mean (9-day, 27-day, 81-day)
        - SSN rolling std (27-day)
        - SSN rate of change
    
    Features (Model B - SSN + helioseismic):
        - All of Model A
        - dnu(t), dnu(t-1), ..., dnu(t-max_lag)
        - dnu rolling mean (9-day, 27-day, 81-day)
        - dnu rolling std (27-day)
        - dnu rate of change
        - dnu - dnu_27day_mean (deviation from trend)
    """
    data = df.copy()
    
    # === SSN features (Model A) ===
    ssn_features = []
    
    # Lagged SSN
    for lag in range(0, max_lag + 1):
        col = f'ssn_lag{lag}'
        data[col] = data['ssn'].shift(lag)
        ssn_features.append(col)
    
    # Rolling statistics
    for w, label in [(3, '9d'), (9, '27d'), (27, '81d')]:
        col = f'ssn_rmean_{label}'
        data[col] = data['ssn'].rolling(w, min_periods=1).mean()
        ssn_features.append(col)
    
    data['ssn_rstd_27d'] = data['ssn'].rolling(9, min_periods=3).std()
    ssn_features.append('ssn_rstd_27d')
    
    # Rate of change
    data['ssn_roc'] = data['ssn'] - data['ssn'].shift(1)
    ssn_features.append('ssn_roc')
    
    data['ssn_roc_9d'] = data['ssn'] - data['ssn'].shift(3)
    ssn_features.append('ssn_roc_9d')
    
    # === Helioseismic features (Model B adds these) ===
    helio_features = []
    
    # Lagged frequency shifts
    for lag in range(0, max_lag + 1):
        col = f'dnu_lag{lag}'
        data[col] = data['freq_shift_uHz'].shift(lag)
        helio_features.append(col)
    
    # Rolling statistics
    for w, label in [(3, '9d'), (9, '27d'), (27, '81d')]:
        col = f'dnu_rmean_{label}'
        data[col] = data['freq_shift_uHz'].rolling(w, min_periods=1).mean()
        helio_features.append(col)
    
    data['dnu_rstd_27d'] = data['freq_shift_uHz'].rolling(9, min_periods=3).std()
    helio_features.append('dnu_rstd_27d')
    
    # Rate of change
    data['dnu_roc'] = data['freq_shift_uHz'] - data['freq_shift_uHz'].shift(1)
    helio_features.append('dnu_roc')
    
    data['dnu_roc_9d'] = data['freq_shift_uHz'] - data['freq_shift_uHz'].shift(3)
    helio_features.append('dnu_roc_9d')
    
    # Deviation from 81-day trend
    data['dnu_dev_81d'] = data['freq_shift_uHz'] - data['freq_shift_uHz'].rolling(27, min_periods=5).mean()
    helio_features.append('dnu_dev_81d')
    
    # === Target ===
    data['target'] = data['ssn'].shift(-horizon)
    
    # Drop rows with NaN
    all_features = ssn_features + helio_features
    data = data.dropna(subset=all_features + ['target']).reset_index(drop=True)
    
    return data, ssn_features, helio_features


# =============================================================
# Model Training and Evaluation
# =============================================================

def train_evaluate(data, features, target_col='target', n_splits=5, model_name='Model'):
    """Train XGBoost with time-series cross-validation.
    
    Returns dict of metrics and feature importances.
    """
    X = data[features].values
    y = data[target_col].values
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    all_preds = np.full(len(y), np.nan)
    all_true = np.full(len(y), np.nan)
    fold_metrics = []
    feature_importance = np.zeros(len(features))
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if HAS_XGB:
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42,
            )
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        fold_metrics.append({'fold': fold, 'rmse': rmse, 'mae': mae, 'r2': r2})
        
        all_preds[test_idx] = preds
        all_true[test_idx] = y_test
        
        if HAS_XGB:
            feature_importance += model.feature_importances_
        else:
            feature_importance += model.feature_importances_
    
    feature_importance /= n_splits
    
    # Overall metrics (on all test folds combined)
    valid = ~np.isnan(all_preds)
    overall_rmse = np.sqrt(mean_squared_error(all_true[valid], all_preds[valid]))
    overall_mae = mean_absolute_error(all_true[valid], all_preds[valid])
    overall_r2 = r2_score(all_true[valid], all_preds[valid])
    
    fold_df = pd.DataFrame(fold_metrics)
    
    return {
        'model_name': model_name,
        'rmse': overall_rmse,
        'mae': overall_mae,
        'r2': overall_r2,
        'rmse_std': fold_df['rmse'].std(),
        'r2_std': fold_df['r2'].std(),
        'predictions': all_preds,
        'true_values': all_true,
        'feature_importance': dict(zip(features, feature_importance)),
        'fold_metrics': fold_df,
    }


# =============================================================
# Visualization
# =============================================================

def plot_predictions(dates, true_vals, pred_a, pred_b, horizon, output):
    """Plot true vs predicted for both models."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              gridspec_kw={'hspace': 0.15, 'height_ratios': [2, 2, 1]})
    
    valid_a = ~np.isnan(pred_a)
    valid_b = ~np.isnan(pred_b)
    
    # Model A
    axes[0].plot(dates[valid_a], true_vals[valid_a], 'k-', lw=0.5, alpha=0.5, label='Observed')
    axes[0].plot(dates[valid_a], pred_a[valid_a], 'C0-', lw=0.8, alpha=0.7, label='Model A (SSN only)')
    axes[0].set_ylabel('SSN')
    axes[0].set_title(f'SSN Prediction (horizon = {horizon*3} days)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Model B
    axes[1].plot(dates[valid_b], true_vals[valid_b], 'k-', lw=0.5, alpha=0.5, label='Observed')
    axes[1].plot(dates[valid_b], pred_b[valid_b], 'C1-', lw=0.8, alpha=0.7, label='Model B (SSN + Helio)')
    axes[1].set_ylabel('SSN')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Residual comparison
    valid = valid_a & valid_b
    res_a = np.abs(true_vals[valid] - pred_a[valid])
    res_b = np.abs(true_vals[valid] - pred_b[valid])
    improvement = res_a - res_b  # positive = B is better
    
    axes[2].fill_between(dates[valid], 0, improvement,
                          where=(improvement > 0), color='C2', alpha=0.5, label='B better')
    axes[2].fill_between(dates[valid], 0, improvement,
                          where=(improvement <= 0), color='C3', alpha=0.5, label='A better')
    axes[2].axhline(0, color='gray', lw=0.5)
    axes[2].set_ylabel('|err_A| - |err_B|')
    axes[2].set_xlabel('Date')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_feature_importance(importance_dict, top_n=20, output='fig_importance.png'):
    """Plot feature importance comparison."""
    sorted_feats = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [f[0] for f in sorted_feats]
    vals = [f[1] for f in sorted_feats]
    
    colors = ['C1' if 'dnu' in n else 'C0' for n in names]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title('XGBoost Feature Importance\n(blue=SSN features, orange=helioseismic features)')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_horizon_comparison(horizon_results, output='fig_horizon.png'):
    """Plot RMSE and R^2 vs forecast horizon for both models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    horizons = [r['horizon_days'] for r in horizon_results if r['model'] == 'A']
    rmse_a = [r['rmse'] for r in horizon_results if r['model'] == 'A']
    rmse_b = [r['rmse'] for r in horizon_results if r['model'] == 'B']
    r2_a = [r['r2'] for r in horizon_results if r['model'] == 'A']
    r2_b = [r['r2'] for r in horizon_results if r['model'] == 'B']
    
    ax1.plot(horizons, rmse_a, 'C0o-', lw=2, ms=8, label='Model A (SSN only)')
    ax1.plot(horizons, rmse_b, 'C1s-', lw=2, ms=8, label='Model B (SSN + Helio)')
    ax1.set_xlabel('Forecast Horizon (days)')
    ax1.set_ylabel('RMSE (SSN units)')
    ax1.set_title('Forecast Error vs Horizon')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(horizons, r2_a, 'C0o-', lw=2, ms=8, label='Model A (SSN only)')
    ax2.plot(horizons, r2_b, 'C1s-', lw=2, ms=8, label='Model B (SSN + Helio)')
    ax2.set_xlabel('Forecast Horizon (days)')
    ax2.set_ylabel('R-squared')
    ax2.set_title('Forecast Skill vs Horizon')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add improvement annotation
    pct_improvements = [(rb - ra) / ra * 100 if ra > 0 else 0 
                        for ra, rb in zip(rmse_a, rmse_b)]
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()
    
    return pct_improvements


def plot_scatter_pred(true_vals, pred_a, pred_b, horizon, output):
    """Scatter plot: predicted vs observed."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    valid_a = ~np.isnan(pred_a)
    valid_b = ~np.isnan(pred_b)
    
    for ax, pred, valid, label, color in [
        (ax1, pred_a, valid_a, 'Model A (SSN only)', 'C0'),
        (ax2, pred_b, valid_b, 'Model B (SSN + Helio)', 'C1')
    ]:
        t = true_vals[valid]
        p = pred[valid]
        r2 = r2_score(t, p)
        
        ax.scatter(t, p, s=3, alpha=0.3, color=color)
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('Observed SSN')
        ax.set_ylabel('Predicted SSN')
        ax.set_title(f'{label}\nR2 = {r2:.4f}')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Prediction vs Observation (horizon = {horizon*3} days)', y=1.02)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


# =============================================================
# Main
# =============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq-csv', default='3day_freq_shifts.csv')
    parser.add_argument('--output-dir', default='results_ml')
    parser.add_argument('--horizons', type=int, nargs='+',
                        default=[1, 3, 5, 9, 15, 27],
                        help='Forecast horizons in 3-day units')
    parser.add_argument('--n-splits', type=int, default=5)
    args = parser.parse_args()
    
    odir = args.output_dir
    os.makedirs(odir, exist_ok=True)
    
    print("=" * 60)
    print("Step 4: XGBoost Solar Activity Prediction")
    print("=" * 60)
    
    # Load
    freq_df = load_freq_shifts(args.freq_csv)
    ssn_df = download_daily_sunspot_numbers(os.path.join(odir, 'SN_d_tot_V2.0.csv'))
    aligned = align_data(freq_df, ssn_df)
    
    # Multi-horizon analysis
    horizon_results = []
    
    print(f"\nTesting {len(args.horizons)} forecast horizons: "
          f"{[h*3 for h in args.horizons]} days")
    print(f"Using {args.n_splits}-fold time series CV\n")
    
    for h in args.horizons:
        h_days = h * 3
        print(f"\n{'='*50}")
        print(f"Horizon: {h_days} days ({h} steps)")
        print(f"{'='*50}")
        
        data, ssn_feats, helio_feats = create_features(aligned, horizon=h, max_lag=10)
        print(f"  Dataset size: {len(data)}")
        print(f"  SSN features: {len(ssn_feats)}")
        print(f"  Helio features: {len(helio_feats)}")
        
        # Model A: SSN only
        print(f"\n  --- Model A (SSN only) ---")
        res_a = train_evaluate(data, ssn_feats, n_splits=args.n_splits,
                               model_name=f'A_h{h_days}d')
        print(f"  RMSE = {res_a['rmse']:.2f} (+/- {res_a['rmse_std']:.2f})")
        print(f"  R2   = {res_a['r2']:.4f} (+/- {res_a['r2_std']:.4f})")
        
        # Model B: SSN + helioseismic
        print(f"\n  --- Model B (SSN + Helioseismic) ---")
        all_feats = ssn_feats + helio_feats
        res_b = train_evaluate(data, all_feats, n_splits=args.n_splits,
                               model_name=f'B_h{h_days}d')
        print(f"  RMSE = {res_b['rmse']:.2f} (+/- {res_b['rmse_std']:.2f})")
        print(f"  R2   = {res_b['r2']:.4f} (+/- {res_b['r2_std']:.4f})")
        
        # Improvement
        rmse_improvement = (res_a['rmse'] - res_b['rmse']) / res_a['rmse'] * 100
        r2_improvement = res_b['r2'] - res_a['r2']
        print(f"\n  RMSE improvement: {rmse_improvement:+.2f}%")
        print(f"  R2 improvement:   {r2_improvement:+.4f}")
        
        horizon_results.append({
            'model': 'A', 'horizon': h, 'horizon_days': h_days,
            'rmse': res_a['rmse'], 'r2': res_a['r2']
        })
        horizon_results.append({
            'model': 'B', 'horizon': h, 'horizon_days': h_days,
            'rmse': res_b['rmse'], 'r2': res_b['r2']
        })
        
        # Plots for key horizons
        if h in [1, 9, 27]:
            dates = data['mid_date'].values
            
            plot_predictions(dates, res_a['true_values'], 
                           res_a['predictions'], res_b['predictions'],
                           h, os.path.join(odir, f'pred_h{h_days}d.png'))
            
            plot_scatter_pred(res_a['true_values'],
                            res_a['predictions'], res_b['predictions'],
                            h, os.path.join(odir, f'scatter_h{h_days}d.png'))
            
            plot_feature_importance(res_b['feature_importance'], top_n=20,
                                   output=os.path.join(odir, f'importance_h{h_days}d.png'))
    
    # Horizon comparison plot
    pct_impr = plot_horizon_comparison(
        horizon_results, os.path.join(odir, 'fig_horizon_comparison.png'))
    
    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    
    print(f"\n{'Horizon':>10s} {'RMSE_A':>8s} {'RMSE_B':>8s} {'Impr%':>7s} "
          f"{'R2_A':>8s} {'R2_B':>8s} {'R2_diff':>8s}")
    print("-" * 60)
    
    for h in args.horizons:
        h_days = h * 3
        a = [r for r in horizon_results if r['model'] == 'A' and r['horizon'] == h][0]
        b = [r for r in horizon_results if r['model'] == 'B' and r['horizon'] == h][0]
        impr = (a['rmse'] - b['rmse']) / a['rmse'] * 100
        print(f"{h_days:>7d} d  {a['rmse']:>8.2f} {b['rmse']:>8.2f} {impr:>+6.2f}% "
              f"{a['r2']:>8.4f} {b['r2']:>8.4f} {b['r2']-a['r2']:>+8.4f}")
    
    # Save results
    results_df = pd.DataFrame(horizon_results)
    results_df.to_csv(os.path.join(odir, 'ml_results.csv'), index=False)
    
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    
    # Check if helio features help at any horizon
    improvements = []
    for h in args.horizons:
        a = [r for r in horizon_results if r['model'] == 'A' and r['horizon'] == h][0]
        b = [r for r in horizon_results if r['model'] == 'B' and r['horizon'] == h][0]
        improvements.append({
            'horizon_days': h * 3,
            'rmse_pct': (a['rmse'] - b['rmse']) / a['rmse'] * 100,
            'r2_diff': b['r2'] - a['r2']
        })
    
    best = max(improvements, key=lambda x: x['rmse_pct'])
    worst = min(improvements, key=lambda x: x['rmse_pct'])
    
    print(f"""
  Model A: Predict SSN from its own history (autoregressive)
  Model B: Predict SSN from its history + p-mode frequency shifts
  
  Best improvement from helioseismic features:
    Horizon {best['horizon_days']} days: RMSE improved by {best['rmse_pct']:+.2f}%, R2 improved by {best['r2_diff']:+.4f}
  
  Worst case:
    Horizon {worst['horizon_days']} days: RMSE changed by {worst['rmse_pct']:+.2f}%, R2 changed by {worst['r2_diff']:+.4f}
  
  Interpretation:
    If improvements are small (<1-2%), helioseismic data provides
    redundant information -- SSN's own history already captures the
    predictable component. This is consistent with the lead-lag
    analysis showing synchronous (lag=0) variation.
    
    If improvements are larger at longer horizons, helioseismic 
    features may capture subsurface processes that surface activity
    indices miss, providing genuine predictive value.
""")
    
    print(f"All results saved to: {odir}/")


if __name__ == '__main__':
    main()
