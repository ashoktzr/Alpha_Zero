# app/cluster_analysis.py
"""
Post-clustering feature importance analysis and confidence-based filtering.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import streamlit as st

# --- From clustering.py ---

def run_clustering(X: pd.DataFrame, method: str='hdbscan', params: Dict=None) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Run clustering on DataFrame X (rows = samples, index preserved).
    """
    params = params or {}
    Xnum = X.fillna(0).astype(float)
    labels_arr = None

    # Respect an optional n_jobs param (if present)
    n_jobs = params.get('n_jobs', None)

    if method == 'hdbscan':
        try:
            import hdbscan
            hdb_params = {k: v for k, v in params.items() if k not in ['n_jobs', 'n_clusters', 'n_init']}
            if n_jobs is not None:
                hdb_params.setdefault('core_dist_n_jobs', n_jobs)
                # hdb_params.setdefault('n_jobs', n_jobs) # Some versions don't support n_jobs in init
            clusterer = hdbscan.HDBSCAN(**hdb_params)
            labels_arr = clusterer.fit_predict(Xnum.values)
        except Exception as e:
            st.error(f"HDBSCAN failed: {e}")
            print(f"HDBSCAN failed: {e}")
            method = 'kmeans'  # fallback

    if method == 'kmeans':
        from sklearn.cluster import KMeans
        k = int(params.get('n_clusters', 8))
        n_init = params.get('n_init', 'auto')
        km = KMeans(n_clusters=k, random_state=42, n_init=n_init)
        labels_arr = km.fit_predict(Xnum.values)

    if labels_arr is None:
        labels = pd.Series([-1] * len(X), index=X.index, name='cluster_id', dtype=int)
    else:
        labels = pd.Series(labels_arr.astype(int), index=X.index, name='cluster_id')

    # fast computation of sizes
    uniq, counts = np.unique(labels.values, return_counts=True)
    stats = pd.DataFrame({'cluster_id': uniq, 'size': counts})
    stats = stats.sort_values('cluster_id').reset_index(drop=True)

    return labels, stats

def compute_cluster_detailed_stats(df: pd.DataFrame, labels_col: str = 'cluster_id', dur_cap: Optional[int] = None) -> pd.DataFrame:
    """
    Compute detailed per-cluster statistics using vectorized operations.
    """
    if labels_col not in df.columns:
        raise ValueError(f"{labels_col} not in df")

    df = df.copy()
    
    # Pre-calculate boolean flags and numeric values
    df['is_long_tp'] = (df['long_exit'] == 'TP')
    df['is_short_tp'] = (df['short_exit'] == 'TP')
    
    # Ensure numeric types
    df['long_duration'] = pd.to_numeric(df['long_duration'], errors='coerce')
    df['short_duration'] = pd.to_numeric(df['short_duration'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['exit_price'] = pd.to_numeric(df['exit_price'], errors='coerce')
    
    # PnL
    # Long PnL: (Exit - Entry) / Entry
    df['long_pnl'] = (df['exit_price'] - df['close']) / df['close']
    # Short PnL: (Entry - Exit) / Entry
    df['short_pnl'] = (df['close'] - df['exit_price']) / df['close']
    
    # Capped durations
    if dur_cap is not None:
        df['long_capped'] = (df['is_long_tp']) & (df['long_duration'] > dur_cap)
        df['short_capped'] = (df['is_short_tp']) & (df['short_duration'] > dur_cap)
    else:
        df['long_capped'] = False
        df['short_capped'] = False

    # Group by cluster
    g = df.groupby(labels_col)
    
    # Aggregations
    stats = g.agg(
        size=('close', 'count'),
        long_tp_count=('is_long_tp', 'sum'),
        short_tp_count=('is_short_tp', 'sum'),
        avg_long_dur=('long_duration', lambda x: x[df.loc[x.index, 'is_long_tp']].mean()),
        median_long_dur=('long_duration', lambda x: x[df.loc[x.index, 'is_long_tp']].median()),
        avg_short_dur=('short_duration', lambda x: x[df.loc[x.index, 'is_short_tp']].mean()),
        median_short_dur=('short_duration', lambda x: x[df.loc[x.index, 'is_short_tp']].median()),

        avg_long_pnl=('long_pnl', 'mean'),
        avg_short_pnl=('short_pnl', 'mean'),
        long_capped_count=('long_capped', 'sum'),
        short_capped_count=('short_capped', 'sum')
    ).reset_index()
    
    # Global counts for Recall/Specificity calculations
    total_long_tp = df['is_long_tp'].sum()
    total_short_tp = df['is_short_tp'].sum()
    total_rows = len(df)
    
    # Derived Metrics
    stats['coverage'] = stats['size'] / total_rows
    
    # Long Metrics
    stats['long_TP'] = stats['long_tp_count']
    stats['long_FP'] = stats['size'] - stats['long_TP']
    stats['long_precision'] = stats['long_TP'] / stats['size']
    
    # Short Metrics
    stats['short_TP'] = stats['short_tp_count']
    stats['short_FP'] = stats['size'] - stats['short_TP']
    stats['short_precision'] = stats['short_TP'] / stats['size']
    
    
    # Capped Pct
    
    # Capped Pct
    stats['pct_dur_long_capped'] = stats['long_capped_count'] / stats['long_TP'].replace(0, 1)
    stats['pct_dur_short_capped'] = stats['short_capped_count'] / stats['short_TP'].replace(0, 1)
    
    # Cleanup
    stats = stats.drop(columns=['long_capped_count', 'short_capped_count'])
    stats = stats.sort_values('size', ascending=False).reset_index(drop=True)
    
    return stats

# --- Feature Analysis & Filtering ---
# (Removed unused functions)
