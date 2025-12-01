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
            hdb_params = {k: v for k, v in params.items() if k != 'n_jobs'}
            if n_jobs is not None:
                hdb_params.setdefault('core_dist_n_jobs', n_jobs)
                # hdb_params.setdefault('n_jobs', n_jobs) # Some versions don't support n_jobs in init
            clusterer = hdbscan.HDBSCAN(**hdb_params)
            labels_arr = clusterer.fit_predict(Xnum.values)
        except Exception:
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

def analyze_cluster_features(df_fs: pd.DataFrame, cluster_id: int, feature_cols: list) -> Dict[str, float]:
    """
    Analyze which features are most important for distinguishing a cluster.
    """
    # Create binary labels: cluster vs all others
    y = (df_fs['cluster_id'] == cluster_id).astype(int)
    
    # Extract features
    X = df_fs[feature_cols].fillna(0).astype(float)
    
    # Train Random Forest classifier
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    
    # Extract and normalize feature importances
    importances = rf.feature_importances_
    importances_pct = (importances / importances.sum()) * 100
    
    # Create dict sorted by importance
    feature_importance = {
        feature: float(imp) 
        for feature, imp in zip(feature_cols, importances_pct)
    }
    
    # Sort by importance descending
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    return feature_importance


def compute_cluster_confidence(
    df_fs: pd.DataFrame, 
    cluster_id: int, 
    feature_cols: list,
    feature_importance: Dict[str, float]
) -> pd.Series:
    """
    Compute confidence score for ALL trades relative to a specific cluster centroid.
    Vectorized implementation.
    """
    # 1. Calculate Centroid (using only trades in the cluster)
    cluster_mask = df_fs['cluster_id'] == cluster_id
    cluster_data = df_fs.loc[cluster_mask, feature_cols].fillna(0).astype(float)
    
    if len(cluster_data) == 0:
        return pd.Series(0.0, index=df_fs.index, name='confidence')
        
    centroid = cluster_data.mean(axis=0)
    
    # 2. Prepare Data (All trades)
    X = df_fs[feature_cols].fillna(0).astype(float)
    
    # 3. Feature Selection & Weighting
    # Use top features that make up 80% of importance
    importance_array = np.array([feature_importance.get(f, 0) for f in feature_cols])
    sorted_idx = np.argsort(importance_array)[::-1]
    cumsum = np.cumsum(importance_array[sorted_idx])
    top_features_idx = sorted_idx[cumsum <= 80]
    
    if len(top_features_idx) == 0:
        top_features_idx = sorted_idx[:min(10, len(sorted_idx))]
        
    # Filter to top features
    X_top = X.iloc[:, top_features_idx].values
    centroid_top = centroid.iloc[top_features_idx].values
    weights = importance_array[top_features_idx]
    weights = weights / weights.sum() # Normalize weights to sum to 1
    
    # 4. Standardization (Fit on cluster data, transform all data)
    # We standardize based on the CLUSTER'S distribution to measure deviation from IT.
    # Alternatively, we could standardize on global data. 
    # Standardizing on global data is safer for outliers.
    scaler = StandardScaler()
    scaler.fit(X_top) # Fit on global distribution
    
    X_scaled = scaler.transform(X_top)
    centroid_scaled = scaler.transform(centroid_top.reshape(1, -1))[0]
    
    # 5. Weighted Euclidean Distance
    # dist = sqrt(sum(w * (x - c)^2))
    diff_sq = (X_scaled - centroid_scaled) ** 2
    weighted_diff = diff_sq * weights
    distances = np.sqrt(np.sum(weighted_diff, axis=1))
    
    # 6. Convert to Confidence (0-1)
    # Use 95th percentile of distances as "far" reference
    max_dist = np.percentile(distances, 95)
    if max_dist == 0:
        confidence = np.ones(len(distances))
    else:
        normalized_dist = distances / max_dist
        confidence = 1 / (1 + normalized_dist)
        
    return pd.Series(confidence, index=df_fs.index, name='confidence')


def add_confusion_tags(df_fs: pd.DataFrame, cluster_id: int, direction: str = 'long') -> pd.DataFrame:
    """
    Tag trades as TP, FP, TN, FN relative to the selected cluster and direction.
    Returns DataFrame with 'confusion_tag' column.
    """
    df = df_fs.copy()
    
    # Define Ground Truth (Success) based on direction
    if direction == 'long':
        is_success = (df['long_exit'] == 'TP')
    elif direction == 'short':
        is_success = (df['short_exit'] == 'TP')
    else:
        # Fallback
        is_success = (df['long_exit'] == 'TP')
    
    # Define Prediction (Cluster Membership)
    # A trade is "Predicted Positive" if it is in the cluster.
    is_in_cluster = (df['cluster_id'] == cluster_id)
    
    # Vectorized Tagging
    conditions = [
        (is_in_cluster & is_success),      # TP: In cluster AND Success
        (is_in_cluster & ~is_success),     # FP: In cluster AND Fail
        (~is_in_cluster & ~is_success),    # TN: Not in cluster AND Fail
        (~is_in_cluster & is_success)      # FN: Not in cluster AND Success
    ]
    choices = ['TP', 'FP', 'TN', 'FN']
    
    df['confusion_tag'] = np.select(conditions, choices, default='Unknown')
    
    return df


def analyze_tag_features(
    df_fs: pd.DataFrame, 
    cluster_id: int, 
    target_tag: str, 
    feature_cols: list,
    scope: str = 'cluster'
) -> Dict[str, float]:
    """
    Analyze features that distinguish a specific confusion tag (e.g., 'FP') 
    from the rest.
    
    Args:
        scope: 'cluster' (compare against rest of cluster) or 'global' (compare against all other data)
    """
    if scope == 'cluster':
        # Filter to just this cluster
        analysis_df = df_fs[df_fs['cluster_id'] == cluster_id].copy()
    else:
        # Global scope: Use entire dataset
        analysis_df = df_fs.copy()
    
    if 'confusion_tag' not in analysis_df.columns:
        # We need tags. If global, we need to ensure tags are generated for the specific cluster context
        # But wait, 'confusion_tag' is relative to a cluster.
        # If scope is global, we want to see what makes "TP in Cluster X" distinct from "Everything else".
        # So we must first tag everything relative to Cluster X.
        analysis_df = add_confusion_tags(analysis_df, cluster_id, direction='long')
        
    # Binary Target: 1 if tag matches target (e.g. 'FP'), 0 otherwise
    # Note: In global scope, this means "Is this row a TP in Cluster X?" vs "Is it ANYTHING else (FP in X, or not in X at all)?"
    y = (analysis_df['confusion_tag'] == target_tag).astype(int)
    
    # Need at least some samples of both classes
    if y.sum() < 2 or (len(y) - y.sum()) < 2:
        return {}
        
    X = analysis_df[feature_cols].fillna(0).astype(float)
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced' # Handle imbalance (e.g. few FPs vs many TPs)
    )
    rf.fit(X, y)
    
    # Extract importances
    importances = rf.feature_importances_
    importances_pct = (importances / importances.sum()) * 100
    
    feature_importance = {
        feature: float(imp) 
        for feature, imp in zip(feature_cols, importances_pct)
    }
    
    return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))


def plot_confidence_boxplot(df: pd.DataFrame, cluster_id: int):
    """
    Create a box plot of confidence scores by confusion tag.
    """
    # Filter out Unknowns if any
    plot_df = df[df['confusion_tag'] != 'Unknown']
    
    # Order: TP, FP, FN, TN
    category_order = {'confusion_tag': ['TP', 'FP', 'FN', 'TN']}
    
    fig = px.box(
        plot_df, 
        x='confusion_tag', 
        y='confidence', 
        color='confusion_tag',
        title=f"Confidence Score Distribution by Outcome (Cluster {cluster_id})",
        category_orders=category_order,
        color_discrete_map={
            'TP': 'green',
            'FP': 'red',
            'FN': 'orange',
            'TN': 'gray'
        },
        points="outliers" # Show outliers
    )
    
    fig.update_layout(
        yaxis_title="Confidence Score (1.0 = Centroid)",
        xaxis_title="Outcome Category",
        showlegend=False
    )
    
    return fig

# --- Presentation Logic ---

def compare_cluster_stats(row_before: pd.Series, row_after: pd.Series) -> pd.DataFrame:
    """
    Compare cluster statistics before and after filtering.
    Returns a DataFrame suitable for display.
    """
    metrics_to_show = [
        ('size', 'Total Trades'),
        ('long_precision', 'Long Precision (%)'),
        ('short_precision', 'Short Precision (%)'),
        ('avg_long_pnl', 'Avg Long PnL (%)'),
        ('avg_short_pnl', 'Avg Short PnL (%)'),
    ]
    
    comparison_data = []
    for metric_key, metric_label in metrics_to_show:
        before_val = row_before[metric_key]
        after_val = row_after[metric_key]
        
        # Calculate delta
        if metric_key == 'size':
            delta = after_val - before_val
            delta_pct = (delta / before_val * 100) if before_val > 0 else 0
            delta_str = f"{delta:+.0f} ({delta_pct:+.1f}%)"
        else:
            delta = after_val - before_val
            delta_str = f"{delta:+.2f}"
        
        # Arrow indicator
        if delta > 0:
            arrow = "↑"
            color = "🟢"
        elif delta < 0:
            arrow = "↓"
            color = "🔴"
        else:
            arrow = "→"
            color = "⚪"
        
        comparison_data.append({
            'Metric': metric_label,
            'Before': f"{before_val:.2f}" if metric_key != 'size' else f"{int(before_val)}",
            'After': f"{after_val:.2f}" if metric_key != 'size' else f"{int(after_val)}",
            'Delta': f"{color} {delta_str} {arrow}"
        })
    
    return pd.DataFrame(comparison_data)


def analyze_tag_divergence(
    df_fs: pd.DataFrame, 
    cluster_id: int, 
    feature_cols: list
) -> Dict[str, float]:
    """
    Calculate the divergence (Standardized Mean Difference) of features between TP and FP trades 
    within a specific cluster.
    
    Returns:
        Dict[str, float]: Dictionary mapping feature names to their divergence score.
                          Positive score = Higher in TP.
                          Negative score = Higher in FP.
    """
    # Filter to cluster
    cluster_df = df_fs[df_fs['cluster_id'] == cluster_id].copy()
    
    if 'confusion_tag' not in cluster_df.columns:
        # Should have been added already, but safety check
        cluster_df = add_confusion_tags(cluster_df, cluster_id, direction='long')
        
    # Split into TP and FP
    tp_df = cluster_df[cluster_df['confusion_tag'] == 'TP']
    fp_df = cluster_df[cluster_df['confusion_tag'] == 'FP']
    
    if len(tp_df) < 2 or len(fp_df) < 2:
        return {}
        
    divergence = {}
    
    # Calculate stats for normalization
    cluster_std = cluster_df[feature_cols].std()
    
    for col in feature_cols:
        if col not in cluster_df.columns:
            continue
            
        mean_tp = tp_df[col].mean()
        mean_fp = fp_df[col].mean()
        std = cluster_std[col]
        
        # Avoid division by zero
        if std == 0 or pd.isna(std):
            score = 0
        else:
            # Standardized Mean Difference
            # (Mean TP - Mean FP) / Std Dev
            score = (mean_tp - mean_fp) / std
            
        divergence[col] = score
        
    # Sort by absolute magnitude
    sorted_div = dict(sorted(divergence.items(), key=lambda item: abs(item[1]), reverse=True))
    
    return sorted_div


def suggest_fp_filters(
    df_fs: pd.DataFrame, 
    cluster_id: int, 
    feature_cols: list,
    divergence_scores: Dict[str, float] = None,
    min_divergence: float = -0.5
) -> List[Dict[str, Any]]:
    """
    Suggest filters to reduce False Positives based on negative divergent features.
    Returns a list of suggested filters with impact statistics.
    """
    # Filter to cluster
    cluster_df = df_fs[df_fs['cluster_id'] == cluster_id].copy()
    
    if 'confusion_tag' not in cluster_df.columns:
        cluster_df = add_confusion_tags(cluster_df, cluster_id, direction='long')
        
    # Get Negative Divergent Features (Higher in FP)
    if not divergence_scores:
        divergence_scores = analyze_tag_divergence(cluster_df, cluster_id, feature_cols)
        
    fp_drivers = {k: v for k, v in divergence_scores.items() if v < min_divergence} # Threshold for relevance
    
    suggestions = []
    
    tp_df = cluster_df[cluster_df['confusion_tag'] == 'TP']
    fp_df = cluster_df[cluster_df['confusion_tag'] == 'FP']
    
    total_tp = len(tp_df)
    total_fp = len(fp_df)
    
    if total_fp == 0:
        return []
        
    for feature, div_score in fp_drivers.items():
        # Logic: Since feature is higher in FP, we want to filter out HIGH values.
        # Suggested Threshold: 80th percentile of TP values.
        # Anything above this is "too high" (likely FP).
        
        if feature not in cluster_df.columns:
            continue
            
        # We want to keep the "TP-like" range.
        # Since FP is higher, we set a MAX threshold.
        threshold = tp_df[feature].quantile(0.80)
        
        # Simulate Filter: Keep only rows where feature < threshold
        # Removed rows: feature >= threshold
        
        fp_removed = fp_df[fp_df[feature] >= threshold]
        tp_removed = tp_df[tp_df[feature] >= threshold]
        
        fp_red_count = len(fp_removed)
        tp_cost_count = len(tp_removed)
        
        if fp_red_count == 0:
            continue
            
        new_tp = total_tp - tp_cost_count
        new_fp = total_fp - fp_red_count
        new_total = new_tp + new_fp
        
        if new_total == 0:
            new_precision = 0
        else:
            new_precision = new_tp / new_total
            
        current_precision = total_tp / (total_tp + total_fp)
        
        suggestions.append({
            'feature': feature,
            'threshold': threshold,
            'operator': '<',
            'fp_reduction': fp_red_count,
            'fp_reduction_pct': (fp_red_count / total_fp) * 100,
            'tp_cost': tp_cost_count,
            'tp_cost_pct': (tp_cost_count / total_tp) * 100 if total_tp > 0 else 0,
            'new_precision': new_precision,
            'precision_lift': new_precision - current_precision,
            'divergence': div_score
        })
        
    # Sort by Precision Lift
    suggestions = sorted(suggestions, key=lambda x: x['precision_lift'], reverse=True)
    
    return suggestions
