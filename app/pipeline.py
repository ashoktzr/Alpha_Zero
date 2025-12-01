# app/pipeline.py
import pandas as pd
import os
from datetime import datetime
import json
from typing import List, Optional, Dict, Any, Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from .config_types import IndicatorConfig, ScanConfig, ClusterConfig, PipelineResult
from .indicators import compute_indicators
from .features import extract_features
from .cluster_analysis import run_clustering, compute_cluster_detailed_stats
from .forward_scan import forward_scan
from .io_utils import ensure_dir

class Pipeline:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.df_ind: Optional[pd.DataFrame] = None
        self.df_scaled: Optional[pd.DataFrame] = None # Stores only scaled features for clustering
        self.df_fs: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = [] # All available features
        self.selected_features: List[str] = [] # Features selected for clustering
        self.labels: Optional[pd.Series] = None
        self.cluster_stats: Optional[pd.DataFrame] = None
        self.result: Optional[PipelineResult] = None
        # Clustering configuration tracking
        self.clustering_method: str = "Unknown"
        self.scaling_method: str = "Unknown"
        self.num_features: int = 0

    def load_data(self, df: pd.DataFrame):
        """Load initial dataframe."""
        self.df = df.copy()
        # Reset downstream states
        self.df_ind = None
        self.df_scaled = None
        self.df_fs = None
        self.feature_cols = []
        self.selected_features = []
        self.labels = None
        self.cluster_stats = None
        self.result = None

    def run_indicators(self, config: IndicatorConfig, progress_cb: Optional[Callable] = None):
        """Step 1: Compute indicators."""
        if self.df is None:
            raise ValueError("No data loaded")
        
        if progress_cb: progress_cb({'event': 'start', 'stage': 'indicators', 'msg': 'Computing indicators...'})
        
        # Convert Pydantic config to dict expected by legacy indicators.py (or refactor indicators.py later)
        # For now, adapter logic:
        cfg_dict = {
            'rsi': config.rsi,
            'macd': config.macd,
            'bb': config.bb,
            'atr': config.atr
        }
        
        self.df_ind, generated_feats = compute_indicators(self.df, cfg_dict)
        
        # Default feature selection logic (can be overridden)
        if generated_feats:
            self.feature_cols = generated_feats
        else:
             # Fallback
             self.feature_cols = [c for c in self.df_ind.columns if pd.api.types.is_numeric_dtype(self.df_ind[c]) and c not in ['timestamp']]

        if progress_cb: progress_cb({'event': 'done', 'stage': 'indicators', 'msg': 'Indicators computed'})

    def run_feature_extraction(self, progress_cb: Optional[Callable] = None, **kwargs):
        """Step 1.5: Extract additional features (vectorized)."""
        if self.df_ind is None:
            raise ValueError("Indicators not computed")
            
        if progress_cb: progress_cb({'event': 'start', 'stage': 'features', 'msg': 'Extracting features...'})
        
        # Use the new vectorized features module
        # Assuming 'atr_14' exists if ATR=14 was used, else pick first atr
        atr_col = next((c for c in self.df_ind.columns if c.startswith('atr_')), None)
        
        self.df_ind, extra_feats = extract_features(self.df_ind, atr_col=atr_col, **kwargs)
        
        # Add new features to the list if not present
        for f in extra_feats:
            if f not in self.feature_cols:
                self.feature_cols.append(f)
                
        if progress_cb: progress_cb({'event': 'done', 'stage': 'features', 'msg': 'Features extracted'})

    def run_clustering_only(self, cluster_cfg: ClusterConfig, selected_features: List[str], scaling_method: str = None, rolling_window: int = 100, progress_cb: Optional[Callable] = None):
        """Step 2a: Clustering Only."""
        if self.df_ind is None:
            raise ValueError("Indicators not computed")
        
        self.selected_features = selected_features
        
        # Store clustering configuration
        self.clustering_method = cluster_cfg.method
        self.scaling_method = scaling_method if scaling_method else "None"
        self.num_features = len(selected_features)
        
        # Create a subset for clustering (Memory Optimization)
        # We only copy the selected features
        X = self.df_ind[self.selected_features].copy()
        X = X.fillna(0) # Handle NaNs before scaling

        # Scaling
        if scaling_method == "Standard":
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        elif scaling_method == "MinMax":
            scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        elif scaling_method == "Robust":
            scaler = RobustScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        elif scaling_method == "Rolling Z-Score":
            # Manual Rolling Z-Score
            roll_mean = X.rolling(window=rolling_window, min_periods=1).mean()
            roll_std = X.rolling(window=rolling_window, min_periods=1).std().replace(0, 1e-9)
            X = (X - roll_mean) / roll_std
            X = X.fillna(0) # Handle initial NaNs or division issues
            
        # Store scaled features for visualization/debugging
        self.df_scaled = X
        
        # Clustering
        if progress_cb: progress_cb({'event': 'start', 'stage': 'clustering', 'msg': 'Running clustering...'})
        
        params = {
            'min_cluster_size': cluster_cfg.min_cluster_size,
            'n_clusters': cluster_cfg.n_clusters,
            'n_jobs': cluster_cfg.n_jobs,
            'n_init': 'auto'
        }
        
        self.labels, _ = run_clustering(self.df_scaled, method=cluster_cfg.method, params=params)
        
        # Assign labels back to main DF (lightweight operation)
        self.df_ind['cluster_id'] = self.labels.reindex(self.df_ind.index).fillna(-1).astype(int)
        
        if progress_cb: progress_cb({'event': 'done', 'stage': 'clustering', 'msg': 'Clustering complete'})

    def run_forward_scan_only(self, scan_cfg: ScanConfig, n_jobs: int = 1, progress_cb: Optional[Callable] = None):
        """Step 2b: Forward Scan Only (requires clustering to be done)."""
        if self.df_ind is None:
            raise ValueError("Indicators not computed")
        if 'cluster_id' not in self.df_ind.columns:
            raise ValueError("Clustering not run yet")

        # Forward Scan
        if progress_cb: progress_cb({'event': 'start', 'stage': 'forward_scan', 'msg': 'Running forward scan...'})
        
        self.df_fs = forward_scan(
            self.df_ind, 
            tp_pct=scan_cfg.tp_pct, 
            sl_pct=scan_cfg.sl_pct, 
            max_lookahead=scan_cfg.max_lookahead,
            n_jobs=n_jobs, 
            progress_cb=progress_cb
        )
        
        if progress_cb: progress_cb({'event': 'done', 'stage': 'forward_scan', 'msg': 'Forward scan complete'})

        # Stats
        try:
            self.cluster_stats = compute_cluster_detailed_stats(self.df_fs, dur_cap=scan_cfg.max_lookahead)
        except Exception as e:
            if progress_cb: progress_cb({'event': 'info', 'stage': 'stats', 'msg': f'Stats failed: {e}'})
            self.cluster_stats = pd.DataFrame()

        self.result = PipelineResult(
            n_rows=len(self.df_fs),
            n_features=len(self.selected_features),
            cluster_stats=self.cluster_stats.to_dict(orient='records')
        )

    def save_run(self, name_prefix: str, meta: Dict[str, Any]) -> str:
        """Save run results to disk."""
        if self.df_fs is None:
            raise ValueError("No results to save")
            
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        run_folder = os.path.join("outputs", "runs", f"{name_prefix}_{timestamp}")
        ensure_dir(run_folder)
        
        # Save CSV and Parquet
        self.df_fs.reset_index().to_csv(os.path.join(run_folder, "annotated.csv"), index=False)
        self.df_fs.to_parquet(os.path.join(run_folder, "annotated.parquet"))
        
        # Save Meta
        with open(os.path.join(run_folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
            
        return run_folder
