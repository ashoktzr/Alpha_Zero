# app/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any

class IndicatorConfig(BaseModel):
    rsi: List[int] = Field(default_factory=lambda: [14])
    macd: List[Tuple[int, int, int]] = Field(default_factory=lambda: [(12, 26, 9)])
    bb: List[Tuple[int, float]] = Field(default_factory=lambda: [(20, 2.0)])
    atr: List[int] = Field(default_factory=lambda: [14])

class ScanConfig(BaseModel):
    tp_pct: float = 0.01
    sl_pct: float = 0.005
    max_lookahead: int = 400

class ClusterConfig(BaseModel):
    method: str = "hdbscan"
    n_clusters: int = 8
    min_cluster_size: int = 25
    n_jobs: int = 1
    
class FeatureConfig(BaseModel):
    piv_lookback: int = 50
    vol_lookback: int = 20
    ma_windows: List[int] = Field(default_factory=lambda: [20, 50, 200])

class UserProfile(BaseModel):
    name: str
    indicator_config: IndicatorConfig
    scan_config: ScanConfig
    cluster_config: ClusterConfig
    feature_config: FeatureConfig = Field(default_factory=FeatureConfig)
    feature_cols: List[str] = Field(default_factory=list)

class PipelineResult(BaseModel):
    n_rows: int
    n_features: int
    cluster_stats: List[Dict[str, Any]] = Field(default_factory=list)
