# app/indicators.py
"""
Compute technical indicators for pipeline.

Functions:
  compute_indicators(df, cfg) -> (df_out, feature_cols)

cfg is a dict, possible keys:
  - 'rsi': list of windows (e.g. [14])
  - 'macd': list of triples (fast, slow, signal) as tuples
  - 'bb': list of tuples (n, k) e.g. (20, 2)
  - 'atr': list of windows (e.g. [14])
  - 'parabolic_sar': not implemented (placeholder)
  - volume flags handled automatically
Returns:
  df_out: original df plus indicator columns
  feature_cols: list of generated feature column names
"""
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

def _rsi(series: pd.Series, window: int):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, min_periods=window).mean()
    ma_down = down.ewm(alpha=1/window, min_periods=window).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def _macd(series: pd.Series, fast: int, slow: int, signal: int):
    fast_ema = _ema(series, fast)
    slow_ema = _ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def _bb(series: pd.Series, n: int, k: float):
    ma = series.rolling(n, min_periods=1).mean()
    std = series.rolling(n, min_periods=1).std().fillna(0)
    upper = ma + k * std
    lower = ma - k * std
    pct_b = (series - lower) / (upper - lower + 1e-12)
    width = (upper - lower) / (ma.abs() + 1e-12)
    return upper, lower, pct_b, width

def _atr(df: pd.DataFrame, n: int):
    # True range
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=1).mean()
    return atr



def compute_indicators(df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute indicators, append to df and return list of feature column names.
    """
    df = df.copy()
    feature_cols = []

    # ensure numeric types for OHLCV
    for c in ['open','high','low','close','volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Volume - keep raw volume column, remove derived features
    # Raw volume is available as a feature in features.py

    # RSI
    for w in (cfg.get('rsi') or []):
        name = f"rsi_{w}"
        df[name] = _rsi(df['close'], w)
        feature_cols.append(name)
        # slope
        df[f"{name}_slope3"] = df[name].diff().rolling(3, min_periods=1).mean()
        feature_cols.append(f"{name}_slope3")

    # MACD variants
    for triple in (cfg.get('macd') or []):
        a,b,c = triple
        macd_line, signal_line, hist = _macd(df['close'], a, b, c)
        base = f"macd_hist_{a}_{b}_{c}"
        df[f"{base}"] = hist
        feature_cols.append(f"{base}")
        # slope
        df[f"{base}_slope3"] = hist.diff().rolling(3, min_periods=1).mean()
        feature_cols.append(f"{base}_slope3")

    # Bollinger Bands
    for n,k in (cfg.get('bb') or []):
        upper, lower, pct_b, width = _bb(df['close'], n, k)
        base = f"bb_{n}_{k}"
        df[f"{base}_pctb"] = pct_b
        df[f"{base}_width"] = width
        feature_cols += [f"{base}_pctb", f"{base}_width"]

    # ATR
    for n in (cfg.get('atr') or []):
        name = f"atr_{n}"
        df[name] = _atr(df, n)
        # normalize by price to make version scale-free
        df[f"{name}_rel"] = df[name] / (df['close'].rolling(n, min_periods=1).mean().replace(0,1))
        feature_cols += [name, f"{name}_rel"]

    # Fill NaNs for features with zeros (safer for clustering)
    for c in feature_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(float)

    return df, feature_cols
