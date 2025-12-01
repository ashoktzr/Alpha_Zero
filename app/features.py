# --- FILE: app/features.py ---
import pandas as pd
import numpy as np

def extract_features(df, atr_col=None, piv_lookback=50, vol_lookback=20, ma_windows=[20, 50, 200], 
                     stoch_params=[(14, 3)], willr_periods=[14], cci_periods=[20], adx_periods=[14]):
    """
    Extract features from dataframe. Vectorized implementation.
    """
    df = df.copy()
    # Unlimited version: Keep float64 precision for accuracy

    close = df['close']
    high = df['high']
    low = df['low']
    
    if atr_col and atr_col in df.columns:
        atr = df[atr_col].replace(0, np.nan)
    else:
        atr = pd.Series(np.nan, index=df.index)

    # pivots
    pivot_high = high.rolling(piv_lookback, min_periods=1).max()
    pivot_low = low.rolling(piv_lookback, min_periods=1).min()
    df['dist_to_pivot_high'] = (pivot_high - close) / atr
    df['dist_to_pivot_low'] = (close - pivot_low) / atr

    # volume features - removed vol_ma5, vol_spike, vol_z as requested
    # Raw volume is available as a feature instead

    # returns normalized by ATR
    for lag in [1, 3, 5]:
        df[f'ret_{lag}_atr'] = (close - close.shift(lag)) / atr

    # slopes for macd_hist
    # User Request: Remove 2nd and 3rd order derivatives
    # macd_hist_cols = [c for c in df.columns if c.startswith('macd_hist')]
    # for c in macd_hist_cols:
    #    # First derivative (slope)
    #    slope_col = c + '_slope3'
    #    df[slope_col] = (df[c] - df[c].shift(3))/3
        
    # --- Advanced Features (Restricted) ---
    
    # 1. Market Structure & Trend
    # MA Angle (Momentum acceleration)
    ma_main = close.rolling(vol_lookback).mean()
    # Angle in radians (approx)
    df['ma_angle'] = np.arctan((ma_main - ma_main.shift(1)) / (close * 0.001)) # Scaling factor for angle visibility
    
    # Lag-1 Autocorrelation (Proxy for ret_ac)
    ret = close.pct_change()
    df['ret_ac'] = ret.rolling(vol_lookback).corr(ret.shift(1))

    # Log Return
    df['log_return'] = np.log(close / close.shift(1))

    # --- New Indicators (User Request) ---
    
    # 1. Stochastic Oscillator  
    for k_period, d_period in stoch_params:
        low_min = low.rolling(k_period).min()
        high_max = high.rolling(k_period).max()
        
        k_col = f'stoch_k_{k_period}_{d_period}'
        d_col = f'stoch_d_{k_period}_{d_period}'
        df[k_col] = 100 * (close - low_min) / (high_max - low_min + 1e-9)
        df[d_col] = df[k_col].rolling(d_period).mean()
    
    # 2. Williams %R
    for period in willr_periods:
        low_min_w = low.rolling(period).min()
        high_max_w = high.rolling(period).max()
        df[f'willr_{period}'] = -100 * (high_max_w - close) / (high_max_w - low_min_w + 1e-9)
    
    # 3. CCI (Commodity Channel Index)
    for period in cci_periods:
        tp = (high + low + close) / 3
        tp_sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df[f'cci_{period}'] = (tp - tp_sma) / (0.015 * mad + 1e-9)
    
    # 4. ADX (Average Directional Index)
    for period in adx_periods:
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed TR and DM (Wilder's Smoothing)
        tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr_smooth)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        df[f'adx_{period}'] = dx.ewm(alpha=1/period, adjust=False).mean()

    # keep only relevant features for modeling
    # Dynamically select features
    base_features = [
        'dist_to_pivot_high', 'dist_to_pivot_low', 
        'log_return', 'ma_angle', 'ret_ac',
        'volume'
    ]
    
    # Add MA distance features
    for w in ma_windows:
        base_features.append(f'dist_ma{w}')
    
    # Add calculated indicator columns dynamically
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in ['stoch_k_', 'stoch_d_', 'willr_', 'cci_', 'adx_']):
            base_features.append(col)
    
    # Filter to existing columns
    feature_cols = [c for c in base_features if c in df.columns]
    
    return df, feature_cols
