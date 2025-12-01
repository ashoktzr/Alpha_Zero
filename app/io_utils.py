# app/io_utils.py
import pandas as pd
import io
import os
import json

def infer_time_col(cols):
    candidates = []
    lower = [c.lower() for c in cols]
    for name in ['timestamp','time','date','open_time','opentime','start_time','datetime']:
        if name.lower() in lower:
            candidates.append(cols[lower.index(name.lower())])
    return candidates

def parse_timestamp(ts_col):
    """
    Robustly parse timestamp column.
    Handles: nanoseconds, milliseconds, seconds, ISO strings, and datetime objects
    Returns: datetime64[ns] pandas series
    """
    # Already datetime
    if pd.api.types.is_datetime64_any_dtype(ts_col):
        return pd.to_datetime(ts_col)
    
    # Try numeric first
    if pd.api.types.is_numeric_dtype(ts_col):
        # Check magnitude to determine unit
        sample_val = ts_col.iloc[0] if len(ts_col) > 0 else 0
        
        # Nanoseconds: > 1e15 (year 2001+ in nanoseconds)
        if sample_val > 1e15:
            return pd.to_datetime(ts_col, unit='ns', utc=True).dt.tz_localize(None)
        
        # Milliseconds: between 1e12 and 1e15 (year 2001+ in milliseconds)
        elif sample_val > 1e12:
            return pd.to_datetime(ts_col, unit='ms', utc=True).dt.tz_localize(None)
        
        # Seconds: between 1e9 and 1e12 (year 2001+ in seconds)
        elif sample_val > 1e9:
            return pd.to_datetime(ts_col, unit='s', utc=True).dt.tz_localize(None)
        
        # Assume milliseconds for smaller values (legacy data)
        else:
            return pd.to_datetime(ts_col, unit='ms', utc=True).dt.tz_localize(None)
    
    # Try parsing as string
    try:
        return pd.to_datetime(ts_col, utc=True).dt.tz_localize(None)
    except Exception as e:
        raise ValueError(f"Could not parse timestamp column. Error: {e}")

def smart_read_csv(use_file):
    """
    Robust CSV reader. Returns DataFrame with DatetimeIndex named 'timestamp'.
    Accepts file path string or file-like object (streamlit UploadedFile).
    Handles nanosecond, millisecond, and second timestamps automatically.
    """
    if hasattr(use_file, "read"):
        raw = use_file.read()
        sample = pd.read_csv(io.BytesIO(raw), nrows=5)
    else:
        sample = pd.read_csv(use_file, nrows=5)
    
    # Find timestamp column
    cols = list(sample.columns)
    candidates = infer_time_col(cols)
    dt_col = candidates[0] if candidates else cols[0]
    
    # Read full dataframe
    if hasattr(use_file, "read"):
        df_full = pd.read_csv(io.BytesIO(raw))
    else:
        df_full = pd.read_csv(use_file)
    
    # Parse timestamp using robust function
    df_full[dt_col] = parse_timestamp(df_full[dt_col])
    
    # Validate
    if df_full[dt_col].isna().mean() > 0.5:
        raise ValueError(f"More than 50% values in chosen time column '{dt_col}' could not be parsed as datetimes.")
    
    # Set as index
    df_full.set_index(pd.DatetimeIndex(df_full[dt_col]), inplace=True)
    df_full.index.name = 'timestamp'
    
    # Memory Optimization: Downcast float64 to float32
    fcols = df_full.select_dtypes('float').columns
    if len(fcols) > 0:
        df_full[fcols] = df_full[fcols].astype('float32')
        
    return df_full

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_profile(profile_dict, name='last'):
    ensure_dir('configs/profiles')
    path = f'configs/profiles/{name}.json'
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(profile_dict, fh, indent=2)
    return path

def load_profile(name='last'):
    path = f'configs/profiles/{name}.json'
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)
