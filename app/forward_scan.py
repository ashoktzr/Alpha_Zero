# app/forward_scan.py
from typing import Optional, Callable, Dict
import pandas as pd
import numpy as np
from numba import jit, prange
import math

@jit(nopython=True, parallel=True)
def _scan_numba(closes, highs, lows, tp_pct, sl_pct, limit_distance):
    n = len(closes)
    # arrays to fill
    # 0=NEITHER, 1=TP, 2=SL
    long_exit_code = np.zeros(n, dtype=np.int8) 
    short_exit_code = np.zeros(n, dtype=np.int8)
    long_duration = np.full(n, np.nan)
    short_duration = np.full(n, np.nan)
    long_exit_price = np.full(n, np.nan)
    short_exit_price = np.full(n, np.nan)
    combo_code = np.zeros(n, dtype=np.int8)
    exit_price = np.full(n, np.nan)

    for i in prange(n):
        entry = closes[i]
        if np.isnan(entry):
            continue
            
        limit = min(n - 1, i + limit_distance)
        
        long_tp_price = entry * (1 + tp_pct)
        long_sl_price = entry * (1 - sl_pct)
        short_tp_price = entry * (1 - tp_pct)
        short_sl_price = entry * (1 + sl_pct)
        
        long_tp_idx = -1
        long_sl_idx = -1
        short_tp_idx = -1
        short_sl_idx = -1
        
        # Scan forward
        # Start from i + 1 to include the immediate next candle
        for j in range(i + 1, limit + 1):
            # Use High/Low for checking hits
            h = highs[j]
            l = lows[j]
            
            if np.isnan(h) or np.isnan(l):
                continue
                
            if long_tp_idx == -1 and h >= long_tp_price:
                long_tp_idx = j
            if long_sl_idx == -1 and l <= long_sl_price:
                long_sl_idx = j
            if short_tp_idx == -1 and l <= short_tp_price:
                short_tp_idx = j
            if short_sl_idx == -1 and h >= short_sl_price:
                short_sl_idx = j
                
            # Optimization: if all found, break early
            if long_tp_idx != -1 and long_sl_idx != -1 and short_tp_idx != -1 and short_sl_idx != -1:
                break
        
        # Determine Long Outcome
        if long_tp_idx != -1 and (long_sl_idx == -1 or long_tp_idx <= long_sl_idx):
            long_exit_code[i] = 1 # TP
            long_duration[i] = long_tp_idx - i
            long_exit_price[i] = long_tp_price
        elif long_sl_idx != -1:
            long_exit_code[i] = 2 # SL
            long_duration[i] = long_sl_idx - i
            long_exit_price[i] = long_sl_price
        else:
            long_exit_code[i] = 0 # NEITHER
            long_exit_price[i] = closes[limit] # Exit at end if neither
            
        # Determine Short Outcome
        if short_tp_idx != -1 and (short_sl_idx == -1 or short_tp_idx <= short_sl_idx):
            short_exit_code[i] = 1 # TP
            short_duration[i] = short_tp_idx - i
            short_exit_price[i] = short_tp_price
        elif short_sl_idx != -1:
            short_exit_code[i] = 2 # SL
            short_duration[i] = short_sl_idx - i
            short_exit_price[i] = short_sl_price
        else:
            short_exit_code[i] = 0 # NEITHER
            short_exit_price[i] = closes[limit] # Exit at end if neither
            
        # Combo Logic
        # Only consider TPs that actually happened (TP before SL)
        valid_lt = long_tp_idx if (long_exit_code[i] == 1) else 1e9
        valid_st = short_tp_idx if (short_exit_code[i] == 1) else 1e9
        
        if valid_lt >= 1e9 and valid_st >= 1e9:
            combo_code[i] = 0
            exit_price[i] = closes[limit]
        elif valid_lt < valid_st:
            combo_code[i] = 1 # Long Wins
            exit_price[i] = long_tp_price
        elif valid_st < valid_lt:
            combo_code[i] = 2 # Short Wins
            exit_price[i] = short_tp_price
        else:
            # Equal time? Prioritize Long (1) as requested
            combo_code[i] = 1
            exit_price[i] = long_tp_price

    return long_exit_code, long_duration, long_exit_price, short_exit_code, short_duration, short_exit_price, combo_code, exit_price

def forward_scan(df: pd.DataFrame, tp_pct: float=0.01, sl_pct: float=0.005, max_lookahead: Optional[int]=None,
                 n_jobs: int = 1, progress_cb: Optional[Callable]=None) -> pd.DataFrame:
    """
    Perform forward scan using Numba for high performance.
    """
    df = df.copy()
    n = len(df)
    closes = df['close'].astype(float).values
    highs = df['high'].astype(float).values
    lows = df['low'].astype(float).values
    
    if max_lookahead is None or int(max_lookahead) == 0:
        limit_distance = n - 1
    else:
        limit_distance = int(max_lookahead)

    if progress_cb:
        progress_cb({'event':'start', 'stage':'forward_scan', 'msg':'starting forward scan (numba)'})

    try:
        # Run Numba optimized scan
        # First run might trigger compilation overhead
        long_exit_code, long_duration, long_exit_price, short_exit_code, short_duration, short_exit_price, combo_code, exit_price = \
            _scan_numba(closes, highs, lows, float(tp_pct), float(sl_pct), int(limit_distance))
            
        # Map codes back to strings
        code_map = {0: 'NEITHER', 1: 'TP', 2: 'SL'}
        
        df['long_exit'] = [code_map[c] for c in long_exit_code]
        df['short_exit'] = [code_map[c] for c in short_exit_code]
        df['long_duration'] = pd.Series(long_duration, index=df.index).astype('Float64')
        df['short_duration'] = pd.Series(short_duration, index=df.index).astype('Float64')
        df['long_exit_price'] = pd.Series(long_exit_price, index=df.index)
        df['short_exit_price'] = pd.Series(short_exit_price, index=df.index)
        df['combo_code'] = combo_code
        df['exit_price'] = pd.Series(exit_price, index=df.index)
        
    except Exception as e:
        if progress_cb:
             progress_cb({'event':'info', 'stage':'forward_scan', 'msg':f'Error in forward_scan: {e}'})
        raise e

    if progress_cb:
        progress_cb({'event':'done', 'stage':'forward_scan', 'msg':'forward scan complete'})

    return df
