# app/visualization.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_trade_snap(df: pd.DataFrame, trade_row: pd.Series, features: list = None, buffer_candles: int = 20, tp_pct: float = 0.01, sl_pct: float = 0.005):
    """
    Generate static snapshots for BOTH Long and Short trades.
    Always shows both perspectives regardless of which actually hit TP/SL.
    Returns: dict with keys 'long' and 'short', each containing a Figure
    """
    entry_idx = trade_row.name
    
    try:
        loc = df.index.get_loc(entry_idx)
    except KeyError:
        return {'long': None, 'short': None, 'error': "Trade index not found"}

    result = {'long': None, 'short': None}
    entry_price = trade_row['close']
    
    # ===== LONG CHART =====
    long_duration = int(trade_row['long_duration']) if not pd.isna(trade_row['long_duration']) else 0
    long_tp_price = entry_price * (1 + tp_pct)
    long_sl_price = entry_price * (1 - sl_pct)
    
    start_pos = max(0, loc - buffer_candles)
    end_pos = min(len(df), loc + long_duration + buffer_candles)
    subset = df.iloc[start_pos:end_pos].copy()
    
    rows = 2 if features else 1
    row_heights = [0.7, 0.3] if features else [1.0]
    
    fig_long = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights)
    
    fig_long.add_trace(go.Candlestick(
        x=subset.index,
        open=subset['open'], high=subset['high'],
        low=subset['low'], close=subset['close'],
        name='Price'
    ), row=1, col=1)
    
    fig_long.add_trace(go.Scatter(
        x=[trade_row.name], y=[entry_price],
        mode='markers', marker=dict(color='blue', size=12, symbol='triangle-right'),
        name='Entry'
    ), row=1, col=1)
    
    # Exit marker removed as per user request
    # if long_duration > 0:
    #     exit_idx = df.index[loc + long_duration] if loc + long_duration < len(df) else subset.index[-1]
    #     ...
    
    line_end = subset.index[-1]
    
    # TP/SL levels - use actual exit prices if available
    if 'long_exit_price' in trade_row and not pd.isna(trade_row['long_exit_price']):
        # Use actual exit price for more accurate visualization
        if trade_row['long_exit'] == 'TP':
            actual_long_tp = trade_row['long_exit_price']
            long_tp_price = actual_long_tp
        elif trade_row['long_exit'] == 'SL':
            actual_long_sl = trade_row['long_exit_price']
            long_sl_price = actual_long_sl
    
    fig_long.add_shape(type="line",
        x0=trade_row.name, y0=long_tp_price,
        x1=line_end, y1=long_tp_price,
        line=dict(color="Green", width=2, dash="dash"),
        row=1, col=1
    )
    fig_long.add_annotation(x=line_end, y=long_tp_price, text="TP", showarrow=False, xanchor="right", yanchor="bottom", font=dict(color="Green", size=12))

    fig_long.add_shape(type="line",
        x0=trade_row.name, y0=long_sl_price,
        x1=line_end, y1=long_sl_price,
        line=dict(color="Red", width=2, dash="dash"),
        row=1, col=1
    )
    fig_long.add_annotation(x=line_end, y=long_sl_price, text="SL", showarrow=False, xanchor="right", yanchor="top", font=dict(color="Red", size=12))
    
    if features:
        for f in features:
            if f in subset.columns:
                fig_long.add_trace(go.Scatter(
                    x=subset.index, y=subset[f],
                    mode='lines', name=f, line=dict(width=2)
                ), row=2, col=1)
    
    exit_status = trade_row['long_exit'] if trade_row['long_exit'] != 'NEITHER' else 'No Exit'
    
    # Update layout with proper axis labels
    fig_long.update_layout(
        title=f"📈 LONG: {exit_status} (Dur: {long_duration} bars)",
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=True
    )
    
    # Label the axes
    fig_long.update_yaxes(title_text="Price", row=1, col=1)
    if features:
        feature_label = ", ".join(features) if len(features) > 1 else features[0]
        fig_long.update_yaxes(title_text=feature_label, row=2, col=1)
        fig_long.update_xaxes(title_text="Time", row=2, col=1)
    
    result['long'] = fig_long
    
    # ===== SHORT CHART =====
    short_duration = int(trade_row['short_duration']) if not pd.isna(trade_row['short_duration']) else 0
    short_tp_price = entry_price * (1 - tp_pct)
    short_sl_price = entry_price * (1 + sl_pct)
    
    start_pos = max(0, loc - buffer_candles)
    end_pos = min(len(df), loc + short_duration + buffer_candles)
    subset = df.iloc[start_pos:end_pos].copy()
    
    fig_short = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights)
    
    fig_short.add_trace(go.Candlestick(
        x=subset.index,
        open=subset['open'], high=subset['high'],
        low=subset['low'], close=subset['close'],
        name='Price'
    ), row=1, col=1)
    
    fig_short.add_trace(go.Scatter(
        x=[trade_row.name], y=[entry_price],
        mode='markers', marker=dict(color='orange', size=12, symbol='triangle-right'),
        name='Entry'
    ), row=1, col=1)
    
    # Exit marker removed as per user request
    # if short_duration > 0:
    #     exit_idx = df.index[loc + short_duration] if loc + short_duration < len(df) else subset.index[-1]
    #     ...
    
    line_end = subset.index[-1]
    
    # TP/SL levels - use actual exit prices if available
    if 'short_exit_price' in trade_row and not pd.isna(trade_row['short_exit_price']):
        # Use actual exit price for more accurate visualization
        if trade_row['short_exit'] == 'TP':
            actual_short_tp = trade_row['short_exit_price']
            short_tp_price = actual_short_tp
        elif trade_row['short_exit'] == 'SL':
            actual_short_sl = trade_row['short_exit_price']
            short_sl_price = actual_short_sl
    
    fig_short.add_shape(type="line",
        x0=trade_row.name, y0=short_tp_price,
        x1=line_end, y1=short_tp_price,
        line=dict(color="Green", width=2, dash="dash"),
        row=1, col=1
    )
    fig_short.add_annotation(x=line_end, y=short_tp_price, text="TP", showarrow=False, xanchor="right", yanchor="bottom", font=dict(color="Green", size=12))

    fig_short.add_shape(type="line",
        x0=trade_row.name, y0=short_sl_price,
        x1=line_end, y1=short_sl_price,
        line=dict(color="Red", width=2, dash="dash"),
        row=1, col=1
    )
    fig_short.add_annotation(x=line_end, y=short_sl_price, text="SL", showarrow=False, xanchor="right", yanchor="top", font=dict(color="Red", size=12))
    
    if features:
        for f in features:
            if f in subset.columns:
                fig_short.add_trace(go.Scatter(
                    x=subset.index, y=subset[f],
                    mode='lines', name=f, line=dict(width=2)
                ), row=2, col=1)
    
    exit_status = trade_row['short_exit'] if trade_row['short_exit'] != 'NEITHER' else 'No Exit'
    
    # Update layout with proper axis labels
    fig_short.update_layout(
        title=f"📉 SHORT: {exit_status} (Dur: {short_duration} bars)",
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=True
    )
    
    # Label the axes
    fig_short.update_yaxes(title_text="Price", row=1, col=1)
    if features:
        feature_label = ", ".join(features) if len(features) > 1 else features[0]
        fig_short.update_yaxes(title_text=feature_label, row=2, col=1)
        fig_short.update_xaxes(title_text="Time", row=2, col=1)
    
    result['short'] = fig_short
    
    return result
