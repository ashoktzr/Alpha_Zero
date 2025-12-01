# app/pages/inspection_page.py
"""Trade inspection and visualization page"""
import streamlit as st
import pandas as pd
from app.visualization import plot_trade_snap


def render(tp_pct, sl_pct):
    """Render the trade inspection page"""
    st.subheader("Trade Inspection")
    
    st.info("⚙️ **Verification Tool**: This page is designed for efficient verification of cluster trade outcomes. Use the filters and navigation to inspect individual trades.")
    
    if st.session_state['pipeline'].df_fs is None:
        st.info("Run the pipeline first to inspect trades.")
        return
    
    df_fs = st.session_state['pipeline'].df_fs
    
    # === SELECTION AT TOP ===
    st.write("### Trade Selection")
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    with col1:
        # Filter by Cluster
        clusters = sorted(df_fs['cluster_id'].unique())
        sel_cluster = st.selectbox("Cluster", clusters, key='cluster_selector')
        
    with col2:
        # Filter by Direction
        sel_direction = st.selectbox("Direction", ["Long", "Short"], key='dir_selector')
        
    with col3:
        # Filter by Result
        sel_result = st.selectbox("Result", ["All", "TP", "SL", "No Action"], key='res_selector')
    
    with col4:
        # Filter trades based on selection
        mask = (df_fs['cluster_id'] == sel_cluster)
        
        if sel_direction == "Long":
            if sel_result == "TP":
                mask &= (df_fs['long_exit'] == 'TP')
            elif sel_result == "SL":
                mask &= (df_fs['long_exit'] == 'SL')
            elif sel_result == "No Action":
                mask &= (df_fs['long_exit'] == 'NEITHER')
        else: # Short
            if sel_result == "TP":
                mask &= (df_fs['short_exit'] == 'TP')
            elif sel_result == "SL":
                mask &= (df_fs['short_exit'] == 'SL')
            elif sel_result == "No Action":
                mask &= (df_fs['short_exit'] == 'NEITHER')
                
        cluster_trades = df_fs[mask]
        
        if cluster_trades.empty:
            st.warning("No trades match criteria.")
            sel_trade_idx = None
            trade_opts = []
        else:
            trade_opts = cluster_trades.index.astype(str).tolist()
            
            # Selectbox for navigation
            sel_trade_ts = st.selectbox(
                "Trade (Timestamp)", 
                trade_opts,
                key='trade_selector'
            )
            
            # Next button
            current_idx = trade_opts.index(sel_trade_ts)
            if st.button("→ Next Trade") and current_idx < len(trade_opts) - 1:
                next_trade = trade_opts[current_idx + 1]
                st.session_state['trade_selector'] = next_trade
                st.rerun()
                
            sel_trade_idx = pd.Timestamp(sel_trade_ts)
    
    # === CHARTS BELOW ===
    if sel_trade_idx is not None and len(trade_opts) > 0:
        st.markdown("---")
        trade_row = df_fs.loc[sel_trade_idx]
        
        # Trade info
        trade_num = trade_opts.index(sel_trade_ts) + 1
        st.caption(f"📊 Trade {trade_num} of {len(trade_opts)} in Cluster {sel_cluster}")
        
        # Auto-generate visualization without features
        try:
            charts = plot_trade_snap(df_fs, trade_row, features=[], tp_pct=tp_pct, sl_pct=sl_pct)
            
            if 'error' in charts:
                st.error(charts['error'])
            else:
                # Show both Long and Short side by side
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(charts['long'], use_container_width=True)
                with c2:
                    st.plotly_chart(charts['short'], use_container_width=True)
                
        except Exception as e:
            st.error(f"Visualization Error: {e}")
            st.exception(e)
