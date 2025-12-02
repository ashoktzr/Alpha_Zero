# app/pages/pipeline_page.py
"""Pipeline page for data loading, feature selection, and clustering"""
import streamlit as st
import pandas as pd
import os
from sklearn.decomposition import PCA
import plotly.express as px

from app.config_types import IndicatorConfig, ScanConfig, ClusterConfig
from app.io_utils import smart_read_csv
from app.profiles import load_profile, save_profile


def render(config, progress_cb):
    """Render the pipeline page"""
    
    # Challenge Message
    with st.expander("📖 About Alphazero: The Indicator Challenge", expanded=False):
        st.markdown("""
        ### The Challenge
        
        **Can technical indicators predict price movements?** This tool lets you test that hypothesis rigorously.
        
        #### The Process:
        1. **Data**: Download or upload OHLCV price data
        2. **Indicators**: Compute RSI, MACD, Bollinger Bands, ATR, and more
        3. **Feature Engineering**: Extract market structure features (pivots, MA angles, autocorrelation, etc.)
        4. **Clustering**: Group similar market conditions using K-Means or HDBSCAN algorithms (machine learning clustering methods)
           - **PCA Visualization**: Applied only for 2D visualization, not for dimensionality reduction
        5. **Forward Scan**: Simulate trades for **both Long and Short directions simultaneously** from each cluster with your TP/SL settings
        6. **Results**: Evaluate precision (win rate) for each cluster
        
        #### The Feature Space:
        You can select up to **4 features** from:
        - **Price/Structure**: log returns, autocorrelation, distance to pivots, MA angles, volume
        - **Indicators**: RSI, MACD, Bollinger Bands, ATR, Stochastic, Williams %R, CCI, ADX
        
        Each combination creates a unique "feature space" where similar market conditions **will cluster together** based on the algorithm.
        
        #### Understanding Clusters vs Forward Scan:
        - **Clusters are independent** of TP/SL settings or the forward scan mechanism
        - **Clustering** identifies similar market conditions based solely on feature similarity
        - **Forward Scan** verifies whether TP/SL hit points are spatially closer within a cluster, answering: *"Can this cluster be used for rule-based trading?"*
        - The goal is to find if a cluster reliably leads to one outcome (TP) more than the other (SL) in any particular direction long or short.
    
        
        #### The Holy Grail:
        **Any feature/indicator combination that consistently separates profitable trades from losing ones is pure gold.**
        
        #### Your Mission:
        Find a combination of features where **Precision > 50%** for any direction (Long or Short) with **reward/risk (TP/SL) > 1** (e.g., TP=1%, SL=0.5%).
        
        ⚠️ **Spoiler**: With TP/SL ratio above 1.00 (e.g., TP=1%, SL=0.5%), achieving >50% precision in a cluster with **coverage ≥ 3%** on a dataset of **at least 25K-50K rows** is virtually impossible due to random market behavior. *(Note: This is very modest; real historical tests are done on millions of rows)*
        
        If you find a robust combination, you might be onto something valuable! *(You still have to account for slippage and other costs)*
        
        💡 **Note**: Technical indicators are inherently **backward-looking**. They describe what *has happened*, not what *will happen*. This tool helps you test if historical patterns actually predict future outcomes.
        
        #### Limitations:
        - **Streamlit constraints**: Higher number of feature combinations and larger datasets are limited on this platform
        - Visit the [GitHub repository](https://github.com/ashoktzr/Alpha_Zero) to access the code without limitations and contribute if you have ideas to take this further!
        """)
    
    # Feedback Form
    with st.expander("💬 Submit Feedback", expanded=False):
        st.markdown("**Help us improve Alphazero!** Share your thoughts, bugs, or feature requests.")
        
        feedback_name = st.text_input("Name (Optional)", key="fb_name")
        feedback_email = st.text_input("Email (Optional)", key="fb_email")
        feedback_type = st.selectbox("Feedback Type", ["Bug Report", "Feature Request", "General Feedback", "Results Found!"], key="fb_type")
        feedback_text = st.text_area("Your Feedback", key="fb_text", height=150)
        
        if st.button("Submit Feedback"):
            if feedback_text.strip():
                try:
                    import requests
                    from datetime import datetime
                    
                    # Get webhook from secrets (optional)
                    try:
                        FEEDBACK_WEBHOOK = st.secrets.get("FEEDBACK_WEBHOOK", "")
                    except Exception:
                        FEEDBACK_WEBHOOK = ""
                    
                    payload = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "name": feedback_name or "Anonymous",
                        "email": feedback_email or "N/A",
                        "type": feedback_type,
                        "feedback": feedback_text
                    }
                    
                    if FEEDBACK_WEBHOOK:
                        response = requests.post(FEEDBACK_WEBHOOK, json=payload, timeout=5)
                        if response.status_code == 200:
                            st.success("✅ Thank you! Your feedback has been submitted.")
                        else:
                            st.warning("⚠️ Feedback could not be submitted to webhook.")
                    else:
                        st.info("ℹ️ Feedback webhook not configured. Your feedback was noted locally.")
                except Exception as e:
                    st.error(f"Error submitting feedback: {e}")
            else:
                st.warning("Please enter your feedback before submitting.")
    
    st.subheader("1. Data & Indicators")
    
    # --- Data Source Selection ---
    # Detect if running on Streamlit Cloud
    is_cloud = (
        os.getenv('STREAMLIT_RUNTIME_ENV') == 'cloud' or 
        os.path.exists('/mount/src') or 
        os.getenv('HOME', '').startswith('/home/appuser')
    )
    
    if is_cloud:
        st.warning("""
        📋 **CSV Upload Mode** (Cloud Deployment)
        
        Binance API is geo-restricted on this server. Please:
        1. Download data locally using the desktop app, OR
        2. Get Binance data from: https://data.binance.vision/ 
        3. Upload your CSV file below
        
        **Required columns**: `timestamp`, `open`, `high`, `low`, `close`, `volume`
        """)
        data_source = "Upload CSV"
    else:
    	data_source = st.radio("Data Source", ["Upload CSV", "Download from Binance"], horizontal=True)
    
    if data_source == "Download from Binance":
        c_bin_1, c_bin_2, c_bin_3 = st.columns(3)
        with c_bin_1:
            symbol = st.text_input("Symbol", "BTC/USDT")
        with c_bin_2:
            timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h"], index=1)
        with c_bin_3:
            # Estimate rows
            # Default days based on timeframe
            default_days = 30
            if timeframe == '1m': default_days = 30
            elif timeframe == '5m': default_days = 90
            elif timeframe == '15m': default_days = 180
            elif timeframe == '1h': default_days = 365
            
            days_back = st.number_input("Days Back", min_value=1, value=default_days, step=1)
            
        if st.button("Fetch & Load Data"):
            try:
                import ccxt
                
                # Get API credentials from secrets (optional, for geo-restricted regions)
                try:
                    api_key = st.secrets.get("BINANCE_API_KEY", "")
                    api_secret = st.secrets.get("BINANCE_API_SECRET", "")
                except Exception:
                    # Secrets file doesn't exist - use public API
                    api_key = ""
                    api_secret = ""
                
                if api_key and api_secret:
                    exchange = ccxt.binance({
                        'apiKey': api_key,
                        'secret': api_secret
                    })
                    st.info("🔐 Using authenticated Binance API (private)")
                else:
                    exchange = ccxt.binance()  # Public API
                    st.info("🌍 Using public Binance API (unauthenticated)")
                
                
                # Calculate limit based on days (approx)
                # 1 day = 1440 mins
                # Limit to 50k rows max as per requirement
                limit_rows = 50000
                
                with st.spinner(f"Fetching {symbol} ({timeframe}) for last {days_back} days..."):
                    # Calculate start timestamp
                    since = exchange.milliseconds() - (days_back * 24 * 60 * 60 * 1000)
                    
                    all_ohlcv = []
                    while since < exchange.milliseconds():
                        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                        if not ohlcv:
                            break
                        all_ohlcv.extend(ohlcv)
                        since = ohlcv[-1][0] + 1 # Next timestamp
                        
                        # Safety break for row limit
                        if len(all_ohlcv) >= limit_rows:
                            all_ohlcv = all_ohlcv[:limit_rows]
                            st.warning(f"⚠️ Data truncated to {limit_rows} rows to save memory.")
                            break
                            
                        # Rate limit
                        # time.sleep(exchange.rateLimit / 1000) 
                        
                    if not all_ohlcv:
                        st.error("No data fetched.")
                    else:
                        # Convert to DataFrame
                        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                        # Ensure float32
                        for c in ['open', 'high', 'low', 'close', 'volume']:
                            df[c] = df[c].astype('float32')
                            
                        st.session_state['pipeline'].load_data(df)
                        st.success(f"Loaded {len(df)} rows for {symbol}")
                        
                        # Raw Data Download
                        csv = df.to_csv()
                        st.download_button(
                            label="Download Raw Data (CSV)",
                            data=csv,
                            file_name=f"{symbol.replace('/','_')}_{timeframe}_raw.csv",
                            mime='text/csv'
                        )
                        
            except Exception as e:
                st.error(f"Download Failed: {e}")

    # File Upload (Conditional)
    uploaded_file = None
    local_path = ""
    
    if data_source == "Upload CSV":
        st.info("ℹ️ **CSV Schema**: Column names must match exactly: `timestamp`, `open`, `high`, `low`, `close`, `volume`. Works best if data is from binance.")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        local_path = st.text_input("Or Local Path", "")
    
    if st.button("Compute Indicators"): # Changed label to be generic
        # Determine source
        if data_source == "Download from Binance":
            # Data should already be in pipeline from the Fetch button
            if st.session_state['pipeline'].df is None:
                st.error("Please fetch data first.")
                st.stop()
        else:
            # Load from file
            path = local_path if local_path else uploaded_file
            if not path:
                st.error("Please provide a file.")
                st.stop()
            else:
                try:
                    df = smart_read_csv(path)
                    st.session_state['pipeline'].load_data(df)
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    st.stop()

        try:
            # Proceed with indicators (Data is now in st.session_state['pipeline'].df)
            
            # Build Config Objects
            rsi_parsed = [int(x) for x in config['rsi_vals'].split(',') if x.strip()]
            macd_parsed = []
            for p in config['macd_vals'].split('|'):
                if p.strip():
                    parts = p.split('-')
                    macd_parsed.append((int(parts[0]), int(parts[1]), int(parts[2])))
            bb_parsed = []
            for p in config['bb_vals'].split('|'):
                if p.strip():
                    parts = p.split('-')
                    bb_parsed.append((int(parts[0]), float(parts[1])))
            atr_parsed = [int(x) for x in config['atr_vals'].split(',') if x.strip()]
            
            ind_cfg = IndicatorConfig(rsi=rsi_parsed, macd=macd_parsed, bb=bb_parsed, atr=atr_parsed)
            
            st.session_state['pipeline'].run_indicators(ind_cfg, progress_cb=progress_cb)
            
            # Parse MA windows
            ma_wins = [int(x) for x in config['ma_windows_str'].split(',') if x.strip()]
            
            # Parse new indicator configs
            stoch_parsed = []
            for p in config['stoch_vals'].split('|'):
                if p.strip():
                    parts = p.split('-')
                    stoch_parsed.append((int(parts[0]), int(parts[1])))
            
            willr_parsed = [int(x) for x in config['willr_vals'].split(',') if x.strip()]
            cci_parsed = [int(x) for x in config['cci_vals'].split(',') if x.strip()]
            adx_parsed = [int(x) for x in config['adx_vals'].split(',') if x.strip()]
            
            # Pass kwargs to feature extraction
            st.session_state['pipeline'].run_feature_extraction(
                progress_cb=progress_cb, 
                piv_lookback=int(config['piv_lookback']),
                vol_lookback=int(config['vol_lookback']),
                ma_windows=ma_wins,
                stoch_params=stoch_parsed,
                willr_periods=willr_parsed,
                cci_periods=cci_parsed,
                adx_periods=adx_parsed
            )
            
            # Memory Cleanup
            import gc
            gc.collect()
            
            st.success("Indicators & Features Computed!  ")
        except Exception as e:
            st.error(f"Error: {e}")

    # Feature Selection
    if st.session_state['pipeline'].df_ind is not None:
        st.subheader("2. Feature Selection")
        
        all_feats = st.session_state['pipeline'].feature_cols
        
        # Group features for better UX
        # User Request: 2 Categories (Indicators, Others)
        groups = {
            "Indicators": [],
            "Price/Volume & Others": []
        }
        
        for f in all_feats:
            if any(x in f for x in ['rsi', 'macd', 'bb_', 'atr', 'stoch', 'willr', 'cci', 'adx']):
                groups["Indicators"].append(f)
            else:
                groups["Price/Volume & Others"].append(f)
        
        # Enforce Restrictions
        # st.info("ℹ️ Select up to 4 features.")
        
        selected_feats = []
        
        c_ind, c_oth = st.columns(2)
        
        with c_ind:
            sel_ind = st.multiselect("Indicators", groups["Indicators"])
            
        with c_oth:
            sel_oth = st.multiselect("Price/Volume & Others", groups["Price/Volume & Others"])
            
        selected_feats = sel_ind + sel_oth
        
        st.write(f"**Total Selected:** {len(selected_feats)} / 4")
        
        if len(selected_feats) > 4:
            st.error("⚠️ You have selected more than 4 features. Please deselect some.")
        
        if st.button("Save Selection to Profile"):
            prof = load_profile(config['profile_name'])
            if prof:
                prof.feature_cols = selected_feats
                save_profile(prof)
                st.success(f"Saved {len(selected_feats)} features to profile '{config['profile_name']}'")
            else:
                st.error(f"Could not load profile '{config['profile_name']}'")
        
        # Feature Preview - Data Quality Check
        if selected_feats:
            with st.expander("📊 Data Quality Checks", expanded=False):
                df_features = st.session_state['pipeline'].df_ind[selected_feats]
                
                # Compute statistics
                stats_data = []
                warnings_found = False
                
                for feat in selected_feats:
                    col_data = df_features[feat]
                    na_count = col_data.isna().sum()
                    na_pct = (na_count / len(col_data)) * 100
                    zero_count = (col_data == 0).sum()
                    zero_pct = (zero_count / len(col_data)) * 100
                    
                    # Check for warnings
                    warning = ""
                    if na_pct > 20:
                        warning = "⚠️ High NA%"
                        warnings_found = True
                    elif zero_pct > 30:
                        warning = "⚠️ High zeros%"
                        warnings_found = True
                    
                    unique_count = col_data.nunique()
                    
                    stats_data.append({
                        'Feature': feat,
                        'NA Count': na_count,
                        'NA %': f"{na_pct:.1f}%",
                        'Zero Count': zero_count,
                        'Zero %': f"{zero_pct:.1f}%",
                        'Unique Count': unique_count,
                        'Min': f"{col_data.min():.4f}" if not col_data.isna().all() else "N/A",
                        'Max': f"{col_data.max():.4f}" if not col_data.isna().all() else "N/A",
                        'Mean': f"{col_data.mean():.4f}" if not col_data.isna().all() else "N/A",
                        'Warning': warning
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                if warnings_found:
                    st.warning("⚠️ Some features have high NA or zero counts. Consider feature engineering or exclusion.")
                else:
                    st.success("✅ All features look good - no data quality issues detected.")
        
        st.subheader("3. Clustering")
        
        st.info("ℹ️ **Why Scaling?** Features have different ranges (e.g., volume in millions vs RSI 0-100). Scaling ensures no single feature dominates clustering due to its magnitude. You can also experiment with **'None'** to see raw feature clustering.")
        
        # Step 3a: Run Clustering Only
        # Layout: Scaling options first, then Run button below
        
        st.markdown("##### Scaling Configuration")
        c_scale_1, c_scale_2 = st.columns(2)
        with c_scale_1:
            scaling_method = st.selectbox("Scaling Method", ["None", "Standard", "MinMax", "Robust", "Rolling Z-Score"], index=1)
            
            # Scaling Info
            if scaling_method == "Standard":
                st.caption("ℹ️ **Standard**: Mean=0, Std=1. Good for normal distributions.")
            elif scaling_method == "MinMax":
                st.caption("ℹ️ **MinMax**: Scales to [0, 1]. Sensitive to outliers.")
            elif scaling_method == "Robust":
                st.caption("ℹ️ **Robust**: Uses median/IQR. Good for data with outliers.")
            elif scaling_method == "Rolling Z-Score":
                st.caption("ℹ️ **Rolling Z-Score**: Adapts to changing volatility over time.")
        
        rolling_window = 100
        with c_scale_2:
            if scaling_method == "Rolling Z-Score":
                rolling_window = st.number_input("Rolling Window", value=100, min_value=10, step=10)
                
        if st.button("Run Clustering"):
            st.session_state['pipeline'].run_clustering_only(
                ClusterConfig(
                    method=config['method'], 
                    n_clusters=int(config['n_clusters']), 
                    min_cluster_size=config['min_cluster_size'], 
                    n_jobs=config['n_jobs']
                ),
                selected_feats,
                scaling_method=scaling_method if scaling_method != "None" else None,
                rolling_window=rolling_window
            )
            st.success("Clustering Complete!")

        # Scaled Data Preview
        if st.session_state['pipeline'].df_scaled is not None:
            with st.expander("🔎 Preview Scaled Features (First 50 rows)"):
                st.dataframe(st.session_state['pipeline'].df_scaled.head(50), use_container_width=True)

        # Step 3b: Visualization (PCA)
        if st.session_state['pipeline'].labels is not None:
            st.subheader("Cluster Visualization (PCA)")
            
            st.info("📊 **What am I seeing?** This 2D projection shows how your data clusters in feature space. Each color = one cluster. **Tight, separated clusters** suggest distinct market conditions. **Overlapping clusters** indicate noise or weak feature separation. Use this to visually assess if your features meaningfully group similar conditions.")
            
            try:
                # Inline PCA visualization
                X = st.session_state['pipeline'].df_scaled.fillna(0)
                pca = PCA(n_components=2)
                pcs = pca.fit_transform(X)
                viz_df = pd.DataFrame({
                    'PC1': pcs[:,0], 
                    'PC2': pcs[:,1], 
                    'Cluster': st.session_state['pipeline'].labels.astype(str)
                })
                fig = px.scatter(viz_df, x='PC1', y='PC2', color='Cluster', title="Clusters in PCA Space")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render PCA: {e}")
            
            # Removed cluster distribution graph as requested

        # Step 4: Forward Scan
        if st.session_state['pipeline'].labels is not None:
            st.subheader("4. Forward Scan & Stats")
            st.info("Clusters are ready. Now configure TP/SL to scan for outcomes.")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                tp_pct_in = st.number_input("TP %", value=config['tp_pct']*100, step=0.1, format="%.2f") / 100
            with c2:
                sl_pct_in = st.number_input("SL %", value=config['sl_pct']*100, step=0.1, format="%.2f") / 100
            with c3:
                lookahead_in = st.number_input("Max Lookahead (candles)", value=min(60, config['max_lookahead']), min_value=1, max_value=60, step=1)
                
            # Risk Check Logic
            bypass_risk = False
            if sl_pct_in >= tp_pct_in:
                st.warning("⚠️ TP % must be greater than SL % for favorable risk-reward ratio.")
                bypass_risk = st.checkbox("Bypass Risk Check & Run Anyway", value=False)
                
            if st.button("Run Forward Scan & Stats"):
                if sl_pct_in >= tp_pct_in and not bypass_risk:
                    st.error("Please adjust TP/SL or check 'Bypass Risk Check'.")
                else:
                    st.session_state['pipeline'].run_forward_scan_only(
                        ScanConfig(
                            tp_pct=tp_pct_in,
                            sl_pct=sl_pct_in,
                            max_lookahead=int(lookahead_in)
                        ),
                        n_jobs=config['n_jobs'],
                        progress_cb=progress_cb
                    )
                    st.success("Forward Scan Complete!")

    # Results Display
    if st.session_state['pipeline'].cluster_stats is not None:
        # --- Logging Section (Auto) ---
        st.markdown("---")
        # Header Removed as per request
        
        # Get webhook from secrets (optional)
        try:
            WEBHOOK_URL = st.secrets.get("HQ_RUNS_WEBHOOK", "")
        except Exception:
            WEBHOOK_URL = "" 
        
        # Check for High Quality Clusters
        # Criteria: Precision ≥ 50% (Long or Short) AND Coverage >= 3%
        stats_df = st.session_state['pipeline'].cluster_stats
        
        # Remove PnL columns entirely
        display_cols = [c for c in stats_df.columns if 'pnl' not in c.lower()]
        stats_df = stats_df[display_cols]
        
        
        st.write("#### Results")
        
        # Display clustering configuration from pipeline object
        clustering_method = st.session_state['pipeline'].clustering_method
        scaling_method_display = st.session_state['pipeline'].scaling_method
        num_features = st.session_state['pipeline'].num_features
        st.caption(f"**Clustering Method:** {clustering_method} | **Scaling:** {scaling_method_display} | **Features:** {num_features}")
        
        st.dataframe(stats_df)
        
        hq_clusters = stats_df[
            ((stats_df['long_precision'] >= 0.5) | (stats_df['short_precision'] >= 0.5)) & 
            (stats_df['coverage'] >= 0.03)
        ]
        
        if not hq_clusters.empty:
            st.success(f"Found {len(hq_clusters)} High Quality Clusters (Precision ≥ 50%, Coverage ≥ 3%)!")
            
            # Auto-Log Logic - Send single row with all cluster data
            import requests
            import json
            from datetime import datetime
            
            # Get dataset time period
            df_data = st.session_state['pipeline'].df
            start_date = df_data.index.min().strftime('%Y-%m-%d') if df_data is not None else "N/A"
            end_date = df_data.index.max().strftime('%Y-%m-%d') if df_data is not None else "N/A"
            
            # Format cluster data as comma-separated strings
            cluster_ids = ", ".join(hq_clusters['cluster_id'].astype(str).tolist())
            coverages = ", ".join(hq_clusters['coverage'].round(4).astype(str).tolist())
            long_precisions = ", ".join(hq_clusters['long_precision'].round(4).astype(str).tolist())
            short_precisions = ", ".join(hq_clusters['short_precision'].round(4).astype(str).tolist())
            long_win_rates = ", ".join(hq_clusters['long_win_rate'].round(4).astype(str).tolist())
            short_win_rates = ", ".join(hq_clusters['short_win_rate'].round(4).astype(str).tolist())
            
            # Get current configuration
            scaling_info = f"{config.get('scaling_method', 'None')}"
            if config.get('scaling_method') == 'Rolling Z-Score':
                scaling_info += f", window={config.get('rolling_window', 100)}"
            
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "profile": config.get('profile_name', 'Default'),
                "time_period": f"{start_date} - {end_date}",
                "timeframe": "N/A",  # Add if you track timeframe
                "features": ", ".join(st.session_state['pipeline'].selected_features),
                "scaling_method": scaling_info,
                "tp_sl": f"{config['tp_pct']*100:.2f}%, {config['sl_pct']*100:.2f}%",
                "max_lookahead": config['max_lookahead'],
                "cluster_ids": cluster_ids,
                "coverage": coverages,
                "long_precision": long_precisions,
                "short_precision": short_precisions,
                "long_win_rate": long_win_rates,
                "short_win_rate": short_win_rates
            }
            
            # Generate a unique ID for this run configuration
            run_id = f"{log_entry['timestamp']}_{len(hq_clusters)}"
            
            if st.session_state.get('last_logged_run_id') != run_id:
                # Log to Webhook (Cloud Only - No Local CSV)
                if WEBHOOK_URL:
                    try:
                        response = requests.post(WEBHOOK_URL, json=log_entry, timeout=5)
                    except Exception:
                        pass # Silent fail for auto-log
                
                st.session_state['last_logged_run_id'] = run_id
                

        # Download Annotated Data
            # Annotated Data Download
            if st.session_state['pipeline'].df_fs is not None:
                # Prepare annotated DF
                # Include: Timestamp, OHLCV, Features, Cluster ID, Exit Outcomes
                df_anno = st.session_state['pipeline'].df_fs.copy()
                
                # Select relevant columns
                base_cols = ['open', 'high', 'low', 'close', 'volume']
                feat_cols = st.session_state['pipeline'].feature_cols
                res_cols = ['cluster_id', 'long_exit', 'short_exit', 'long_exit_price', 'short_exit_price']
                
                final_cols = base_cols + feat_cols + [c for c in res_cols if c in df_anno.columns]
                df_anno = df_anno[final_cols]
                
                csv_anno = df_anno.to_csv()
                st.download_button("Download Annotated Data (CSV)", csv_anno, "annotated_data.csv", "text/csv")
        
        st.markdown("---")
        
        # Enhanced Visualization (Step 4b)
        st.write("#### 🔬 Cluster Visualization")
        
        st.info("🎯 **Visual Pattern Recognition**: This scatter plot shows trade outcomes within clusters. **Green (TP)** = winning trades, **Red (SL)** = losing trades, **Gray** = no action. If you see clear spatial separation between TP and SL points, your features are successfully identifying profitable patterns! Overlapping colors indicate the cluster contains random outcomes—features aren't separating winners from losers.")
        
        try:
            df_fs = st.session_state['pipeline'].df_fs
            df_scaled = st.session_state['pipeline'].df_scaled  # Use the same scaled data as first plot
            
            # Re-compute PCA using the SAME SCALED features that were used for the first plot
            X = df_scaled.fillna(0)
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X)
            
            viz_df = df_fs.copy()
            viz_df['PC1'] = pcs[:,0]
            viz_df['PC2'] = pcs[:,1]
            viz_df['Cluster'] = viz_df['cluster_id'].astype(str)
            
            # Filter Logic
            c_filter, c_dir = st.columns(2)
            
            with c_filter:
                all_clusters = sorted(df_fs['cluster_id'].unique())
                cluster_opts = ["All"] + [str(c) for c in all_clusters]
                sel_cluster = st.selectbox("Filter Cluster", cluster_opts)
                
            with c_dir:
                sel_direction = st.selectbox("Direction", ["Long", "Short"])
            
            if sel_cluster != "All":
                viz_df = viz_df[viz_df['Cluster'] == sel_cluster]
                
            # Determine Outcome based on Direction
            if sel_direction == "Long":
                viz_df['Outcome'] = viz_df['long_exit']
            else:
                viz_df['Outcome'] = viz_df['short_exit']
                
            # Calculate Counts
            counts = viz_df['Outcome'].value_counts()
            total = len(viz_df)
            
            # Display Counts
            st.markdown("##### Outcome Distribution")
            
            # Layout: True Positives | False Positives (SL + No Action) | Total
            c_tp_main, c_fp_main, c_total_main = st.columns([1, 2, 0.8])
            
            # True Positives Section
            with c_tp_main:
                st.markdown("### True Positives")
                c_tp = counts.get('TP', 0)
                p_tp = (c_tp / total * 100) if total > 0 else 0
                st.metric("Win", f"{c_tp}", f"{p_tp:.1f}%", delta_color="off")
            
            # False Positives Section
            with c_fp_main:
                # Calculate Total FP (SL + No Action)
                c_sl = counts.get('SL', 0)
                c_na = counts.get('NEITHER', 0)
                c_fp_total = c_sl + c_na
                p_fp_total = (c_fp_total / total * 100) if total > 0 else 0
                
                st.markdown(f"### False Positives (Total: {c_fp_total} | {p_fp_total:.1f}%)")
                
                # Breakdown of FP
                c_fp1, c_fp2 = st.columns(2)
                with c_fp1:
                    p_sl = (c_sl / total * 100) if total > 0 else 0
                    st.metric("SL (Loss)", f"{c_sl}", f"{p_sl:.1f}%", delta_color="off")
                with c_fp2:
                    p_na = (c_na / total * 100) if total > 0 else 0
                    st.metric("No Action", f"{c_na}", f"{p_na:.1f}%", delta_color="off")
            
            # Total Section
            with c_total_main:
                st.markdown("### Total")
                st.metric("Trades", f"{total}")
            
            # Color Map
            color_map = {
                'TP': 'green',
                'SL': 'red',
                'NEITHER': 'lightgray'
            }
            
            fig = px.scatter(
                viz_df, 
                x='PC1', 
                y='PC2', 
                color='Outcome', 
                color_discrete_map=color_map,
                hover_data=['Cluster', 'close', 'exit_price'],
                title=f"Cluster Analysis - {sel_direction} - {'All Clusters' if sel_cluster == 'All' else 'Cluster ' + sel_cluster}",
                opacity=0.7
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"outcome_viz_{sel_direction}_{sel_cluster}")
            
            # Legend / Explainer
            with st.expander("ℹ️ Legend: What do the colors mean?"):
                st.markdown("""
                - **TP (Green)**: Take Profit hit first.
                - **SL (Red)**: Stop Loss hit first.
                - **No Action (Gray)**: Neither TP nor SL was hit within the lookahead.
                """)
                
        except Exception as e:
            st.warning(f"Could not render Visualization: {e}")

        st.write("#### Next Steps")
        c_insp, c_clust = st.columns(2)
        
        with c_insp:
            if st.button("🔍 Go to Inspection", use_container_width=True):
                st.session_state['active_tab'] = "Inspection"
                st.rerun()
