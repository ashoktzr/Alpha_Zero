# app/pages/sidebar_config.py
"""Sidebar configuration and profile management"""
import streamlit as st
import os
from app.config_types import IndicatorConfig, ScanConfig, ClusterConfig, UserProfile, FeatureConfig
from app.profiles import list_profiles, load_profile, save_profile, delete_profile


def render():
    """Render sidebar configuration and return config dict"""
    st.title("Configuration")
    
    # Profile Management
    st.subheader("Profiles")
    profiles = list_profiles()
    selected_profile_name = st.selectbox("Load Profile", ["New..."] + profiles)
    
    if selected_profile_name == "New...":
        profile_name_input = st.text_input("Profile Name", value="MyProfile")
    else:
        profile_name_input = selected_profile_name

    # Load Profile Data
    if selected_profile_name != "New...":
        loaded_prof = load_profile(selected_profile_name)
        if loaded_prof:
            # Populate defaults from profile
            def_rsi = ",".join(map(str, loaded_prof.indicator_config.rsi))
            def_macd = "|".join([f"{a}-{b}-{c}" for a,b,c in loaded_prof.indicator_config.macd])
            def_bb = "|".join([f"{a}-{b}" for a,b in loaded_prof.indicator_config.bb])
            def_atr = ",".join(map(str, loaded_prof.indicator_config.atr))
            
            def_tp = loaded_prof.scan_config.tp_pct
            def_sl = loaded_prof.scan_config.sl_pct
            def_lookahead = loaded_prof.scan_config.max_lookahead
            
            def_method = loaded_prof.cluster_config.method
            def_n_clusters = loaded_prof.cluster_config.n_clusters
            def_min_cluster = loaded_prof.cluster_config.min_cluster_size
            def_n_jobs = loaded_prof.cluster_config.n_jobs

            def_piv = loaded_prof.feature_config.piv_lookback
            def_vol = loaded_prof.feature_config.vol_lookback
            def_ma = ",".join(map(str, loaded_prof.feature_config.ma_windows))
        else:
            # Fallback defaults
            def_rsi, def_macd, def_bb, def_atr = "14", "12-26-9", "20-2.0", "14"
            def_tp, def_sl, def_lookahead = 0.01, 0.005, 400
            def_method, def_n_clusters, def_min_cluster, def_n_jobs = "hdbscan", 8, 25, 1
            def_piv, def_vol, def_ma = 50, 20, "20, 50, 200"
    else:
        # Defaults for new
        def_rsi, def_macd, def_bb, def_atr = "14", "12-26-9", "20-2.0", "14"
        def_tp, def_sl, def_lookahead = 0.01, 0.005, 400
        def_method, def_n_clusters, def_min_cluster, def_n_jobs = "hdbscan", 8, 25, 1
        def_piv, def_vol, def_ma = 50, 20, "20, 50, 200"

    # Inputs
    st.markdown("---")
    st.subheader("Indicators")
    rsi_vals = st.text_input("RSI (comma sep)", value=def_rsi)
    macd_vals = st.text_input("MACD (fast-slow-sig | ...)", value=def_macd)
    bb_vals = st.text_input("BB (n-k | ...)", value=def_bb)
    atr_vals = st.text_input("ATR (comma sep)", value=def_atr)
    stoch_vals = st.text_input("Stochastic (k_period-d_period | ...)", value="14-3")
    willr_vals = st.text_input("Williams %R (comma sep)", value="14")
    cci_vals = st.text_input("CCI (comma sep)", value="20")
    adx_vals = st.text_input("ADX (comma sep)", value="14")

    st.markdown("---")
    st.subheader("Feature Settings")
    piv_lookback = st.number_input("Pivot Lookback", value=def_piv)
    vol_lookback = st.number_input("Volume/Vol Lookback", value=def_vol)
    ma_windows_str = st.text_input("MA Windows (comma sep)", value=def_ma)
    
    st.markdown("---")
    st.subheader("Scan Settings")
    tp_pct_val = st.number_input("TP %", value=float(def_tp)*100, format="%.2f")
    sl_pct_val = st.number_input("SL %", value=float(def_sl)*100, format="%.2f")
    max_lookahead = st.number_input("Max Lookahead", value=int(def_lookahead))

    st.markdown("---")
    st.subheader("Clustering")
    method = st.selectbox("Method", ["hdbscan", "kmeans"], index=0 if def_method=="hdbscan" else 1)
    min_cluster_size = st.number_input("Min Cluster Size (HDBSCAN)", value=int(def_min_cluster))
    n_clusters = st.number_input("N Clusters (KMeans)", value=int(def_n_clusters))
    
    st.markdown("---")
    st.subheader("Performance")
    n_jobs = st.number_input("N Jobs (Parallelism)", value=int(def_n_jobs), min_value=-1, max_value=os.cpu_count())
    st.caption("Note: Set to -1 to use all cores. Higher values speed up scanning but increase memory usage.")

    # Save Profile Button
    if st.button("Save Profile"):
        try:
            # Parse inputs
            rsi_parsed = [int(x) for x in rsi_vals.split(',') if x.strip()]
            macd_parsed = []
            for p in macd_vals.split('|'):
                if p.strip():
                    parts = p.split('-')
                    macd_parsed.append((int(parts[0]), int(parts[1]), int(parts[2])))
            bb_parsed = []
            for p in bb_vals.split('|'):
                if p.strip():
                    parts = p.split('-')
                    bb_parsed.append((int(parts[0]), float(parts[1])))
            atr_parsed = [int(x) for x in atr_vals.split(',') if x.strip()]
            ma_parsed = [int(x) for x in ma_windows_str.split(',') if x.strip()]

            prof = UserProfile(
                name=profile_name_input,
                indicator_config=IndicatorConfig(rsi=rsi_parsed, macd=macd_parsed, bb=bb_parsed, atr=atr_parsed),
                scan_config=ScanConfig(tp_pct=tp_pct_val/100, sl_pct=sl_pct_val/100, max_lookahead=max_lookahead),
                cluster_config=ClusterConfig(method=method, n_clusters=n_clusters, min_cluster_size=min_cluster_size, n_jobs=n_jobs),
                feature_config=FeatureConfig(piv_lookback=piv_lookback, vol_lookback=vol_lookback, ma_windows=ma_parsed)
            )
            save_profile(prof)
            st.success(f"Profile '{profile_name_input}' saved!")
            st.rerun()
        except Exception as e:
            st.error(f"Error saving profile: {e}")

    if selected_profile_name != "New..." and st.button("Delete Profile"):
        delete_profile(selected_profile_name)
        st.warning(f"Deleted {selected_profile_name}")
        st.rerun()
    
    # Return configuration as dict
    return {
        'profile_name': selected_profile_name,
        'rsi_vals': rsi_vals,
        'macd_vals': macd_vals,
        'bb_vals': bb_vals,
        'atr_vals': atr_vals,
        'stoch_vals': stoch_vals,
        'willr_vals': willr_vals,
        'cci_vals': cci_vals,
        'adx_vals': adx_vals,
        'piv_lookback': piv_lookback,
        'vol_lookback': vol_lookback,
        'ma_windows_str': ma_windows_str,
        'tp_pct': tp_pct_val/100,
        'sl_pct': sl_pct_val/100,
        'max_lookahead': max_lookahead,
        'method': method,
        'n_clusters': n_clusters,
        'min_cluster_size': min_cluster_size,
        'n_jobs': n_jobs,
        'feature_cols': loaded_prof.feature_cols if selected_profile_name != "New..." and loaded_prof else []
    }
