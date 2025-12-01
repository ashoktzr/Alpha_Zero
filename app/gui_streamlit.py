# app/gui_streamlit.py
"""Main Streamlit GUI entry point - modularized"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from app.pipeline import Pipeline
from app.tabs import (
    sidebar_config,
    pipeline_page,
    inspection_page
)

# --- Setup & State ---
st.set_page_config(layout="wide", page_title="AlphaZero")

# NOTE: To increase upload limit, run with: streamlit run app/gui_streamlit.py --server.maxUploadSize=200
# We can't set this programmatically inside the script easily without config.toml

if 'pipeline' not in st.session_state:
    st.session_state['pipeline'] = Pipeline()

if 'active_profile' not in st.session_state:
    st.session_state['active_profile'] = "Default"

# Helper for progress
progress_text = st.empty()
def progress_cb(event):
    stage = event.get('stage', 'Unknown')
    msg = event.get('msg', '')
    ev_type = event.get('event', 'info')
    
    if ev_type == 'start':
        progress_text.info(f"⏳ {stage}: {msg}")
    elif ev_type == 'done':
        progress_text.success(f"✅ {stage}: {msg}")
    elif ev_type == 'info':
        progress_text.write(f"ℹ️ {stage}: {msg}")

# --- Sidebar ---
with st.sidebar:
    config = sidebar_config.render()

# --- Main Area ---
st.title("AlphaZero: The Indicator Challenge")



# Navigation State
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = "Pipeline"

# Callback to handle radio change
def on_nav_change():
    if 'nav_radio' in st.session_state:
        st.session_state['active_tab'] = st.session_state['nav_radio']

# Sync radio with session state
# We use a separate key for the widget to avoid circular updates, 
# but we initialize it from the state.
tab_options = ["Pipeline", "Inspection"]
try:
    idx = tab_options.index(st.session_state['active_tab'])
except ValueError:
    # Default to Pipeline if current tab is gone
    idx = 0
    st.session_state['active_tab'] = "Pipeline"

selection = st.radio(
    "", 
    tab_options, 
    index=idx, 
    horizontal=True, 
    key="nav_radio", 
    on_change=on_nav_change,
    label_visibility="collapsed"
)

st.markdown("---")

# Render Active Tab
if st.session_state['active_tab'] == "Pipeline":
    pipeline_page.render(config, progress_cb)

elif st.session_state['active_tab'] == "Inspection":
    inspection_page.render(config['tp_pct'], config['sl_pct'])
