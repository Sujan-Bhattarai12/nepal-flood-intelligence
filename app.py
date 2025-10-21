"""
Nepal Flood Intelligence Platform - Streamlit Dashboard
Main application entry point
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
import os

# Import from src modules
from src.config import Config
from src.data_loader import ERA5DataLoader, CacheManager
from src.analytics import FloodAnalytics
from src.models import FloodDataset, FloodForecaster
from src.dashboard import (
    apply_custom_styles,
    render_sidebar,
    render_overview_tab,
    render_historical_tab,
    render_monsoon_tab,
    render_forecast_tab,
    render_flood_timing_tab,
    render_flash_flood_tab
)


def main():
    """Main application"""
    st.set_page_config(
        page_title="Nepal Flood Intelligence", 
        layout="wide", 
        page_icon="🇳🇵"
    )
    
    # Apply custom styling
    apply_custom_styles()
    
    # Professional header
    st.markdown('<p class="main-header">Nepal Flood Risk Intelligence Platform</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">ERA5 Climate Reanalysis & Machine Learning Forecasting System</p>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    
    # Render sidebar and get station config
    station_config = render_sidebar()
    
    # Main content
    if st.session_state['data'] is not None:
        data = st.session_state['data']
        analytics = st.session_state['analytics']
        station_config = st.session_state['station_config']
        
        # Cache info
        cache_manager = CacheManager()
        cache_age_hours = 0
        cache_key = cache_manager._get_cache_key(station_config['station_id'], 5)
        cache_path = cache_manager._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            cache_time = os.path.getmtime(cache_path)
            cache_age_hours = (datetime.now().timestamp() - cache_time) / 3600
        
        st.info(f"Data Source: ERA5 Climate Reanalysis | Cache Age: {cache_age_hours:.1f} hours (refreshes daily)")
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Overview", 
            "Historical Analysis", 
            "Monsoon Risk", 
            "Forecast", 
            "Flood Timing", 
            "Flash Flood Analysis"
        ])
        
        with tab1:
            render_overview_tab(data, analytics, station_config)
        
        with tab2:
            render_historical_tab(analytics)
        
        with tab3:
            render_monsoon_tab(analytics)
        
        with tab4:
            render_forecast_tab(data, station_config)
        
        with tab5:
            render_flood_timing_tab(data, station_config)
        
        with tab6:
            render_flash_flood_tab(data, station_config)
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="background: #ffffff; 
             padding: 3rem; border-radius: 8px; border: 1px solid #e2e8f0; margin: 2rem 0;">
            <h2 style="color: #1e293b; margin: 0;">Welcome to Nepal Flood Intelligence</h2>
            <p style="font-size: 1.05rem; margin: 1.5rem 0; color: #64748b;">
                Advanced flood forecasting and risk analysis for Nepal's major river basins
            </p>
            <p style="color: #64748b; margin: 0.5rem 0;">
                • Real-time monitoring of 5 major rivers<br>
                • ERA5 climate reanalysis data<br>
                • Machine learning-based 7-day forecasts<br>
                • Historical trend analysis and seasonal risk profiling
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("Select a river station from the sidebar to begin analysis")


if __name__ == "__main__":
    main()