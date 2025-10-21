"""
Dashboard UI Components and Tab Rendering
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime, timedelta

from .config import Config
from .data_loader import ERA5DataLoader, CacheManager
from .models import FloodDataset, FloodForecaster


def apply_custom_styles():
    """Apply custom CSS styling"""
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 800;
            text-align: center;
            background: linear-gradient(135deg, #dc143c 0%, #003893 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 1rem 0;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] strong {
            color: white !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
        }
        [data-testid="stSidebar"] button {
            background: linear-gradient(135deg, #dc143c 0%, #b91c1c 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
        }
        </style>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_station_data(station_id, station_name):
    """Load and cache ERA5 data for a station"""
    station_config = Config.TARGET_STATIONS[station_id]
    loader = ERA5DataLoader(station_config, 5)
    data = loader.fetch_era5_data()
    return data


@st.cache_resource(ttl=86400)  # Cache for 24 hours
def load_station_model(station_id, data_hash):
    """Load and cache trained model for a station"""
    station_config = Config.TARGET_STATIONS[station_id]
    cache_manager = CacheManager()
    
    # Check if model is cached
    if cache_manager.is_model_cache_valid(station_config['station_id']):
        forecaster, dataset = cache_manager.load_model_from_cache(station_config['station_id'])
        if forecaster is not None and dataset is not None:
            return forecaster, dataset
    
    # If not cached or invalid, return None
    return None, None


def render_sidebar():
    """Render sidebar with station selection and controls"""
    
    # Station selection
    station_key = st.sidebar.selectbox(
        "Select Monitoring Station",
        list(Config.TARGET_STATIONS.keys()),
        format_func=lambda x: Config.TARGET_STATIONS[x]['name'],
        label_visibility="collapsed"
    )
    
    station_config = Config.TARGET_STATIONS[station_key]
    
    # River name banner at top
    st.sidebar.markdown(f"""
        <div class="river-banner">
            <div class="river-name">{station_config['river']} River</div>
            <div class="river-subtitle">{station_config['district']} District</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Station information section
    st.sidebar.markdown('<div class="section-title">Station Details</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f"""
        <div class="info-item">
            <span class="info-label">Station</span>
            <span class="info-value">{station_config['station_id']}</span>
        </div>
        <div class="info-item">
            <span class="info-label">Drainage Area</span>
            <span class="info-value">{station_config['drainage_area']:,} km²</span>
        </div>
        <div class="info-item">
            <span class="info-label">Elevation</span>
            <span class="info-value">{station_config['elevation']} m</span>
        </div>
        <div class="info-item">
            <span class="info-label">Location</span>
            <span class="info-value">{station_config['lat']:.2f}°N, {station_config['lon']:.2f}°E</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Flood thresholds section
    st.sidebar.markdown('<div class="section-title">Flood Thresholds</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f"""
        <div class="info-item">
            <span class="info-label">Flood Stage</span>
            <span class="info-value">{station_config['flood_stage']:.1f} m</span>
        </div>
        <div class="info-item">
            <span class="info-label">Moderate Flood</span>
            <span class="info-value">{station_config['moderate_flood']:.1f} m</span>
        </div>
        <div class="info-item">
            <span class="info-label">Major Flood</span>
            <span class="info-value">{station_config['major_flood']:.1f} m</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    cache_manager = CacheManager()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Clear Cache", help="Clear all cached data and models", use_container_width=True):
            cache_manager.clear_cache()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.success("Cache cleared!")
            st.rerun()
    
    with col2:
        if st.button("Force Refresh", help="Force complete data reload", use_container_width=True):
            cache_manager.clear_cache()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.success("Ready for fresh load!")
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Auto-load data when station changes
    if station_key not in st.session_state.get('loaded_stations', set()):
        with st.spinner(f"Loading data for {station_config['name']}..."):
            # Load data with caching
            data = load_station_data(station_key, station_config['name'])
            
            if data is not None and len(data) > 1000:
                st.session_state['data'] = data
                st.session_state['station_config'] = station_config
                
                from .analytics import FloodAnalytics
                st.session_state['analytics'] = FloodAnalytics(data, station_config)
                
                # Create a hash of the data for model caching
                data_hash = hash(len(data))
                
                # Try to load cached model
                forecaster, dataset = load_station_model(station_key, data_hash)
                
                if forecaster is None or dataset is None:
                    # Train model if not cached
                    with st.spinner("Training forecast model (one-time setup)..."):
                        try:
                            dataset = FloodDataset(data[-8760:])
                            train_size = int(0.8 * len(dataset))
                            train_dataset, val_dataset = torch.utils.data.random_split(
                                dataset, [train_size, len(dataset) - train_size]
                            )
                            
                            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=32)
                            
                            forecaster = FloodForecaster()
                            forecaster.train_silent(train_loader, val_loader, 30)
                            
                            cache_manager.save_model_to_cache(forecaster, dataset, station_config['station_id'])
                            
                            st.success("Model trained and cached!")
                        except Exception as e:
                            st.warning(f"Model training skipped: {str(e)}")
                            forecaster, dataset = None, None
                
                st.session_state['forecaster'] = forecaster
                st.session_state['dataset'] = dataset
                
                # Mark this station as loaded
                if 'loaded_stations' not in st.session_state:
                    st.session_state['loaded_stations'] = set()
                st.session_state['loaded_stations'].add(station_key)
                
                st.sidebar.success(f"Data loaded: {len(data):,} records")
                st.rerun()
    
    return station_config


def render_overview_tab(data, analytics, station_config):
    """Render Overview tab"""
    st.subheader(f"Current Status - {station_config['name']}")
    current_stats = analytics.get_current_vs_historical()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Discharge", f"{current_stats['current_discharge']:.0f} m3/s")
    col2.metric("Water Level", f"{current_stats['current_level']:.2f} m")
    col3.metric("Percentile", f"{current_stats['percentile_discharge']:.0f}th")
    col4.metric("Flood Stage", f"{station_config['flood_stage']:.1f} m", 
               "Normal" if current_stats['current_level'] < station_config['flood_stage'] else "ABOVE")
    
    st.markdown("---")
    
    st.subheader("Recent Flood Events")
    flood_events = analytics.identify_flood_events()
    if not flood_events.empty:
        recent = flood_events.tail(10).sort_values('start', ascending=False)
        display_df = recent[['start', 'duration_hours', 'peak_level', 'severity']].copy()
        display_df['start'] = display_df['start'].dt.strftime('%Y-%m-%d')
        display_df['duration_hours'] = display_df['duration_hours'].round(1)
        display_df['peak_level'] = display_df['peak_level'].round(2)
        display_df.columns = ['Date', 'Duration (hrs)', 'Peak Level (m)', 'Severity']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        col1, col2, col3 = st.columns(3)
        total_floods = len(flood_events)
        years = data['year'].nunique()
        col1.metric("Total Floods", total_floods)
        col2.metric("Avg per Year", f"{total_floods/years:.1f}")
        if 'severity' in flood_events.columns:
            major_floods = len(flood_events[flood_events['severity'] == 'Major'])
            col3.metric("Major Floods", major_floods)
    else:
        st.info("No flood events detected")


def render_historical_tab(analytics):
    """Render Historical Analysis tab"""
    st.subheader("5-Year Historical Analysis")
    
    yoy = analytics.year_over_year_comparison()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=yoy['year'], y=yoy['peak_discharge'], marker_color='#dc143c'))
        fig.update_layout(title="Annual Peak Discharge", xaxis_title="Year", yaxis_title="Discharge (m3/s)", height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=yoy['year'], y=yoy['flood_days'], marker_color='#003893'))
        fig.update_layout(title="Annual Flood Days", xaxis_title="Year", yaxis_title="Days", height=350)
        st.plotly_chart(fig, use_container_width=True)


def render_monsoon_tab(analytics):
    """Render Monsoon Risk tab"""
    st.subheader("Monsoon & Seasonal Risk Analysis")
    
    seasonal = analytics.seasonal_risk_profile()
    
    if not seasonal.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=seasonal['season'],
            y=seasonal['risk_score'],
            marker_color=['#dc2626' if x > 60 else '#f59e0b' if x > 30 else '#10b981' for x in seasonal['risk_score']],
            text=seasonal['risk_score'].round(0),
            textposition='outside'
        ))
        fig.update_layout(title="Seasonal Risk Scores", xaxis_title="Season", yaxis_title="Risk Score", height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_forecast_tab(data, station_config):
    """Render Forecast tab"""
    st.subheader("7-Day Flood Forecast")
    
    if st.session_state.get('forecaster') and st.session_state.get('dataset'):
        forecaster = st.session_state['forecaster']
        dataset = st.session_state['dataset']
        
        try:
            lookback = 168 // 6
            input_discharge = data['discharge_cumecs'].iloc[-lookback:].values
            input_level = data['water_level_m'].iloc[-lookback:].values
            
            input_norm = np.stack([
                (input_discharge - dataset.discharge_mean) / (dataset.discharge_std + 1e-8),
                (input_level - dataset.level_mean) / (dataset.level_std + 1e-8)
            ], axis=-1)
            
            prediction = forecaster.predict(input_norm, dataset)
            
            last_timestamp = data['timestamp'].iloc[-1]
            forecast_times = pd.date_range(start=last_timestamp + timedelta(hours=6), periods=28, freq='6H')
            
            col1, col2, col3 = st.columns(3)
            max_level = prediction[:, 1].max()
            current_level = data['water_level_m'].iloc[-1]
            flood_prob = (prediction[:, 1] > station_config['flood_stage']).sum() / 28 * 100
            
            col1.metric("Peak Forecast", f"{max_level:.2f} m", f"{(max_level - current_level):+.2f} m")
            col2.metric("Peak Discharge", f"{prediction[:, 0].max():.0f} m3/s")
            col3.metric("Flood Probability", f"{flood_prob:.0f}%")
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=("Discharge", "Water Level"), vertical_spacing=0.12)
            
            hist_data = data.iloc[-168:]
            fig.add_trace(go.Scatter(x=hist_data['timestamp'], y=hist_data['discharge_cumecs'], mode='lines', name='Historical', line=dict(color='#64748b')), row=1, col=1)
            fig.add_trace(go.Scatter(x=forecast_times, y=prediction[:, 0], mode='lines+markers', name='Forecast', line=dict(color='#dc143c', width=3)), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=hist_data['timestamp'], y=hist_data['water_level_m'], mode='lines', showlegend=False, line=dict(color='#64748b')), row=2, col=1)
            fig.add_trace(go.Scatter(x=forecast_times, y=prediction[:, 1], mode='lines+markers', showlegend=False, line=dict(color='#003893', width=3)), row=2, col=1)
            
            fig.add_hline(y=station_config['flood_stage'], line_dash="dash", line_color="orange", annotation_text="Flood Stage", row=2, col=1)
            
            fig.update_layout(height=600, hovermode='x unified')
            fig.update_yaxes(title_text="Discharge (m3/s)", row=1, col=1)
            fig.update_yaxes(title_text="Water Level (m)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Forecast error: {str(e)}")
    else:
        st.info("Forecast model not available")


def render_flood_timing_tab(data, station_config):
    """Render Flood Timing tab"""
    st.subheader("Multi-Year Flood Timing Analysis")
    
    st.info("Overlays all years to reveal seasonal flood patterns and identify unusual non-monsoon floods")
    st.caption("Note: Red star markers indicate flood events outside monsoon season")
    
    years_available = sorted(data['year'].unique())
    
    if len(years_available) > 0:
        fig = go.Figure()
        
        colors = ['#dc143c', '#003893', '#f59e0b', '#10b981', '#8b5cf6']
        unusual_floods_summary = []
        
        for idx, year in enumerate(years_available):
            year_data = data[data['year'] == year].copy()
            year_data = year_data.sort_values('day_of_year')
            
            color = colors[idx % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=year_data['day_of_year'],
                y=year_data['water_level_m'],
                mode='lines',
                name=f'{year}',
                line=dict(color=color, width=2),
                hovertemplate=f'<b>{year}</b><br>Day: %{{x}}<br>Level: %{{y:.2f}} m<extra></extra>'
            ))
            
            unusual_flood_mask = (
                (year_data['water_level_m'] > station_config['flood_stage']) &
                ((year_data['day_of_year'] < 152) | (year_data['day_of_year'] > 273))
            )
            
            unusual_floods = year_data[unusual_flood_mask]
            
            if len(unusual_floods) > 0:
                fig.add_trace(go.Scatter(
                    x=unusual_floods['day_of_year'],
                    y=unusual_floods['water_level_m'],
                    mode='markers',
                    name='Unusual Floods',
                    marker=dict(
                        color=color,
                        size=8,
                        symbol='star',
                        line=dict(color='red', width=2)
                    ),
                    hovertemplate=f'<b>{year} UNUSUAL FLOOD</b><br>Day: %{{x}}<br>Level: %{{y:.2f}} m<extra></extra>',
                    showlegend=(idx == 0)
                ))
                
                unusual_count = len(unusual_floods)
                max_unusual_level = unusual_floods['water_level_m'].max()
                unusual_floods_summary.append({
                    'year': year,
                    'count': unusual_count,
                    'max_level': max_unusual_level
                })
        
        fig.add_hline(y=station_config['flood_stage'], line_dash="dash", line_color="orange", annotation_text="Flood Stage", annotation_position="right")
        fig.add_hline(y=station_config['moderate_flood'], line_dash="dot", line_color="red", annotation_text="Moderate Flood", annotation_position="right")
        
        fig.add_vrect(x0=152, x1=273, fillcolor="lightblue", opacity=0.1, layer="below", line_width=0, annotation_text="Monsoon Season", annotation_position="top left")
        
        fig.update_layout(
            title="Water Level Throughout the Year - All Years Overlaid",
            xaxis_title="Day of Year",
            yaxis_title="Water Level (m)",
            height=600,
            hovermode='closest',
            legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)")
        )
        
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig.update_xaxes(tickmode='array', tickvals=month_starts, ticktext=month_names)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Summary Statistics**")
            st.markdown(f"Years Analyzed: {len(years_available)}")
            st.markdown(f"Flood Threshold: {station_config['flood_stage']:.2f} m")
            st.markdown(f"Monsoon Period: Day 152-273 (June-September)")
        
        with col2:
            st.markdown("**Unusual Floods (Non-Monsoon)**")
            if unusual_floods_summary:
                for item in unusual_floods_summary:
                    st.markdown(f"**{item['year']}:** {item['count']} events, peak {item['max_level']:.2f} m")
            else:
                st.markdown("No unusual non-monsoon floods detected")
        
        st.markdown("---")
        st.markdown("""
        **Legend:**
        - **Solid lines**: Water level trends for each year
        - **⭐ Star markers**: Unusual flood events outside monsoon season
        - **Blue shaded area**: Typical monsoon season (June-September)
        - **Orange dashed line**: Flood stage threshold
        - **Red dotted line**: Moderate flood threshold
        """)
    else:
        st.warning("No data available for multi-year comparison")


def render_flash_flood_tab(data, station_config):
    """Render Flash Flood Analysis tab"""
    st.subheader("Extreme Event Analysis (Top 1% Water Levels)")
    
    st.info("Identifies when extreme water levels occur throughout the year")
    st.caption("Each point represents a top 1% water level event for that year")
    
    years_available = sorted(data['year'].unique())
    
    if len(years_available) > 0:
        fig = go.Figure()
        
        colors = ['#dc143c', '#003893', '#f59e0b', '#10b981', '#8b5cf6']
        flash_flood_summary = []
        
        for idx, year in enumerate(years_available):
            year_data = data[data['year'] == year].copy()
            
            percentile_99 = year_data['water_level_m'].quantile(0.99)
            top_1_percent = year_data[year_data['water_level_m'] >= percentile_99].copy()
            top_1_percent = top_1_percent.sort_values('day_of_year')
            
            color = colors[idx % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=top_1_percent['day_of_year'],
                y=top_1_percent['water_level_m'],
                mode='markers',
                name=f'{year}',
                marker=dict(color=color, size=8, opacity=0.7, line=dict(color='white', width=1)),
                hovertemplate=f'<b>{year}</b><br>Day: %{{x}}<br>Level: %{{y:.2f}} m<extra></extra>'
            ))
            
            flash_flood_summary.append({
                'year': year,
                'count': len(top_1_percent),
                'threshold_99': percentile_99,
                'max_level': top_1_percent['water_level_m'].max(),
                'monsoon_events': len(top_1_percent[(top_1_percent['day_of_year'] >= 152) & (top_1_percent['day_of_year'] <= 273)]),
                'non_monsoon_events': len(top_1_percent[(top_1_percent['day_of_year'] < 152) | (top_1_percent['day_of_year'] > 273)])
            })
        
        fig.add_hline(y=station_config['flood_stage'], line_dash="dash", line_color="orange", annotation_text="Flood Stage", annotation_position="right")
        fig.add_hline(y=station_config['moderate_flood'], line_dash="dot", line_color="red", annotation_text="Moderate Flood", annotation_position="right")
        
        fig.add_vrect(x0=152, x1=273, fillcolor="lightblue", opacity=0.1, layer="below", line_width=0, annotation_text="Monsoon Season", annotation_position="top left")
        
        fig.update_layout(
            title="Top 1% Extreme Water Levels - When Do Flash Floods Occur?",
            xaxis_title="Day of Year",
            yaxis_title="Water Level (m)",
            height=600,
            hovermode='closest',
            legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)")
        )
        
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig.update_xaxes(tickmode='array', tickvals=month_starts, ticktext=month_names)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("**Extreme Event Summary (Top 1%)**")
        
        summary_df = pd.DataFrame(flash_flood_summary)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Event Counts by Year**")
            display_df = summary_df[['year', 'count', 'monsoon_events', 'non_monsoon_events']].copy()
            display_df.columns = ['Year', 'Total Events', 'Monsoon', 'Non-Monsoon']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Extreme Levels by Year**")
            display_df2 = summary_df[['year', 'threshold_99', 'max_level']].copy()
            display_df2['threshold_99'] = display_df2['threshold_99'].round(2)
            display_df2['max_level'] = display_df2['max_level'].round(2)
            display_df2.columns = ['Year', '99th Percentile (m)', 'Max Level (m)']
            st.dataframe(display_df2, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("**Key Insights**")
        
        total_events = summary_df['count'].sum()
        total_monsoon = summary_df['monsoon_events'].sum()
        total_non_monsoon = summary_df['non_monsoon_events'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Extreme Events", total_events)
        col2.metric("During Monsoon", total_monsoon, f"{total_monsoon/total_events*100:.0f}%")
        col3.metric("Outside Monsoon", total_non_monsoon, f"{total_non_monsoon/total_events*100:.0f}%")
        col4.metric("Avg per Year", f"{total_events/len(years_available):.1f}")
        
        st.markdown("---")
        st.markdown("""
        **How to read this chart:**
        - Each point represents a day when water level was in the **top 1%** for that year
        - Clustering of points shows **when flash floods typically occur**
        - Points outside the blue monsoon zone indicate **unusual flash flood periods**
        - Use this to identify **seasonal patterns** and **unexpected flood timing**
        """)
    else:
        st.warning("No data available for flash flood analysis")