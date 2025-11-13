"""
FLOOD RISK INTELLIGENCE PLATFORM FOR NEPAL - WITH REAL ERA5 DATA
Uses ERA5 Climate Reanalysis data from Copernicus for Nepal river basins
WITH RIVER BACKGROUND IMAGE AND FLOOD TIMING ANALYSIS
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import pickle
import hashlib
import base64
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

#==============================================================================
# CONFIGURATION
#==============================================================================

class Config:
    """System configuration for Nepal flood monitoring"""
    
    LSTM_HIDDEN = 128
    LSTM_LAYERS = 2
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 50
    LOOKBACK_HOURS = 168
    FORECAST_HOURS = 168
    SAMPLING_INTERVAL = 6
    YEARS_OF_HISTORY = 10
    
    # Cache settings
    CACHE_DIR = ".era5_cache"
    CACHE_EXPIRY_DAYS = 1
    
    # River images
    RIVER_IMAGES = {
        'bagmati_khokana': 'https://images.unsplash.com/photo-1626278664285-f796b9ee7806?w=1200&q=80',
        'koshi_chatara': 'https://images.unsplash.com/photo-1542401886-65d6c61db217?w=1200&q=80',
        'narayani_narayanghat': 'https://images.unsplash.com/photo-1520645992065-6f1c4b6a7041?w=1200&q=80',
        'karnali_chisapani': 'https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=1200&q=80',
        'kankai_mainachuli': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&q=80'
    }
    
    TARGET_STATIONS = {
        'bagmati_khokana': {
            'station_id': 'DHM_330',
            'name': 'Bagmati River at Khokana (Kathmandu Valley)',
            'river': 'Bagmati',
            'district': 'Lalitpur',
            'drainage_area': 678,
            'flood_stage': 2.5,
            'moderate_flood': 3.5,
            'major_flood': 4.5,
            'lat': 27.63,
            'lon': 85.29,
            'elevation': 1300
        },
        'koshi_chatara': {
            'station_id': 'DHM_695',
            'name': 'Koshi River at Chatara',
            'river': 'Koshi (Sapta Koshi)',
            'district': 'Sunsari',
            'drainage_area': 54100,
            'flood_stage': 4.0,
            'moderate_flood': 5.5,
            'major_flood': 7.0,
            'lat': 26.87,
            'lon': 87.17,
            'elevation': 150
        },
        'narayani_narayanghat': {
            'station_id': 'DHM_425',
            'name': 'Narayani River at Narayanghat',
            'river': 'Narayani (Gandaki)',
            'district': 'Chitwan',
            'drainage_area': 31100,
            'flood_stage': 4.5,
            'moderate_flood': 6.0,
            'major_flood': 7.5,
            'lat': 27.70,
            'lon': 84.43,
            'elevation': 200
        },
        'karnali_chisapani': {
            'station_id': 'DHM_560',
            'name': 'Karnali River at Chisapani',
            'river': 'Karnali',
            'district': 'Bardiya',
            'drainage_area': 42890,
            'flood_stage': 4.0,
            'moderate_flood': 5.5,
            'major_flood': 7.0,
            'lat': 28.64,
            'lon': 81.26,
            'elevation': 180
        },
        'kankai_mainachuli': {
            'station_id': 'DHM_680',
            'name': 'Kankai River at Mainachuli',
            'river': 'Kankai',
            'district': 'Jhapa',
            'drainage_area': 1148,
            'flood_stage': 2.0,
            'moderate_flood': 3.0,
            'major_flood': 4.0,
            'lat': 26.68,
            'lon': 87.68,
            'elevation': 100
        }
    }


#==============================================================================
# CACHE MANAGER
#==============================================================================

class CacheManager:
    """Manage ERA5 data and model caching"""
    
    def __init__(self, cache_dir=Config.CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, station_id, years_back):
        key_str = f"{station_id}_{years_back}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key):
        return os.path.join(self.cache_dir, f"era5_{cache_key}.pkl")
    
    def _get_model_cache_path(self, station_id):
        return os.path.join(self.cache_dir, f"model_{station_id}.pt")
    
    def _get_dataset_cache_path(self, station_id):
        return os.path.join(self.cache_dir, f"dataset_{station_id}.pkl")
    
    def is_cache_valid(self, station_id, years_back):
        cache_key = self._get_cache_key(station_id, years_back)
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return False
        
        cache_time = os.path.getmtime(cache_path)
        cache_age = (datetime.now().timestamp() - cache_time) / 86400
        
        return cache_age < Config.CACHE_EXPIRY_DAYS
    
    def is_model_cache_valid(self, station_id):
        model_path = self._get_model_cache_path(station_id)
        dataset_path = self._get_dataset_cache_path(station_id)
        
        if not os.path.exists(model_path) or not os.path.exists(dataset_path):
            return False
        
        cache_time = os.path.getmtime(model_path)
        cache_age = (datetime.now().timestamp() - cache_time) / 86400
        
        return cache_age < Config.CACHE_EXPIRY_DAYS
    
    def load_from_cache(self, station_id, years_back):
        cache_key = self._get_cache_key(station_id, years_back)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded data from cache: {cache_path}")
            return data
        except Exception as e:
            print(f"Cache load failed: {e}")
            return None
    
    def save_to_cache(self, data, station_id, years_back):
        cache_key = self._get_cache_key(station_id, years_back)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved data to cache: {cache_path}")
        except Exception as e:
            print(f"Cache save failed: {e}")
    
    def load_model_from_cache(self, station_id):
        model_path = self._get_model_cache_path(station_id)
        dataset_path = self._get_dataset_cache_path(station_id)
        
        try:
            forecaster = FloodForecaster()
            forecaster.model.load_state_dict(torch.load(model_path, map_location=forecaster.device))
            forecaster.is_trained = True
            
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            
            print(f"Loaded model from cache: {model_path}")
            return forecaster, dataset
        except Exception as e:
            print(f"Model cache load failed: {e}")
            return None, None
    
    def save_model_to_cache(self, forecaster, dataset, station_id):
        model_path = self._get_model_cache_path(station_id)
        dataset_path = self._get_dataset_cache_path(station_id)
        
        try:
            torch.save(forecaster.model.state_dict(), model_path)
            with open(dataset_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"Saved model to cache: {model_path}")
        except Exception as e:
            print(f"Model cache save failed: {e}")
    
    def clear_cache(self):
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
        print("Cache cleared")


#==============================================================================
# ERA5 DATA LOADER WITH CACHING
#==============================================================================

class ERA5DataLoader:
    """Fetch ERA5 data with smart caching"""
    
    def __init__(self, station_config, years_back=10):
        self.station_config = station_config
        self.years_back = years_back
        self.lat = station_config['lat']
        self.lon = station_config['lon']
        self.cache_manager = CacheManager()
        
    def fetch_era5_data(self):
        station_id = self.station_config['station_id']
        
        # Check cache first
        if self.cache_manager.is_cache_valid(station_id, self.years_back):
            st.info("Loading from cache (refreshes daily)...")
            data = self.cache_manager.load_from_cache(station_id, self.years_back)
            if data is not None:
                st.success(f"Loaded {len(data):,} records from cache")
                return data
        
        # Check API config
        if not self._check_cds_api():
            st.warning("ERA5 API not configured. Using synthetic data.")
            data = self._generate_realistic_synthetic_data()
            self.cache_manager.save_to_cache(data, station_id, self.years_back)
            return data
        
        # Fetch from API
        try:
            import cdsapi
            
            st.info("Fetching ERA5 data from Copernicus (will cache for 24h)...")
            
            c = cdsapi.Client(timeout=600, retry_max=5)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 5)
            
            north = min(self.lat + 0.25, 90)
            south = max(self.lat - 0.25, -90)
            east = min(self.lon + 0.25, 180)
            west = max(self.lon - 0.25, -180)
            
            all_data = []
            
            for year in range(start_date.year, end_date.year + 1):
                try:
                    st.text(f"Fetching year {year}...")
                    
                    if year == start_date.year:
                        start_month = start_date.month
                        months = [f"{m:02d}" for m in range(start_month, 13)]
                    elif year == end_date.year:
                        end_month = end_date.month
                        months = [f"{m:02d}" for m in range(1, end_month + 1)]
                    else:
                        months = [f"{m:02d}" for m in range(1, 13)]
                    
                    output_file = f"era5_nepal_{year}_{self.station_config['station_id']}.nc"
                    
                    c.retrieve(
                        'reanalysis-era5-single-levels',
                        {
                            'product_type': 'reanalysis',
                            'variable': ['total_precipitation', 'runoff'],
                            'year': str(year),
                            'month': months,
                            'day': [f"{d:02d}" for d in range(1, 32)],
                            'time': ['00:00', '06:00', '12:00', '18:00'],
                            'area': [north, west, south, east],
                            'format': 'netcdf'
                        },
                        output_file
                    )
                    
                    year_data = self._process_era5_netcdf(output_file)
                    all_data.append(year_data)
                    
                    if os.path.exists(output_file):
                        os.remove(output_file)
                    
                except Exception as e:
                    st.warning(f"Failed to fetch year {year}: {str(e)}")
                    continue
            
            if not all_data:
                st.warning("Could not fetch ERA5 data. Using synthetic.")
                data = self._generate_realistic_synthetic_data()
                self.cache_manager.save_to_cache(data, station_id, self.years_back)
                return data
            
            data = pd.concat(all_data, ignore_index=True)
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            data = data.set_index('timestamp')
            data = data.resample('1H').interpolate(method='linear')
            data = data.reset_index()
            
            data['year'] = data['timestamp'].dt.year
            data['month'] = data['timestamp'].dt.month
            data['day_of_year'] = data['timestamp'].dt.dayofyear
            data['season'] = data['month'].apply(lambda x:
                'Winter' if x in [12, 1, 2] else
                'Pre-Monsoon' if x in [3, 4, 5] else
                'Monsoon' if x in [6, 7, 8, 9] else 'Post-Monsoon'
            )
            
            self.cache_manager.save_to_cache(data, station_id, self.years_back)
            
            st.success(f"Fetched and cached {len(data):,} hours of ERA5 data")
            return data
            
        except ImportError:
            st.warning("cdsapi not installed. Run: pip install cdsapi xarray netCDF4")
            data = self._generate_realistic_synthetic_data()
            self.cache_manager.save_to_cache(data, station_id, self.years_back)
            return data
        
        except Exception as e:
            st.warning(f"ERA5 fetch failed: {str(e)}")
            data = self._generate_realistic_synthetic_data()
            self.cache_manager.save_to_cache(data, station_id, self.years_back)
            return data
    
    def _check_cds_api(self):
        cdsapirc = os.path.expanduser("~/.cdsapirc")
        return os.path.exists(cdsapirc)
    
    def _process_era5_netcdf(self, filename):
        import xarray as xr
        
        ds = xr.open_dataset(filename)
        
        time_var = 'valid_time' if 'valid_time' in ds.dims else 'time'
        
        precip = ds['tp'].sel(latitude=self.lat, longitude=self.lon, method='nearest')
        runoff = ds['ro'].sel(latitude=self.lat, longitude=self.lon, method='nearest')
        
        if hasattr(precip, time_var):
            time_values = getattr(precip, time_var).values
        elif 'valid_time' in precip.coords:
            time_values = precip.coords['valid_time'].values
        elif 'time' in precip.coords:
            time_values = precip.coords['time'].values
        else:
            time_values = ds[time_var].values
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(time_values),
            'precipitation_m': precip.values.flatten(),
            'runoff_m': runoff.values.flatten()
        })
        
        df = df.dropna()
        df['precipitation_mm'] = df['precipitation_m'] * 1000
        df = self._calculate_discharge_from_era5(df)
        
        ds.close()
        
        return df
    
    def _calculate_discharge_from_era5(self, df):
        area_m2 = self.station_config['drainage_area'] * 1e6
        
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        df['runoff_coef'] = df['month'].apply(lambda m: 0.75 if m in [6,7,8,9] else 0.45)
        
        timestep_seconds = 6 * 3600
        
        df['discharge_cumecs'] = (
            (df['precipitation_m'] * area_m2 * df['runoff_coef']) / timestep_seconds +
            (df['runoff_m'] * area_m2 / timestep_seconds)
        )
        
        baseflow = 15 + (self.station_config['drainage_area'] * 0.02)
        df['discharge_cumecs'] = df['discharge_cumecs'] + baseflow
        
        df['discharge_cumecs'] = df['discharge_cumecs'].rolling(window=4, min_periods=1).mean()
        
        # DEBUG: Print discharge statistics
        print(f"\n=== DISCHARGE STATISTICS ===")
        print(f"Discharge range: {df['discharge_cumecs'].min():.2f} - {df['discharge_cumecs'].max():.2f} m3/s")
        print(f"Discharge mean: {df['discharge_cumecs'].mean():.2f} m3/s")
        print(f"Discharge 90th percentile: {df['discharge_cumecs'].quantile(0.90):.2f} m3/s")
        
        # Scale rating curve parameters based on drainage area and flood stage
        drainage_area = self.station_config['drainage_area']
        flood_stage = self.station_config['flood_stage']
        
        # Calculate 90th percentile of discharge for calibration
        q90 = df['discharge_cumecs'].quantile(0.90)
        
        # Rating curve: h = h0 + (Q/a)^(1/b)
        # We want h = flood_stage when Q = q90
        h0 = 0.3
        b = 0.55
        # Solve for a: flood_stage = h0 + (q90/a)^(1/b)
        # => a = q90 / ((flood_stage - h0)^b)
        a = q90 / ((flood_stage - h0) ** b)
        
        print(f"\n=== RATING CURVE PARAMETERS ===")
        print(f"Station: {self.station_config['name']}")
        print(f"Flood stage threshold: {flood_stage:.2f} m")
        print(f"Q90 (for calibration): {q90:.2f} m3/s")
        print(f"Rating curve: h = {h0} + (Q/{a:.2f})^(1/{b})")
        
        df['water_level_m'] = h0 + (df['discharge_cumecs'] / a) ** (1/b)
        
        # DEBUG: Print water level statistics
        print(f"\n=== WATER LEVEL STATISTICS ===")
        print(f"Water level range: {df['water_level_m'].min():.2f} - {df['water_level_m'].max():.2f} m")
        print(f"Water level mean: {df['water_level_m'].mean():.2f} m")
        print(f"Water level 90th percentile: {df['water_level_m'].quantile(0.90):.2f} m")
        print(f"Records above flood stage ({flood_stage:.2f}m): {(df['water_level_m'] > flood_stage).sum()}")
        print("=" * 50)
        
        return df
    
    def _generate_realistic_synthetic_data(self):
        print(f"Generating {self.years_back}-year synthetic data...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.years_back)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        days_since_start = np.arange(len(timestamps)) / 24
        day_of_year = (days_since_start % 365)
        
        base_flow = 25 + (self.station_config['drainage_area'] * 0.025)
        seasonal_factor = np.ones(len(timestamps))
        
        for i, doy in enumerate(day_of_year):
            if 150 <= doy <= 273:
                monsoon_intensity = np.sin((doy - 150) * np.pi / 123)
                seasonal_factor[i] = 10.0 + 6.0 * monsoon_intensity
            elif 60 <= doy < 150:
                seasonal_factor[i] = 2.5 + np.random.uniform(0, 1.5)
            elif 273 < doy <= 305:
                seasonal_factor[i] = 3.5 + np.random.uniform(0, 1.5)
            else:
                seasonal_factor[i] = 1.0 + np.random.uniform(0, 0.5)
        
        discharge = base_flow * seasonal_factor * (1 + np.random.normal(0, 0.2, len(timestamps)))
        
        flood_events = []
        for year in range(self.years_back):
            year_start_idx = int(year * 365 * 24)
            
            if np.random.random() < 0.5:
                day = np.random.randint(60, 150)
                mag = base_flow * np.random.uniform(4, 10)
                flood_events.append((year_start_idx + day * 24, mag, np.random.randint(8, 15)))
            
            num_floods = np.random.randint(4, 9)
            for _ in range(num_floods):
                day = np.random.randint(150, 273)
                mag = base_flow * np.random.uniform(5, 18)
                flood_events.append((year_start_idx + day * 24, mag, np.random.randint(10, 25)))
            
            if np.random.random() < 0.6:
                day = np.random.randint(273, 305)
                mag = base_flow * np.random.uniform(5, 9)
                flood_events.append((year_start_idx + day * 24, mag, np.random.randint(6, 12)))
        
        print(f"Simulated {len(flood_events)} flood events")
        
        for peak_idx, magnitude, duration_days in flood_events:
            duration_steps = duration_days * 24
            start_idx = max(0, peak_idx - int(duration_steps * 0.3))
            end_idx = min(len(timestamps), peak_idx + int(duration_steps * 0.7))
            
            for i in range(start_idx, peak_idx):
                if i >= len(discharge):
                    break
                x = (i - start_idx) / (peak_idx - start_idx) if peak_idx > start_idx else 0
                discharge[i] += magnitude * (x ** 1.8)
            
            for i in range(peak_idx, end_idx):
                if i >= len(discharge):
                    break
                x = (i - peak_idx) / (end_idx - peak_idx) if end_idx > peak_idx else 0
                discharge[i] += magnitude * np.exp(-2.5 * x)
        
        discharge = np.maximum(discharge, base_flow * 0.6)
        
        # Scale rating curve parameters based on drainage area and flood stage
        flood_stage = self.station_config['flood_stage']
        
        # Calculate 90th percentile of discharge for calibration
        q90 = np.percentile(discharge, 90)
        
        # Rating curve: h = h0 + (Q/a)^(1/b)
        h0 = 0.3
        b = 0.55
        # Calibrate 'a' so that q90 corresponds to flood_stage
        a = q90 / ((flood_stage - h0) ** b)
        
        water_level = h0 + (discharge / a) ** (1/b)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'discharge_cumecs': discharge,
            'water_level_m': water_level,
            'year': timestamps.year,
            'month': timestamps.month,
            'day_of_year': timestamps.dayofyear
        })
        
        df['season'] = df['month'].apply(lambda x:
            'Winter' if x in [12, 1, 2] else
            'Pre-Monsoon' if x in [3, 4, 5] else
            'Monsoon' if x in [6, 7, 8, 9] else 'Post-Monsoon'
        )
        
        print(f"Generated {len(df):,} hourly records")
        print(f"Water level range: {df['water_level_m'].min():.2f}m - {df['water_level_m'].max():.2f}m")
        print(f"Flood stage threshold: {flood_stage:.2f}m")
        print(f"Records above flood stage: {(df['water_level_m'] > flood_stage).sum()}")
        
        return df


#==============================================================================
# ANALYTICS
#==============================================================================

class FloodAnalytics:
    """Analytics with improved flood detection"""
    
    def __init__(self, data, station_config):
        self.data = data.copy()
        self.station_config = station_config
        self.flood_threshold = station_config['flood_stage']
        
        if 'year' not in self.data.columns:
            self.data['year'] = self.data['timestamp'].dt.year
        if 'month' not in self.data.columns:
            self.data['month'] = self.data['timestamp'].dt.month
        if 'season' not in self.data.columns:
            self.data['season'] = self.data['month'].apply(lambda x:
                'Winter' if x in [12, 1, 2] else
                'Pre-Monsoon' if x in [3, 4, 5] else
                'Monsoon' if x in [6, 7, 8, 9] else 'Post-Monsoon'
            )
    
    def identify_flood_events(self):
        flood_mask = self.data['water_level_m'] > self.flood_threshold
        flood_periods = []
        in_flood = False
        flood_start = None
        flood_peak = 0
        flood_start_idx = None
        
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            
            if flood_mask.iloc[idx] and not in_flood:
                in_flood = True
                flood_start = row['timestamp']
                flood_start_idx = idx
                flood_peak = row['water_level_m']
                
            elif flood_mask.iloc[idx] and in_flood:
                flood_peak = max(flood_peak, row['water_level_m'])
                
            elif not flood_mask.iloc[idx] and in_flood:
                duration_hours = (row['timestamp'] - flood_start).total_seconds() / 3600
                
                if duration_hours >= 6:
                    flood_periods.append({
                        'start': flood_start,
                        'end': row['timestamp'],
                        'duration_hours': duration_hours,
                        'peak_level': flood_peak,
                        'severity': self._classify_severity(flood_peak),
                        'year': flood_start.year,
                        'month': flood_start.month,
                        'season': self.data.iloc[flood_start_idx]['season']
                    })
                
                in_flood = False
        
        result_df = pd.DataFrame(flood_periods) if flood_periods else pd.DataFrame()
        
        if not result_df.empty:
            print(f"Detected {len(result_df)} flood events")
        
        return result_df
    
    def _classify_severity(self, level):
        if level >= self.station_config['major_flood']:
            return 'Major'
        elif level >= self.station_config['moderate_flood']:
            return 'Moderate'
        else:
            return 'Minor'
    
    def seasonal_risk_profile(self):
        flood_events = self.identify_flood_events()
        if flood_events.empty:
            return pd.DataFrame({
                'season': ['Winter', 'Pre-Monsoon', 'Monsoon', 'Post-Monsoon'],
                'flood_count': [0, 0, 0, 0],
                'risk_score': [0, 0, 0, 0]
            })
        
        seasonal_stats = flood_events.groupby('season').agg({
            'peak_level': ['count', 'mean'],
        }).reset_index()
        seasonal_stats.columns = ['season', 'flood_count', 'avg_peak']
        
        max_count = seasonal_stats['flood_count'].max()
        if max_count > 0:
            seasonal_stats['risk_score'] = (
                (seasonal_stats['flood_count'] / max_count) * 50 +
                ((seasonal_stats['avg_peak'] - self.flood_threshold) /
                 (self.station_config['major_flood'] - self.flood_threshold)) * 50
            )
        else:
            seasonal_stats['risk_score'] = 0
        
        all_seasons = pd.DataFrame({'season': ['Winter', 'Pre-Monsoon', 'Monsoon', 'Post-Monsoon']})
        seasonal_stats = all_seasons.merge(seasonal_stats, on='season', how='left').fillna(0)
        
        return seasonal_stats.sort_values('risk_score', ascending=False)
    
    def year_over_year_comparison(self):
        yearly_stats = self.data.groupby('year').agg({
            'discharge_cumecs': ['mean', 'max'],
            'water_level_m': ['mean', 'max']
        }).reset_index()
        yearly_stats.columns = ['year', 'avg_discharge', 'peak_discharge', 'avg_level', 'peak_level']
        
        flood_events = self.identify_flood_events()
        if not flood_events.empty:
            flood_days = flood_events.groupby('year')['duration_hours'].sum() / 24
            yearly_stats = yearly_stats.merge(
                flood_days.reset_index().rename(columns={'duration_hours': 'flood_days'}),
                on='year', how='left'
            )
            yearly_stats['flood_days'] = yearly_stats['flood_days'].fillna(0)
        else:
            yearly_stats['flood_days'] = 0
        
        return yearly_stats
    
    def get_current_vs_historical(self):
        current = self.data.iloc[-1]
        current_month = current['month']
        historical_same_month = self.data[self.data['month'] == current_month]
        
        percentile_discharge = (
            (historical_same_month['discharge_cumecs'] < current['discharge_cumecs']).sum() /
            len(historical_same_month) * 100
        )
        
        return {
            'current_discharge': current['discharge_cumecs'],
            'current_level': current['water_level_m'],
            'percentile_discharge': percentile_discharge,
            'historical_avg_discharge': historical_same_month['discharge_cumecs'].mean(),
        }


#==============================================================================
# LSTM MODEL
#==============================================================================

class FloodDataset(Dataset):
    def __init__(self, data, lookback=168, forecast=168, stride=6):
        self.lookback_steps = lookback // stride
        self.forecast_steps = forecast // stride
        
        data = data.copy()
        data['discharge_cumecs'] = data['discharge_cumecs'].ffill().bfill()
        data['water_level_m'] = data['water_level_m'].ffill().bfill()
        
        self.discharge = data['discharge_cumecs'].values.astype(np.float32)
        self.water_level = data['water_level_m'].values.astype(np.float32)
        
        self.discharge_mean = self.discharge.mean()
        self.discharge_std = self.discharge.std()
        self.level_mean = self.water_level.mean()
        self.level_std = self.water_level.std()
        
        self.discharge_norm = (self.discharge - self.discharge_mean) / (self.discharge_std + 1e-8)
        self.level_norm = (self.water_level - self.level_mean) / (self.level_std + 1e-8)
        
        self.sequences = []
        self.targets = []
        
        for i in range(0, len(self.discharge) - self.lookback_steps - self.forecast_steps, stride):
            seq = np.stack([
                self.discharge_norm[i:i+self.lookback_steps],
                self.level_norm[i:i+self.lookback_steps]
            ], axis=-1)
            
            target = np.stack([
                self.discharge_norm[i+self.lookback_steps:i+self.lookback_steps+self.forecast_steps],
                self.level_norm[i+self.lookback_steps:i+self.lookback_steps+self.forecast_steps]
            ], axis=-1)
            
            self.sequences.append(seq)
            self.targets.append(target)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx]))
    
    def inverse_transform(self, normalized_data):
        result = normalized_data.copy()
        result[..., 0] = normalized_data[..., 0] * self.discharge_std + self.discharge_mean
        result[..., 1] = normalized_data[..., 1] * self.level_std + self.level_mean
        return result


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2 * 28)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.view(x.size(0), 28, 2)


class FloodForecaster:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.is_trained = False
        
    def train_silent(self, train_loader, val_loader, epochs=30):
        best_val_loss = float('inf')
        patience = 0
        
        for epoch in range(epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = nn.MSELoss()(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    predictions = self.model(batch_x)
                    val_loss += nn.MSELoss()(predictions, batch_y).item()
            
            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    break
        
        self.is_trained = True
        return best_val_loss
    
    def predict(self, input_sequence, dataset=None):
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)
            prediction = self.model(input_tensor).cpu().numpy()[0]
        
        if dataset:
            prediction = dataset.inverse_transform(prediction)
        return prediction


#==============================================================================
# STREAMLIT APP WITH RIVER BACKGROUND
#==============================================================================

def main():
    st.set_page_config(page_title="Nepal Flood Intelligence", layout="wide", page_icon="🇳🇵")
    
    # Convert image to base64 for embedding
    def get_base64_image(image_path):
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except:
            return None
    
    bg_image = get_base64_image("river1.jpg")
    
    background_style = ""
    if bg_image:
        background_style = f"""
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), 
                              url("data:image/jpg;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        """
    
    st.markdown(f"""
        <style>
        {background_style}
        .main-header {{
            font-size: 2.5rem;
            font-weight: 800;
            text-align: center;
            background: linear-gradient(135deg, #dc143c 0%, #003893 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 1rem 0;
        }}
        .executive-card {{
            background: rgba(248, 250, 252, 0.95);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #dc143c;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
        }}
        .river-banner {{
            width: 100%;
            height: 250px;
            object-fit: cover;
            border-radius: 15px;
            margin-bottom: 2rem;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
            backdrop-filter: blur(10px);
        }}
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] strong {{
            color: white !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] span {{
            color: white !important;
        }}
        [data-testid="stSidebar"] button {{
            background: linear-gradient(135deg, #dc143c 0%, #b91c1c 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
        }}
        [data-testid="stSidebar"] button:hover {{
            background: linear-gradient(135deg, #b91c1c 0%, #991b1b 100%) !important;
            transform: scale(1.02);
        }}
        /* Make content cards semi-transparent for background visibility */
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {{
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 1rem;
            backdrop-filter: blur(5px);
        }}
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">🇳🇵 Nepal Flood Risk Intelligence Platform</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#64748b">Real ERA5 Climate Data with Smart Caching</p>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    
    st.sidebar.header("Configuration")
    station_key = st.sidebar.selectbox(
        "Select Station",
        list(Config.TARGET_STATIONS.keys()),
        format_func=lambda x: Config.TARGET_STATIONS[x]['name']
    )
    
    station_config = Config.TARGET_STATIONS[station_key]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Station Info")
    st.sidebar.markdown(f"**River:** {station_config['river']}")
    st.sidebar.markdown(f"**District:** {station_config['district']}")
    st.sidebar.markdown(f"**Drainage:** {station_config['drainage_area']:,} km2")
    st.sidebar.markdown(f"**Coords:** {station_config['lat']:.2f}N, {station_config['lon']:.2f}E")
    
    st.sidebar.markdown("---")
    
    cache_manager = CacheManager()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Clear Cache", help="Clear all cached data and models", use_container_width=True):
            cache_manager.clear_cache()
            st.session_state.clear()
            st.success("Cache cleared!")
            st.rerun()
    
    with col2:
        if st.button("Force Refresh", help="Force complete data reload", use_container_width=True):
            cache_manager.clear_cache()
            st.session_state.clear()
            st.success("Ready for fresh load!")
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Load Data (5 Years)", use_container_width=True):
        with st.spinner("Loading data..."):
            loader = ERA5DataLoader(station_config, 5)
            data = loader.fetch_era5_data()
            
            if data is not None and len(data) > 1000:
                st.session_state['data'] = data
                st.session_state['station_config'] = station_config
                st.session_state['analytics'] = FloodAnalytics(data, station_config)
                
                # Check if model is cached
                if cache_manager.is_model_cache_valid(station_config['station_id']):
                    st.info("Loading pre-trained model from cache...")
                    forecaster, dataset = cache_manager.load_model_from_cache(station_config['station_id'])
                    if forecaster is not None and dataset is not None:
                        st.session_state['forecaster'] = forecaster
                        st.session_state['dataset'] = dataset
                        st.success("Model loaded from cache!")
                    else:
                        st.warning("Model cache invalid, will train on first forecast request")
                else:
                    # Train model in background
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
                            
                            # Cache the trained model
                            cache_manager.save_model_to_cache(forecaster, dataset, station_config['station_id'])
                            
                            st.session_state['forecaster'] = forecaster
                            st.session_state['dataset'] = dataset
                            st.success("Model trained and cached!")
                        except Exception as e:
                            st.warning(f"Model training skipped: {str(e)}")
                
                st.sidebar.success(f"Loaded {len(data):,} records")
                st.rerun()
    
    if st.session_state['data'] is not None:
        data = st.session_state['data']
        analytics = st.session_state['analytics']
        station_config = st.session_state['station_config']
        
        cache_age_hours = 0
        cache_key = cache_manager._get_cache_key(station_config['station_id'], 5)
        cache_path = cache_manager._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            cache_time = os.path.getmtime(cache_path)
            cache_age_hours = (datetime.now().timestamp() - cache_time) / 3600
        
        st.info(f"Data Source: ERA5 Climate Reanalysis | Cache Age: {cache_age_hours:.1f} hours (refreshes daily)")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Historical Analysis", "Monsoon Risk", "Forecast", "Flood Timing", "Flash Flood Analysis"])
        
        with tab1:
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
        
        with tab2:
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
        
        with tab3:
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
        
        with tab4:
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
        
        with tab5:
            st.subheader("Flood Timing Analysis - Multi-Year Comparison")
            
            st.info("This view overlays all years to reveal seasonal flood patterns and identify unusual non-monsoon floods")
            st.caption("⭐ Red-bordered star markers indicate unusual flood events (outside monsoon season) over the last 5 years")
            
            # Prepare data for plotting
            years_available = sorted(data['year'].unique())
            
            if len(years_available) > 0:
                fig = go.Figure()
                
                # Color palette for different years
                colors = ['#dc143c', '#003893', '#f59e0b', '#10b981', '#8b5cf6']
                
                # Track unusual floods for summary
                unusual_floods_summary = []
                
                for idx, year in enumerate(years_available):
                    year_data = data[data['year'] == year].copy()
                    year_data = year_data.sort_values('day_of_year')
                    
                    color = colors[idx % len(colors)]
                    
                    # Plot the main water level line for this year
                    fig.add_trace(go.Scatter(
                        x=year_data['day_of_year'],
                        y=year_data['water_level_m'],
                        mode='lines',
                        name=f'{year}',
                        line=dict(color=color, width=2),
                        hovertemplate=f'<b>{year}</b><br>Day: %{{x}}<br>Level: %{{y:.2f}} m<extra></extra>'
                    ))
                    
                    # Identify unusual floods (outside monsoon: June-September = months 6,7,8,9)
                    # Monsoon months correspond roughly to days 152-273 (June 1 - Sept 30)
                    unusual_flood_mask = (
                        (year_data['water_level_m'] > station_config['flood_stage']) &
                        ((year_data['day_of_year'] < 152) | (year_data['day_of_year'] > 273))
                    )
                    
                    unusual_floods = year_data[unusual_flood_mask]
                    
                    if len(unusual_floods) > 0:
                        # Add markers for unusual flood events
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
                            showlegend=(idx == 0)  # Only show legend for first occurrence
                        ))
                        
                        # Track for summary
                        unusual_count = len(unusual_floods)
                        max_unusual_level = unusual_floods['water_level_m'].max()
                        unusual_floods_summary.append({
                            'year': year,
                            'count': unusual_count,
                            'max_level': max_unusual_level
                        })
                
                # Add flood stage reference line
                fig.add_hline(
                    y=station_config['flood_stage'],
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="Flood Stage",
                    annotation_position="right"
                )
                
                # Add moderate flood reference line
                fig.add_hline(
                    y=station_config['moderate_flood'],
                    line_dash="dot",
                    line_color="red",
                    annotation_text="Moderate Flood",
                    annotation_position="right"
                )
                
                # Add shaded region for monsoon season
                fig.add_vrect(
                    x0=152, x1=273,
                    fillcolor="lightblue",
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                    annotation_text="Monsoon Season",
                    annotation_position="top left"
                )
                
                fig.update_layout(
                    title="Water Level Throughout the Year - All Years Overlaid",
                    xaxis_title="Day of Year",
                    yaxis_title="Water Level (m)",
                    height=600,
                    hovermode='closest',
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(255,255,255,0.8)"
                    )
                )
                
                # Add month labels on x-axis
                month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                fig.update_xaxes(
                    tickmode='array',
                    tickvals=month_starts,
                    ticktext=month_names
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Summary Statistics")
                    st.markdown(f"**Years Analyzed:** {len(years_available)}")
                    st.markdown(f"**Flood Threshold:** {station_config['flood_stage']:.2f} m")
                    st.markdown(f"**Monsoon Period:** Day 152-273 (June-Sept)")
                
                with col2:
                    st.markdown("### ⚠️ Unusual Floods (Non-Monsoon)")
                    if unusual_floods_summary:
                        for item in unusual_floods_summary:
                            st.markdown(f"**{item['year']}:** {item['count']} events, peak {item['max_level']:.2f} m")
                    else:
                        st.markdown("No unusual non-monsoon floods detected")
                
                # Legend explanation
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
        
        with tab6:
            st.subheader("Flash Flood Analysis - Top 5% Extreme Events")
            
            st.info("Identifies when extreme water levels (top 5%) occur throughout the year across different years")
            st.caption("Each point represents a top 5% water level event - helps identify flash flood timing patterns")
            
            # Prepare data for plotting
            years_available = sorted(data['year'].unique())
            
            if len(years_available) > 0:
                fig = go.Figure()
                
                # Color palette for different years
                colors = ['#dc143c', '#003893', '#f59e0b', '#10b981', '#8b5cf6']
                
                # Track statistics
                flash_flood_summary = []
                
                for idx, year in enumerate(years_available):
                    year_data = data[data['year'] == year].copy()
                    
                    # Calculate 95th percentile (top 5%)
                    percentile_95 = year_data['water_level_m'].quantile(0.95)
                    
                    # Filter only top 5% events
                    top_5_percent = year_data[year_data['water_level_m'] >= percentile_95].copy()
                    top_5_percent = top_5_percent.sort_values('day_of_year')
                    
                    color = colors[idx % len(colors)]
                    
                    # Plot scatter points for top 5% events
                    fig.add_trace(go.Scatter(
                        x=top_5_percent['day_of_year'],
                        y=top_5_percent['water_level_m'],
                        mode='markers',
                        name=f'{year}',
                        marker=dict(
                            color=color,
                            size=8,
                            opacity=0.7,
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate=f'<b>{year}</b><br>Day: %{{x}}<br>Level: %{{y:.2f}} m<extra></extra>'
                    ))
                    
                    # Track statistics
                    flash_flood_summary.append({
                        'year': year,
                        'count': len(top_5_percent),
                        'threshold_95': percentile_95,
                        'max_level': top_5_percent['water_level_m'].max(),
                        'monsoon_events': len(top_5_percent[(top_5_percent['day_of_year'] >= 152) & (top_5_percent['day_of_year'] <= 273)]),
                        'non_monsoon_events': len(top_5_percent[(top_5_percent['day_of_year'] < 152) | (top_5_percent['day_of_year'] > 273)])
                    })
                
                # Add flood stage reference line
                fig.add_hline(
                    y=station_config['flood_stage'],
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="Flood Stage",
                    annotation_position="right"
                )
                
                # Add moderate flood reference line
                fig.add_hline(
                    y=station_config['moderate_flood'],
                    line_dash="dot",
                    line_color="red",
                    annotation_text="Moderate Flood",
                    annotation_position="right"
                )
                
                # Add shaded region for monsoon season
                fig.add_vrect(
                    x0=152, x1=273,
                    fillcolor="lightblue",
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                    annotation_text="Monsoon Season",
                    annotation_position="top left"
                )
                
                fig.update_layout(
                    title="Top 5% Extreme Water Levels - When Do Flash Floods Occur?",
                    xaxis_title="Day of Year",
                    yaxis_title="Water Level (m)",
                    height=600,
                    hovermode='closest',
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(255,255,255,0.8)"
                    )
                )
                
                # Add month labels on x-axis
                month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                fig.update_xaxes(
                    tickmode='array',
                    tickvals=month_starts,
                    ticktext=month_names
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.markdown("---")
                st.markdown("### 📊 Flash Flood Event Summary (Top 5% Extremes)")
                
                # Create summary dataframe
                summary_df = pd.DataFrame(flash_flood_summary)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Event Counts by Year")
                    display_df = summary_df[['year', 'count', 'monsoon_events', 'non_monsoon_events']].copy()
                    display_df.columns = ['Year', 'Total Events', 'Monsoon', 'Non-Monsoon']
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("#### Extreme Levels by Year")
                    display_df2 = summary_df[['year', 'threshold_95', 'max_level']].copy()
                    display_df2['threshold_95'] = display_df2['threshold_95'].round(2)
                    display_df2['max_level'] = display_df2['max_level'].round(2)
                    display_df2.columns = ['Year', '95th Percentile (m)', 'Max Level (m)']
                    st.dataframe(display_df2, use_container_width=True, hide_index=True)
                
                # Key insights
                st.markdown("---")
                st.markdown("### 🔍 Key Insights")
                
                total_events = summary_df['count'].sum()
                total_monsoon = summary_df['monsoon_events'].sum()
                total_non_monsoon = summary_df['non_monsoon_events'].sum()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Extreme Events", total_events)
                col2.metric("During Monsoon", total_monsoon, f"{total_monsoon/total_events*100:.0f}%")
                col3.metric("Outside Monsoon", total_non_monsoon, f"{total_non_monsoon/total_events*100:.0f}%")
                col4.metric("Avg per Year", f"{total_events/len(years_available):.1f}")
                
                # Legend explanation
                st.markdown("---")
                st.markdown("""
                **How to read this chart:**
                - Each point represents a day when water level was in the **top 5%** for that year
                - Clustering of points shows **when flash floods typically occur**
                - Points outside the blue monsoon zone indicate **unusual flash flood periods**
                - Use this to identify **seasonal patterns** and **unexpected flood timing**
                """)
                
            else:
                st.warning("No data available for flash flood analysis")
    
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dc143c 0%, #003893 100%); 
             padding: 3rem; border-radius: 15px; color: white; text-align: center; margin: 2rem 0;">
            <h2 style="color: white; margin: 0;">Nepal Flood Risk Intelligence</h2>
            <h3 style="color: white; margin: 1rem 0;">Powered by ERA5 Climate Reanalysis</h3>
            <p style="font-size: 1.1rem; margin: 1rem 0; opacity: 0.9;">
                Real satellite data with smart caching
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("Select a river station from the sidebar and click 'Load Data' to begin")


if __name__ == "__main__":
    main()