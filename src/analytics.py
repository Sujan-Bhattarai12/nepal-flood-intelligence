"""
Flood Analytics and Event Detection
"""

import pandas as pd
import numpy as np


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
        """Detect flood events based on threshold exceedance"""
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
        """Classify flood severity based on water level"""
        if level >= self.station_config['major_flood']:
            return 'Major'
        elif level >= self.station_config['moderate_flood']:
            return 'Moderate'
        else:
            return 'Minor'
    
    def seasonal_risk_profile(self):
        """Calculate flood risk by season"""
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
        """Compare flood statistics across years"""
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
        """Compare current conditions to historical data"""
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