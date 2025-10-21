"""
Configuration settings for Nepal Flood Intelligence Platform
"""

class Config:
    """System configuration for Nepal flood monitoring"""
    
    # Model parameters
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
    
    # River station configurations
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