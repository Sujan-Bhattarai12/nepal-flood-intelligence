# 🇳🇵 Nepal Flood Intelligence Platform

Real-time flood forecasting and risk analysis for Nepal's major river basins using ERA5 climate data and deep learning.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)

##  Features

- **Real-time Monitoring**: Track water levels and discharge for 5 major rivers
- **ERA5 Climate Data**: Powered by Copernicus Climate Reanalysis
- **7-Day Forecasts**: LSTM-based flood prediction
- **Historical Analysis**: 5-year trend analysis and seasonal patterns
- **Flash Flood Detection**: Identify extreme events and timing patterns
- **Smart Caching**: Instant load times with 24-hour data refresh

##  Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/nepal-flood-intelligence.git
cd nepal-flood-intelligence

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

##  River Stations

| River | Station | Drainage Area | District |
|-------|---------|---------------|----------|
| Bagmati | Khokana | 678 km² | Lalitpur |
| Koshi | Chatara | 54,100 km² | Sunsari |
| Narayani | Narayanghat | 31,100 km² | Chitwan |
| Karnali | Chisapani | 42,890 km² | Bardiya |
| Kankai | Mainachuli | 1,148 km² | Jhapa |

## 🔧 Configuration

### ERA5 API Setup (Optional)

For real climate data, configure Copernicus CDS API:

1. Register at [CDS](https://cds.climate.copernicus.eu)
2. Create `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
```

Without API setup, the system uses realistic synthetic data.

## 📁 Project Structure

```
nepal-flood-intelligence/
├── app.py                      # Main Streamlit dashboard
├── notebooks/
│   ├── 01_data_collection.ipynb    # ERA5 data fetching
│   ├── 02_flood_analysis.ipynb     # Event detection & stats
│   ├── 03_model_training.ipynb     # LSTM forecasting
│   └── 04_visualization.ipynb      # Chart development
├── src/
│   ├── data_loader.py          # ERA5 & cache management
│   ├── analytics.py            # Flood detection algorithms
│   ├── models.py               # LSTM architecture
│   └── config.py               # Station configurations
├── requirements.txt
└── README.md
```

##  Output


