# Nepal Flood Intelligence Platform

A real-time flood forecasting and risk analysis system for Nepal's five major river basins. The platform monitors water levels and discharge at gauging stations across the Bagmati, Koshi, Narayani, Karnali, and Kankai rivers, generates seven-day discharge forecasts, and identifies flood events from five years of historical records.

## Problem

Nepal's monsoon season causes recurring flood damage across downstream communities in the Terai region. Existing monitoring infrastructure provides limited lead time for evacuations and emergency response. This platform addresses that gap by combining climate reanalysis data with a time-series forecasting model to produce actionable seven-day outlooks for five critical river basins.

## River Monitoring Stations

| River | Station | Drainage Area | District |
|-------|---------|---------------|----------|
| Bagmati | Khokana | 678 km² | Lalitpur |
| Koshi | Chatara | 54,100 km² | Sunsari |
| Narayani | Narayanghat | 32,000 km² | Chitwan |
| Karnali | Chisapani | 43,900 km² | Bardiya |
| Kankai | Mainachuli | 1,148 km² | Jhapa |

## System Architecture

Data flows through the system in five stages:

1. **Data Ingestion** — Precipitation and atmospheric variables are fetched from the ERA5 climate reanalysis archive via the Copernicus CDS API.
2. **Preprocessing** — Raw discharge values are converted to water levels using rating curve equations calibrated to each station.
3. **Flood Detection** — A statistical algorithm scans the processed time series for threshold exceedances and classifies events by severity (minor, moderate, severe).
4. **Forecasting** — An LSTM (Long Short-Term Memory) neural network, trained on five years of ERA5 data, produces seven-day discharge forecasts with confidence intervals for each station.
5. **Visualization** — Results are rendered as interactive Plotly charts in a Streamlit dashboard.

## Project Structure

```
nepal-flood-intelligence/
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Python dependencies
├── README.md
├── archive/
│   └── app_original.py            # Legacy monolithic version
├── assets/
│   └── images/
│       └── river-background.jpg
├── cache/                         # Auto-generated data and model cache
├── docs/
│   └── images/                    # Documentation screenshots
├── notebooks/
│   ├── 01-era5-data-collection.ipynb
│   ├── 02-flood-event-analysis.ipynb
│   ├── 03-lstm-model-training.ipynb
│   └── 04-visualization-development.ipynb
└── src/
    ├── __init__.py
    ├── config.py                 # Station definitions, LSTM hyperparameters
    ├── data_loader.py            # ERA5 data fetching and local caching
    ├── analytics.py              # Flood detection and severity classification
    ├── models.py                 # PyTorch LSTM model definition and inference
    └── dashboard.py              # Streamlit UI components
```

## Development Notebooks

Each stage of the pipeline has a corresponding notebook for exploration and prototyping:

| Notebook | Purpose |
|----------|---------|
| `01-era5-data-collection.ipynb` | Climate data fetching and preprocessing |
| `02-flood-event-analysis.ipynb` | Statistical analysis and event detection |
| `03-lstm-model-training.ipynb` | Model training, validation, and hyperparameter tuning |
| `04-visualization-development.ipynb` | Chart prototyping and layout design |

To run:

```bash
jupyter notebook notebooks/01-era5-data-collection.ipynb
```

## Key Dependencies

| Package | Role |
|---------|------|
| Streamlit | Web dashboard |
| PyTorch | LSTM model training and inference |
| Plotly | Interactive charts |
| Pandas, NumPy | Data processing |
| cdsapi | Copernicus CDS API client |
| xarray, netCDF4 | Climate data file handling |

Full list with pinned versions in `requirements.txt`.

## Data Source

All climate data is sourced from the [ERA5 reanalysis dataset](https://cds.climate.copernicus.eu) produced by the Copernicus Climate Change Service (C3S). River station metadata is based on records from the Nepal Department of Hydrology and Meteorology.

## License

MIT
