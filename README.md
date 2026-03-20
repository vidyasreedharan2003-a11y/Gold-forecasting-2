# Gold BiLSTM Forecast — Streamlit App

## Project Structure

```
gold_streamlit/
├── app.py                      ← Main Streamlit app
├── requirements.txt
├── gold_bilstm_forecast.png    ← Pre-generated forecast chart
├── model/
│   ├── gold_bilstm_model.keras
│   ├── gold_scaler.pkl
│   ├── gold_target_scaler.pkl
│   └── gold_features.csv
└── data/
    └── gold.csv
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app (from inside the gold_streamlit/ folder)
streamlit run app.py
```

The app will open at http://localhost:8501

## Features
- Loads the trained BiLSTM model with 25 features
- Runs a 10-day multi-step price forecast
- Displays forecast table with colour-coded directions
- Interactive price chart with uncertainty band
- Daily returns bar chart
- Full price history
- CSV download of forecast results
