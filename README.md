# 🚗 Car Price Prediction App [🚗](https://car-by-specifiaction-9.streamlit.app/)

A full-featured machine learning web app built with Streamlit.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

car_price_prediction_.csv
requirements.txt
car-by-specifiaction
/
README.md
in
main

Edit

Preview
Indent mode

Spaces
Indent size

2
Line wrap mode

Soft wrap
Editing README.md file contents
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10

### 2. Place your data file
Make sure `car_price_prediction_.csv` is in the same folder as `car_price_app.py`.

### 3. Run the app
```bash
streamlit run car_price_app.py
```

The app opens automatically at **http://localhost:8501**

---

## 🔢 Features Used for Prediction

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | **Year** | Numeric | Manufacturing year (2000–2023) |
| 2 | **Engine Size** | Numeric | Engine displacement in litres |
| 3 | **Mileage** | Numeric | Distance driven in km |
| 4 | **Car Age** | Derived | `2024 - Year` |
| 5 | **Brand** | Categorical | Tesla, BMW, Audi, Ford, Honda, Mercedes, Toyota |
| 6 | **Fuel Type** | Categorical | Petrol, Electric, Diesel, Hybrid |
| 7 | **Transmission** | Categorical | Manual / Automatic |

---

## 🤖 ML Models

| Model | Strengths |
|-------|-----------|
| **Gradient Boosting** (default) | Best accuracy, handles non-linearity |
| **Random Forest** | Robust, low variance |
| **Ridge Regression** | Fast, interpretable baseline |

---

## 📊 App Tabs

| Tab | Contents |
|-----|----------|
| 📊 Overview | Dataset stats, box plots, donut chart, data table |
| 🔍 EDA | Scatter plots, trend lines, heatmaps, histograms |
| 🤖 Model Performance | R², MAE, RMSE, MAPE, actual vs predicted, residuals |
| 📈 Feature Importance | Importance bar chart, insights, partial dependence plot |
| 🔗 Correlations | Correlation matrix, violin plots, bar comparisons |

---

## 🌐 Deploy to Streamlit Cloud (free)

1. Push this folder to a **GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app"
3. Select your repo, set main file to `car_price_app.py`
4. Click **Deploy** — live URL in ~2 minutes!
