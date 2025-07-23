# Relationship Between AMD Stock Prices and Gold Prices

## Overview
This project analyzes the statistical relationship between Advanced Micro Devices (AMD) stock prices and gold prices using historical market data. The analysis includes data collection, cleaning, exploratory analysis, and statistical modeling to understand the correlation and predictive relationship between these two assets.

## Key Findings
- **Strong positive correlation (0.785)** between AMD and gold prices
- **Regression model** shows gold prices have statistically significant impact on AMD stock:
  - Coefficient: 0.1334 (p < 0.05)
  - Intercept: -152.4914
  - R²: 0.62 (62% of AMD price variance explained)
  - RMSE: 23.22

## Methodology

### Data Collection
- Used `yfinance` Python library to fetch historical daily closing prices
- Automated data collection with `historical_data_to_csv.py` script
- Time period: [Insert date range]

### Data Processing
- Cleaned and formatted raw data
- Handled missing values
- Standardized price metrics

### Analysis Techniques
1. **Exploratory Data Analysis**
   - Distribution analysis (histograms)
   - Scatter plots for correlation visualization

2. **Statistical Modeling**
   - Pearson correlation analysis
   - Linear regression modeling
   - Residual analysis

## Results
![Regression Plot](path/to/regression_plot.png)
*Figure 1: Regression plot showing relationship between gold and AMD prices*

![Residuals Plot](path/to/residuals_plot.png)
*Figure 2: Residual analysis plot*

## How to Reproduce
1. pip install statsmodels
2. pip install openpyxl
3. Ensure path is correct

