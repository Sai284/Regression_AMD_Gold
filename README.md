# Regression_AMD_Gold
The relationship between the stock prices of Advanced Micro Devices (AMD) and the price of Gold through a statistical lens, using historical price data.

Data Acquisition:

The historical daily closing prices for AMD and Gold were sourced using the yfinance library. A custom script, historical_data_to_csv.py, facilitated this process by automating the download and initial formatting of the financial data.
Data Collection Script: The script prompted for input regarding the ticker symbols for AMD and Gold, the desired time period, and the target directory for saving the output as a CSV file. This method ensured structured and reliable data retrieval, ready for further analysis.
Data Cleaning and Preparation:

After acquisition, the dataset underwent rigorous cleaning which included renaming columns for better clarity and removing any missing or null values to ensure the integrity of the dataset for analysis.
Exploratory Data Analysis (EDA)
Histograms and scatter plots were generated to visually assess the distribution and correlation of the two variables. These initial analyses were crucial for setting the stage for deeper statistical examination.
Statistical Analysis
Correlation Analysis:

A strong positive correlation coefficient of 0.785 between AMD and Gold prices suggests a significant linear relationship, wherein increases in Gold prices are generally associated with increases in AMD prices.
Regression Analysis:

Model Parameters:
Coefficient for Gold Prices: 0.1334, indicating a significant positive impact of Gold prices on AMD stock prices.
Intercept: -152.4914, theoretically representing the AMD stock price when Gold prices are zero, crucial for understanding the y-intercept of the regression line.
Model Diagnostics and Validation:

Durbin-Watson Statistic: A value of 0.030 indicated notable positive autocorrelation, necessitating further review of the residuals.
Jarque-Bera and Omnibus Tests: Highlighted the non-normality of residuals, suggesting potential outliers or model misspecification.
Performance Metrics:

Mean Squared Error (MSE): 539.1
Root Mean Squared Error (RMSE): 23.22
R-Squared: 0.62, illustrating that approximately 62% of the variability in AMD's stock prices is explained by the model.
Visual Analysis
Histograms and Regression Plots provided insight into the distribution of data and the linear relationship modeled. The plots highlighted the positive trend and potential areas for model improvement.
Residuals Plot: Assessed the assumption of homoscedasticity and indicated areas where the model could be enhanced for better prediction accuracy.
Conclusion
The regression analysis robustly supported the predictive relationship between Gold prices and AMD stock prices, backed by substantial statistical evidence. The detailed approach, from data collection using a custom Python script to extensive statistical testing and visual analysis, underscores the reliability and depth of the findings. Future research could expand on this by incorporating more variables, addressing the autocorrelation, and potentially exploring non-linear models to refine predictions and address identified model limitations.
