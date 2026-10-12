# Stock Price Predictor Using Linear Regression

## Project Overview

The Stock Price Predictor is a machine learning project that estimates the next trading day's closing price using historical stock market data.

## Objectives

* Collect historical stock price data.
* Apply Linear Regression for prediction.
* Evaluate the model using MAE, RMSE, and R-squared.
* Visualize actual and predicted stock prices.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* yfinance

## Methodology

The project downloads historical stock data, calculates moving averages, and trains a Linear Regression model. The dataset is divided chronologically into training and testing sets.

## How to Run

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the program:

```bash
python stock_predictor.py
```

## Output

The program generates a graph comparing actual and predicted stock prices and saves test predictions to a CSV file.

## Disclaimer

This project is for educational purposes only. Predictions are estimates and are not financial advice.
