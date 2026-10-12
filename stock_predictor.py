
"""
Stock Price Predictor
Predict next-day closing prices using Linear Regression.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

TICKER = "AAPL"
PERIOD = "5y"
OUTPUT_DIR = Path("outputs")


def download_data():
    print(f"Downloading historical data for {TICKER}...")

    data = yf.download(
        TICKER,
        period=PERIOD,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            "No data downloaded. Check internet and ticker symbol."
        )

    # Handle MultiIndex columns in newer yfinance versions.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[["Close", "Volume"]].dropna()

    if len(data) < 50:
        raise ValueError("Not enough historical data.")

    return data


def prepare_data(data):
    data = data.copy()

    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA10"] = data["Close"].rolling(10).mean()

    # Target is the next trading day's closing price.
    data["Target"] = data["Close"].shift(-1)

    return data.dropna()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    try:
        data = download_data()
    except Exception as error:
        print("Error downloading data:", error)
        return

    dataset = prepare_data(data)

    features = ["Close", "Volume", "MA5", "MA10"]

    # Keep the data in chronological order.
    split = int(len(dataset) * 0.80)

    train = dataset.iloc[:split]
    test = dataset.iloc[split:]

    X_train = train[features]
    y_train = train["Target"]

    X_test = test[features]
    y_test = test["Target"]

    # Train the Linear Regression model.
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Evaluate model performance.
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results = pd.DataFrame(
        {
            "Actual_Next_Close": y_test,
            "Predicted_Next_Close": predictions,
        },
        index=test.index,
    )

    results.index.name = "Feature_Date"
    results.to_csv(OUTPUT_DIR / "predictions.csv")

    # Estimate the next trading day's closing price.
    latest = data.copy()
    latest["MA5"] = latest["Close"].rolling(5).mean()
    latest["MA10"] = latest["Close"].rolling(10).mean()
    latest = latest.dropna().iloc[-1]

    latest_features = pd.DataFrame(
        [[
            latest["Close"],
            latest["Volume"],
            latest["MA5"],
            latest["MA10"],
        ]],
        columns=features,
    )

    next_prediction = model.predict(latest_features)[0]

    print("\n===== STOCK PRICE PREDICTOR =====")
    print("Stock:", TICKER)
    print("Training records:", len(train))
    print("Testing records:", len(test))

    print("\nModel Evaluation")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R-squared: {r2:.4f}")

    print(
        "\nEstimated next trading-day closing price:",
        f"{next_prediction:.2f}",
    )

    print("\nNote: This is an educational estimate,")
    print("not financial advice or a guaranteed price.")

    # Plot actual versus predicted prices.
    plt.figure(figsize=(12, 6))

    plt.plot(
        results.index,
        results["Actual_Next_Close"],
        label="Actual Price",
    )

    plt.plot(
        results.index,
        results["Predicted_Next_Close"],
        label="Predicted Price",
    )

    plt.title(f"{TICKER} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Closing Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    chart_path = OUTPUT_DIR / "actual_vs_predicted.png"
    plt.savefig(chart_path, dpi=150)
    plt.show()

    print("\nSaved predictions:", OUTPUT_DIR / "predictions.csv")
    print("Saved graph:", chart_path)


if __name__ == "__main__":
    main()