# Stock Backtesting Using Various Models

This repository contains Jupyter notebooks for backtesting VN30 and VNINDEX stock data using different machine learning and statistical models. Each notebook focuses on a specific model or combination of models: LSTM, GRU, Prophet, and XGBoost.

## Models and Methods

### LSTM (Long Short-Term Memory)
LSTM is a type of recurrent neural network (RNN) that is well-suited for time series forecasting due to its ability to learn long-term dependencies. LSTMs address the vanishing gradient problem commonly encountered with traditional RNNs by using a memory cell that maintains its state over time, controlled by three gates (input, forget, and output gates). This makes LSTMs particularly effective for time series data where capturing temporal dependencies is crucial.

The `LSTM + TA.ipynb` and `LSTM + Prophet + GRU.ipynb` notebooks use LSTM to predict future stock prices based on historical data and technical indicators.

### GRU (Gated Recurrent Unit)
GRU is another type of recurrent neural network similar to LSTM but with a simpler architecture. GRUs combine the forget and input gates into a single update gate, and they also merge the cell state and hidden state. This simplification makes GRUs computationally efficient while still effectively capturing temporal dependencies in sequential data.

The `GRU + TA.ipynb` and `LSTM + Prophet + GRU.ipynb` notebooks use GRU to forecast stock prices.

### Prophet
Prophet is a forecasting tool developed by Facebook that is particularly good at handling time series with strong seasonal effects and missing data. Prophet models time series data as an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, along with holiday effects. It is robust to missing data and shifts in the trend, and it typically requires minimal tuning.

The `Prophet + TA.ipynb` and `LSTM + Prophet + GRU.ipynb` notebooks use Prophet for stock price predictions.

### XGBoost (Extreme Gradient Boosting)
XGBoost is a powerful machine learning algorithm based on gradient boosting, which builds an ensemble of decision trees in a sequential manner. Each tree attempts to correct the errors of its predecessor. XGBoost is known for its efficiency and accuracy, often outperforming other algorithms in many machine learning competitions. It includes advanced features like regularization, which helps prevent overfitting.

The `XGBoost + TA.ipynb` notebook applies XGBoost for time series forecasting.

## Technical Indicators (TA)
Technical indicators are mathematical calculations based on historical price, volume, or open interest information that traders use to predict future market behavior. These indicators provide additional features for the models, helping them capture market trends and patterns. The notebooks use the following technical indicators:
- **Simple Moving Average (SMA)**: A basic moving average that calculates the average price over a specified number of periods.
- **Exponential Moving Average (EMA)**: A moving average that places more weight on recent prices, making it more responsive to new information.
- **Relative Strength Index (RSI)**: A momentum oscillator that measures the speed and change of price movements.
- **Moving Average Convergence Divergence (MACD)**: A trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.

## Voting Mechanism
The `LSTM + Prophet + GRU.ipynb` notebook implements a voting mechanism to combine predictions from multiple models (LSTM, GRU, and Prophet). The voting mechanism is designed to leverage the strengths of each model and improve overall prediction accuracy. The process works as follows:

1. **Model Predictions**: Each model (LSTM, GRU, and Prophet) makes a prediction for the future stock price based on historical data and technical indicators.
2. **Calculate Differences**: For each model, the difference between the predicted price and the current price is calculated.
3. **Cast Votes**: A vote is cast based on the difference:
   - **Long Vote**: If the difference is greater than 2, the model casts a "long" vote, indicating a buy signal.
   - **Short Vote**: If the difference is less than -2, the model casts a "short" vote, indicating a sell signal.
   - **Neutral Vote**: If the difference is between -2 and 2, the model casts a "neutral" vote, indicating a hold signal.
4. **Aggregate Votes**: The votes from all models are aggregated.
5. **Final Decision**: The final trading position is determined based on the majority of votes:
   - **Buy (1)**: If there are more "long" votes, the position is set to 1.
   - **Sell (-1)**: If there are more "short" votes, the position is set to -1.
   - **Hold (0)**: If the votes are tied, the position is set to 0.

## Usage

1. **Ensure you have the required Python packages installed:**
    ```bash
    pip install vnstock3 pandas_ta tensorflow scikit-learn statsmodels prophet xgboost matplotlib
    ```

2. **Run the Jupyter notebooks:**
    - [LSTM + TA.ipynb](LSTM%20+%20TA.ipynb)
    - [GRU + TA.ipynb](GRU%20+%20TA.ipynb)
    - [Prophet + TA.ipynb](Prophet%20+%20TA.ipynb)
    - [XGBoost + TA.ipynb](XGBoost%20+%20TA.ipynb)
    - [LSTM + Prophet + GRU.ipynb](LSTM%20+%20Prophet%20+%20GRU.ipynb)
    - [Backtesting Utilities (F.py)](F.py)

3. **Run the notebooks:**
    ```bash
    jupyter notebook LSTM\ +\ TA.ipynb
    jupyter notebook GRU\ +\ TA.ipynb
    jupyter notebook Prophet\ +\ TA.ipynb
    jupyter notebook XGBoost\ +\ TA.ipynb
    jupyter notebook LSTM\ +\ Prophet\ +\ GRU.ipynb
    ```

## Explanation of Notebooks

### `LSTM + TA.ipynb`
This notebook uses LSTM to predict stock prices. It processes the data to generate technical indicators, scales the data, and then trains an LSTM model. The LSTM model makes predictions based on historical data, and these predictions are used for backtesting.

### `GRU + TA.ipynb`
Similar to the LSTM notebook, this notebook uses GRU to forecast stock prices. It follows the same preprocessing steps, trains a GRU model, and performs backtesting based on the predictions.

### `Prophet + TA.ipynb`
This notebook leverages the Prophet model for time series forecasting. It trains the Prophet model on historical stock data and uses it to predict future prices for backtesting.

### `XGBoost + TA.ipynb`
The XGBoost notebook processes the data to create features, scales the data, and trains an XGBoost regressor. It uses the trained model to make predictions and perform backtesting.

### `LSTM + Prophet + GRU.ipynb`
This notebook combines LSTM, GRU, and Prophet models for forecasting. It uses a voting mechanism to make trading decisions based on predictions from all three models. This approach aims to leverage the strengths of each model to improve prediction accuracy.

## Backtesting
Backtesting is performed by comparing the model predictions to actual stock prices to generate buy/sell signals. The `F.py` module contains utility functions for calculating profit and loss (PNL), sharp ratio, margin, and other metrics. These functions are used to evaluate the performance of the models.

### Key Metrics for Evaluation
- **Profit and Loss (PNL)**: Measures the overall profit or loss of the trading strategy.
- **Sharpe Ratio**: A measure of risk-adjusted return, calculated as the ratio of the expected return to the standard deviation of the return.
- **Maximum Drawdown (MDD)**: The maximum observed loss from a peak to a trough of a portfolio, before a new peak is attained.
- **Hit Rate**: The percentage of successful trades out of the total trades executed.
- **Margin**: The amount of leverage used in the trading strategy.

## Visualization
Each notebook includes functions to plot the PNL and visualize the performance of the trading strategy. The plots help in understanding how the strategy performs over time and under different market conditions.

## Author
Your Name

## License
This project is licensed under the MIT License.
