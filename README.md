# AI Quantitative Trading System with LSTM

A comprehensive AI-driven quantitative trading system 
that combines deep learning (LSTM neural networks) with traditional backtesting frameworks
to predict stock market movements and generate trading signals.

## Key Features

- **LSTM Neural Network**: for stock price prediction using sequential market data
- **Backtrader Integration**: for professional-grade backtesting with realistic market simulations
- **Market Regime Detection**: incorporating bull/bear market states into trading decisions
- **Risk Management**: with stop-loss (3%) and take-profit (5%) mechanisms
- **Technical Indicators**: including 60-day returns and volatility calculations
- **Chinese Stock Market Data**: integration with comprehensive financial reporting
- **Concept/Sector Analysis**: for Chinese A-share market using East Money data
- **Financial Statement Processing**: with comprehensive Chinese-English field mappings


## System Architecture

```

├── Neural Network (PyTorch)
│   ├── LSTM Model with 60-day lookback window
│   ├── OHLCV feature engineering
│   └── Regime-aware normalization
│
├── Backtesting Engine (Backtrader)
│   ├── Custom trading strategies
│   ├── Performance analytics (Sharpe ratio, drawdown)
│   └── Sizing and commission handling
│
├── Data Pipeline
│   ├── Historical data fetching (akshare)
│   ├── Market regime calculation
│   └── Financial statement processing
│
└── Sector Analysis
    ├── Concept/sector classification
    └── Leading stock identification
```


## Installation
**Prerequisites**

```
CUDA-compatible GPU (optional, for faster LSTM training)
```

***Setup***

```
# Clone the repository
git clone https://github.com/yourusername/ai-quant-trading.git
cd ai-quant-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


## Dependencies

- **Core ML**: TensorFlow, PyTorch, scikit-learn
- **Data Processing—**: pandas, numpy, yfinance
- **Backtesting**: backtrader
- **Chinese Market Data**: akshare
- **Visualization**: matplotlib, plotly

## Usage

**1. Environment Verification**

```
python env_test.py
```


**2. LSTM Model Training**

```
# Train the LSTM model (separate training script needed)
python train_lstm.py --stock sh600372 --epochs 100
```

**3. Backtesting Execution**

```
python backtest_lstm.py
```

**4. Sector Analysis**

```
from stock_a import get_em_concept, get_em_concept_stocks
# Get all concept sectors
concepts = get_em_concept()

# Get stocks in a specific sector (e.g., BK0918)
stocks = get_em_concept_stocks("BK0918")
```

## LSTM Trading Strategy

**Input Features**

- 60-day sequential OHLCV data
- Market regime (6 states: bear_low, bear_high, neut_low, neut_high, bull_low, bull_high)
- Normalized technical indicators

**Trading Logic**

- Buy Signal: Predicted return > 0.15% AND market regime > 1 (non-bearish)
- Sell Signal: Predicted return < 0% OR stop-loss/take-profit triggered
- Stop-Loss: 3% from entry price
- Take-Profit: 5% from entry price
- Position Sizing: Fixed 400 shares per trade


## Market Regime Classification

Uses 60-day rolling returns and volatility to classify market into 6 regimes:
- Bear market, low volatility
- Bear market, high volatility
- Neutral market, low volatility
- Neutral market, high volatility
- Bull market, low volatility
- Bull market, high volatility


## Project Structure

```

text
ai-quant-trading/
├── backtest_lstm.py          # Main backtesting script
├── env_test.py              # Environment verification
├── stock_a.py               # Data fetching utilities
├── models/
│   └── lstm_model.pth       # Trained LSTM model
├── stock_a_data/
│   └── history/             # Cached historical data
├── libs/
│   └── stock_a.py           # Data fetching module
├── requirements.txt         # Python dependencies
└── README.md               # This file
```


## Performance Metrics

The system tracks comprehensive performance metrics:
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Total Return**: Overall portfolio performance
- **Win Rate**: Percentage of profitable trades
- **Risk-Reward Ratio**: Average profit vs average loss


## Financial Data Coverage

The system includes comprehensive financial statement analysis with 400+ Chinese financial metrics, including:
- Income statement items (revenue, expenses, profits)
- Balance sheet metrics
- Cash flow analysis
- Comprehensive income components
- Specialized Chinese market indicators


## Key Innovations

- **Regime-Aware Trading**: Adjusts strategy based on market conditions
- **Sequence-to-One Prediction**: LSTM predicts next-period returns from 60-day sequences
- **Realistic Backtesting**: Includes commissions, slippage, and market impact
- **Chinese Market Specialization**: Tailored for A-share market characteristics
- **Sector Rotation Signals**: Identifies leading sectors and concept stocks



