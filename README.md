# Stock Price Prediction with LSTM

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

Predicts the next-day closing price of any stock using a two-layer LSTM neural network trained on historical OHLCV data from Yahoo Finance. The default ticker is **MSFT** — swap it for any valid symbol and re-run.

---

## How it works

1. **Data** — daily price history is pulled from Yahoo Finance via `yfinance`
2. **Preprocessing** — close prices are normalised with `StandardScaler` and reshaped into overlapping 60-day sequences (SEQ_LENGTH = 60)
3. **Model** — a 2-layer LSTM with 64 hidden units and a linear output head predicts the next day's scaled close price
4. **Training** — minimises MSE loss with the Adam optimiser over 100 epochs
5. **Evaluation** — predictions are inverse-transformed back to USD and evaluated with RMSE

---

## Model architecture

```
LSTMModel(
  (lstm): LSTM(1, 64, num_layers=2, batch_first=True, dropout=0.2)
  (fc):   Linear(64 → 1)
)
```

- **Input:** sequence of 60 normalised close prices `(batch, 60, 1)`
- **Output:** single predicted close price (scaled), then inverse-transformed to USD
- **Loss:** MSELoss during training → RMSE reported at evaluation

---

## Requirements

```
torch
pandas
numpy
matplotlib
yfinance
scikit-learn
```

Install with:

```bash
pip install torch pandas numpy matplotlib yfinance scikit-learn
```

A CUDA-capable GPU is recommended but not required — the notebook falls back to CPU automatically.

---

## Usage

Open `main.ipynb` and run all cells. To predict a different stock, change the ticker in the data cell:

```python
ticker = 'AAPL'  # or 'TSLA', 'GOOGL', etc.
```

Re-run the notebook from top to bottom.
