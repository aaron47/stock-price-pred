import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

# ── Constants ────────────────────────────────────────────────────────────────

SEQ_LENGTH = 30  # total window size: first 29 days = X, final day = y
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model ────────────────────────────────────────────────────────────────────


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(
            x.device
        )

        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(
            x.device
        )

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out


# ── Helpers ──────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def download_data(ticker: str, start: str) -> pd.DataFrame:
    return yf.download(ticker, start=start, progress=False)


def get_close_prices(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Handles both normal yfinance columns and MultiIndex columns.
    Returns a DataFrame with one column: Close.
    """

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            close = df[("Close", ticker)]
        elif "Close" in df.columns.get_level_values(0):
            close = df["Close"].iloc[:, 0]
        else:
            raise ValueError("Close column not found in downloaded data.")
    else:
        close = df["Close"]

    return close.to_frame(name="Close")


def build_notebook_style_windows(close_scaled: np.ndarray, seq_length: int):
    """
    Same logic as notebook:

    data[i] = close_scaled[i : i + seq_length]
    X      = data[:, :-1, :]
    y      = data[:, -1, :]
    """

    data = []

    for i in range(len(close_scaled) - seq_length):
        data.append(close_scaled[i : i + seq_length])

    data = np.array(data)

    return data


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

st.title("📈 Stock Price Predictor")
st.caption("LSTM neural network trained on historical daily close prices.")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    ticker = st.text_input("Ticker Symbol", value="MSFT").strip().upper()

    start_date = st.date_input("Start Date", value=pd.Timestamp("2025-01-01"))

    epochs = st.slider(
        "Training Epochs", min_value=50, max_value=300, value=100, step=50
    )

    run = st.button("▶ Run Prediction", type="primary", use_container_width=True)

    st.divider()

    st.caption(f"Device: **{'GPU (CUDA)' if device.type == 'cuda' else 'CPU'}**")

# ── Main ─────────────────────────────────────────────────────────────────────

if not run:
    st.info("Enter a ticker in the sidebar and press **Run Prediction** to start.")
    st.stop()

# 1. Download data

with st.spinner(f"Downloading {ticker} data..."):
    df = download_data(ticker, str(start_date))

if df.empty:
    st.error(f"No data found for **{ticker}**. Check the symbol and try again.")
    st.stop()

# 2. Preprocess using your notebook scaling/window logic

try:
    close_df = get_close_prices(df, ticker)
except Exception as e:
    st.error(f"Could not extract close prices: {e}")
    st.stop()

close_prices = close_df.values.reshape(-1, 1)

scaler = StandardScaler()
close_scaled = scaler.fit_transform(close_prices)

data = build_notebook_style_windows(close_scaled, SEQ_LENGTH)

if len(data) < 20:
    st.error("Not enough data to train. Try an earlier start date.")
    st.stop()

train_size = int(len(data) * 0.8)

X_train = torch.from_numpy(data[:train_size, :-1, :]).float().to(device)
y_train = torch.from_numpy(data[:train_size, -1, :]).float().to(device)

X_test = torch.from_numpy(data[train_size:, :-1, :]).float().to(device)
y_test = torch.from_numpy(data[train_size:, -1, :]).float().to(device)

# 3. Train model

model = LSTMModel().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

progress = st.progress(0, text="Training model...")

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()

    y_pred_train = model(X_train)
    loss = criterion(y_pred_train, y_train)

    loss.backward()
    optimizer.step()

    progress.progress((epoch + 1) / epochs, text=f"Training... {epoch + 1}/{epochs}")

progress.empty()

# 4. Evaluate

model.eval()

with torch.no_grad():
    pred_scaled = model(X_test).cpu().numpy()

pred = scaler.inverse_transform(pred_scaled)
actual = scaler.inverse_transform(y_test.cpu().numpy())

rmse = root_mean_squared_error(actual, pred)

# Create matching test dates.
# Since each target is the final item in a 30-day window,
# target dates start at index SEQ_LENGTH - 1.
target_dates = close_df.index[SEQ_LENGTH - 1 : SEQ_LENGTH - 1 + len(data)]
test_dates = target_dates[train_size:]

# 5. Metrics row

c1, c2, c3, c4 = st.columns(4)

c1.metric("Ticker", ticker)
c2.metric("Test RMSE", f"${rmse:.2f}")
c3.metric("Train Samples", len(X_train))
c4.metric("Test Samples", len(X_test))

# 6. Charts

tab1, tab2, tab3 = st.tabs(
    ["📉 Predicted vs Actual", "📆 Quarterly View", "🧪 Scaled Training Data"]
)

with tab1:
    result_df = pd.DataFrame(
        {"Date": test_dates, "Actual": actual.flatten(), "Predicted": pred.flatten()}
    )

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(result_df["Date"], result_df["Actual"], label="Actual", linewidth=1.5)

    ax.plot(
        result_df["Date"],
        result_df["Predicted"],
        label="Predicted",
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_title(f"{ticker} — Predicted vs Actual Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=30)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(result_df.tail(20), use_container_width=True)

with tab2:
    quarterly_df = close_df.reset_index()
    quarterly_df.columns = ["Date", "Close"]

    quarterly_df["Quarter"] = pd.PeriodIndex(quarterly_df["Date"], freq="Q")

    quarters = quarterly_df["Quarter"].unique()
    n = len(quarters)

    fig2, axes = plt.subplots(n, 1, figsize=(12, 3 * n))

    if n == 1:
        axes = [axes]

    for ax, q in zip(axes, quarters):
        subset = quarterly_df[quarterly_df["Quarter"] == q]

        ax.plot(subset["Date"], subset["Close"], linewidth=1.5)

        ax.fill_between(subset["Date"], subset["Close"], alpha=0.15)

        ax.set_title(str(q), fontsize=11)
        ax.set_ylabel("Close ($)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{ticker} Close Price by Quarter", fontsize=14, fontweight="bold")

    plt.tight_layout()

    st.pyplot(fig2)
    plt.close(fig2)

with tab3:
    scaled_df = pd.DataFrame(
        {
            "Date": close_df.index,
            "Original Close": close_prices.flatten(),
            "Scaled Close": close_scaled.flatten(),
        }
    )

    st.write("This shows the `StandardScaler` transformation used before training.")

    st.dataframe(scaled_df.tail(30), use_container_width=True)

    fig3, ax = plt.subplots(figsize=(12, 4))

    ax.plot(scaled_df["Date"], scaled_df["Scaled Close"], linewidth=1.5)

    ax.set_title(f"{ticker} — StandardScaler Normalized Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Scaled Close")
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=30)
    plt.tight_layout()

    st.pyplot(fig3)
    plt.close(fig3)
