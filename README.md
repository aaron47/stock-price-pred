# Stock Price Prediction with LSTM

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge\&logo=PyTorch\&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)
![Yahoo Finance](https://img.shields.io/badge/Yahoo%20Finance-6001D2?style=for-the-badge\&logo=yahoo\&logoColor=white)

A stock price prediction project that uses a **Long Short-Term Memory neural network** built with **PyTorch** to forecast stock closing prices from historical market data.

The project includes both:

* a Jupyter notebook for experimentation and model development
* a Streamlit web interface for running predictions interactively

The default ticker is **MSFT**, but the app supports any valid Yahoo Finance ticker symbol such as `AAPL`, `TSLA`, `GOOGL`, or `NVDA`.

---

## Project Overview

This project predicts future stock closing prices using historical daily close prices downloaded from Yahoo Finance.

The model is based on an **LSTM**, a type of recurrent neural network designed for sequential data. Stock prices are naturally sequential because each price depends on previous market behaviour. The model looks at a rolling window of previous closing prices and learns patterns that may help estimate the next closing price.

The pipeline follows these main steps:

1. Download historical stock data using `yfinance`
2. Extract the daily closing price
3. Normalize the closing prices using `StandardScaler`
4. Convert the time series into overlapping sequences
5. Train an LSTM model using PyTorch
6. Predict closing prices on the test set
7. Convert predictions back to real dollar values
8. Evaluate the model using RMSE
9. Display results through a Streamlit UI

---

## How It Works

### 1. Data Collection

Historical stock data is downloaded using the `yfinance` library.

The Streamlit app allows the user to choose:

* ticker symbol
* start date
* number of training epochs

Example tickers:

```python
MSFT
AAPL
TSLA
GOOGL
NVDA
```

The app downloads daily stock data and uses the **Close** column for training and prediction.

---

### 2. Preprocessing

The model does not use raw prices directly. Stock prices can vary widely depending on the company, so the closing prices are normalized first.

This project uses:

```python
StandardScaler()
```

`StandardScaler` transforms the data so that it has:

* mean close to 0
* standard deviation close to 1

This helps the neural network train more efficiently because the input values are placed on a smaller and more stable scale.

The scaled close prices are then reshaped into rolling sequences.

```python
SEQ_LENGTH = 30
```

Each sequence contains 30 scaled closing prices.

The model uses:

* the first 29 values as input
* the final value as the prediction target

So each training example is structured like this:

```text
Input:  days 1 → 29
Target: day 30
```

This creates a supervised learning dataset from a single time series.

---

## LSTM Concept

An **LSTM**, or Long Short-Term Memory network, is a type of recurrent neural network designed to work with sequences.

A normal feedforward neural network treats each input independently. That is not ideal for stock prices because market data is ordered over time. The price today is often related to prices from previous days.

A basic RNN can process sequences, but it struggles with longer time windows because of the **vanishing gradient problem**. During training, information from earlier timesteps can become weaker as it passes through the network.

LSTMs help solve this by using a memory mechanism called the **cell state**.

Each LSTM cell has three main gates:

### Forget Gate

The forget gate decides which information from the previous cell state should be removed.

For stock prices, this helps the model ignore older signals that are no longer useful.

### Input Gate

The input gate decides what new information should be added to the cell state.

This allows the model to store useful information from the current timestep.

### Output Gate

The output gate decides what information should be passed forward as the hidden state.

This hidden state is used by the next timestep and eventually by the final prediction layer.

Together, these gates allow the LSTM to learn patterns across time, such as:

* upward or downward trends
* short-term momentum
* repeated price movements
* changes in volatility
* relationships between recent and older prices

---

## Model Architecture

The model is implemented using PyTorch.

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out
```

### Architecture Summary

```text
LSTMModel(
  (lstm): LSTM(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
  (fc):   Linear(32 → 1)
)
```

### Input Layer

The model receives one feature per timestep:

```text
Close price
```

Because only the closing price is used, the input size is:

```python
input_size = 1
```

The input tensor has the shape:

```text
(batch_size, sequence_length, input_size)
```

Because `batch_first=True`, PyTorch expects the batch dimension first.

For this project, the model receives inputs shaped like:

```text
(batch_size, 29, 1)
```

Each sample contains 29 previous normalized closing prices.

---

## Hidden Layers

The model uses:

```python
hidden_size = 32
num_layers = 2
```

### Hidden Size

The hidden size controls how many features the LSTM stores in its hidden state.

In this model:

```python
hidden_size = 32
```

This means each LSTM layer produces a hidden representation with 32 values.

A larger hidden size can allow the model to learn more complex patterns, but it also increases training cost and may overfit on small datasets.

### Number of Layers

The model uses two stacked LSTM layers:

```python
num_layers = 2
```

The first LSTM layer processes the input sequence and produces hidden representations. The second LSTM layer processes those representations again, allowing the model to learn deeper sequential patterns.

Stacking LSTM layers can help the model capture more abstract temporal relationships, such as longer-term price movement patterns.

---

## Hidden State and Cell State

Inside the `forward()` method, the model initializes two tensors:

```python
h0 = torch.zeros(...)
c0 = torch.zeros(...)
```

### Hidden State

The hidden state stores short-term information passed from one timestep to the next.

It represents what the LSTM currently knows about the sequence.

### Cell State

The cell state stores longer-term memory.

This is the part of the LSTM that helps it remember useful information across multiple timesteps.

Both are initialized with zeros at the start of each forward pass.

They are also moved to the same device as the input:

```python
.to(x.device)
```

This is important because tensors must be on the same device. If the model is running on the GPU, the hidden state and cell state must also be on the GPU.

---

## Final Prediction Layer

The LSTM returns an output for every timestep.

This project only uses the output from the final timestep:

```python
out[:, -1, :]
```

That final timestep contains the model's learned summary of the full input sequence.

The final hidden representation is passed into a fully connected layer:

```python
self.fc = nn.Linear(hidden_size, output_size)
```

This maps the 32-value hidden state to one output value:

```text
Linear(32 → 1)
```

The output is the predicted scaled closing price.

After prediction, the value is converted back to the original price scale using:

```python
scaler.inverse_transform()
```

---

## Why PyTorch Is Used

PyTorch is used to define, train, and evaluate the neural network.

This project uses PyTorch for:

### Model Definition

The model inherits from:

```python
nn.Module
```

This is the base class for neural networks in PyTorch.

### LSTM Layer

The recurrent part of the model is created with:

```python
nn.LSTM()
```

This handles the sequential processing of the stock price windows.

### Linear Layer

The final prediction layer is created with:

```python
nn.Linear()
```

This converts the LSTM output into a single predicted price.

### Loss Function

The model is trained using Mean Squared Error:

```python
nn.MSELoss()
```

MSE measures the average squared difference between predicted and actual values. It is commonly used for regression problems.

### Optimizer

The model is optimized using Adam:

```python
optim.Adam(model.parameters(), lr=0.01)
```

Adam adjusts the model weights during training to reduce the loss.

### GPU Support

The project automatically checks whether CUDA is available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

If an NVIDIA GPU with CUDA support is available, the model trains on the GPU. Otherwise, it falls back to the CPU.

---

## Training Process

The training process follows the standard PyTorch workflow:

1. Set the model to training mode
2. Clear previous gradients
3. Run a forward pass
4. Compute the loss
5. Run backpropagation
6. Update model weights

```python
model.train()

optimizer.zero_grad()

y_pred_train = model(X_train)
loss = criterion(y_pred_train, y_train)

loss.backward()
optimizer.step()
```

The model trains for a user-selected number of epochs.

An epoch means one full pass through the training data.

The Streamlit UI includes a slider for selecting the number of epochs:

```text
50 → 300
```

---

## Evaluation

After training, the model is evaluated on the test set.

The project uses an 80/20 split:

```text
80% training data
20% testing data
```

Because this is time-series data, the split is chronological. The first 80% of the generated sequences are used for training, and the final 20% are used for testing.

The model predicts scaled prices first. These are then inverse-transformed back to dollar values.

The main evaluation metric is:

```text
RMSE — Root Mean Squared Error
```

RMSE shows the average prediction error in the original price unit.

For example, an RMSE of `$4.50` means the model's predictions are off by about $4.50 on average.

---

## Streamlit UI

The project includes a Streamlit interface in:

```text
app.py
```

The UI makes it possible to run the model without editing notebook cells.

### Sidebar Controls

The sidebar contains:

* ticker symbol input
* start date selector
* training epoch slider
* run prediction button
* device indicator showing CPU or GPU

### Main App Output

After running a prediction, the app displays:

* selected ticker
* test RMSE
* number of training samples
* number of test samples

### Tabs

The interface includes three main tabs.

#### Predicted vs Actual

This tab shows a line chart comparing:

* actual closing prices
* predicted closing prices

It also displays a table with the latest prediction results.

#### Quarterly View

This tab shows the stock's closing price grouped by quarter.

This helps visualize broader price trends over time.

#### Scaled Training Data

This tab shows the normalized close prices produced by `StandardScaler`.

It helps explain what data the LSTM actually receives during training.

---

## Project Structure

```text
stockprice-pred/
│
├── app.py          # Streamlit UI
├── main.ipynb     # Notebook version of the model
├── README.md      # Project documentation
└── .venv/         # Optional virtual environment
```

---

## Requirements

Install the required packages with:

```bash
pip install streamlit torch pandas numpy matplotlib yfinance scikit-learn
```

For Jupyter notebook usage, also install:

```bash
pip install jupyterlab ipykernel
```

A CUDA-capable NVIDIA GPU is optional. The project can run on CPU, but GPU training is faster.

---

## Running the Streamlit App

From the project folder, run:

```bash
streamlit run app.py
```

If Streamlit is not found, use:

```bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal.

Streamlit usually starts at:

```text
http://localhost:8501
```

---

## Running the Notebook

Open:

```text
main.ipynb
```

Run the cells from top to bottom.

To change the stock ticker, update the ticker variable in the notebook:

```python
ticker = "AAPL"
```

Then re-run the notebook.

---

## Example Workflow

1. Start the app:

```bash
streamlit run app.py
```

2. Enter a ticker symbol:

```text
MSFT
```

3. Choose a start date.

4. Select the number of training epochs.

5. Press:

```text
Run Prediction
```

6. Review:

* RMSE
* predicted vs actual chart
* quarterly price trend
* scaled training data

---

## Limitations

This project is designed for learning and experimentation. It should not be used as financial advice.

Important limitations:

* The model only uses historical close prices.
* It does not use volume, news, earnings, fundamentals, or macroeconomic indicators.
* Stock prices are noisy and difficult to predict.
* A low RMSE does not guarantee future trading performance.
* The model is retrained every time the app is run.
* The train/test split is simple and does not include walk-forward validation.
* The current model does not include dropout or regularization beyond the LSTM architecture itself.

---

## Future Improvements

Possible improvements include:

* use OHLCV features instead of only close price
* add dropout between LSTM layers
* increase hidden size from 32 to 64
* add more technical indicators
* implement walk-forward validation
* cache trained models
* add next-day prediction beyond the test set
* compare LSTM results with GRU, RNN, ARIMA, or Transformer models
* add model saving and loading
* improve error handling for invalid tickers
* add confidence intervals or uncertainty estimates

---

## Summary

This project demonstrates how PyTorch can be used to build an LSTM-based stock price predictor.

The model learns from normalized historical close prices, processes them as ordered sequences, and predicts future closing prices using a two-layer LSTM followed by a linear output layer.

The Streamlit UI makes the model easier to use by allowing users to select a ticker, train the model, and visualize the results directly in the browser.
