# ai-trading-0.1
Neural network for crypto candlestick prediction using PyTorch. Converts OHLCV data into sequences and trains LSTM/GRU models to forecast next price move or return. Includes preprocessing, training loop, and modular design with hooks for reinforcement learning extensions.

Neural network for crypto candlestick prediction using PyTorch. Converts OHLCV data into sequences and trains LSTM/GRU models to forecast the next price move or return. Includes preprocessing, training loop, and modular design with hooks for reinforcement learning extensions.

## Step-by-step quickstart

Follow the steps below to go from an empty environment to a trained model.

### 1. Prepare your environment

1. Install **Python 3.9+** and ensure `python` and `pip` are available on your PATH.
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Upgrade pip and install the runtime dependencies:
   ```bash
   pip install --upgrade pip
   pip install torch pandas numpy pyyaml
   ```
   > 💡 Install the CPU or GPU build of PyTorch that matches your platform from [pytorch.org](https://pytorch.org/get-started/locally/) if you need CUDA support.

### 2. Obtain OHLCV market data

1. Export or download candlestick data that contains at least the following columns:
   - `open`, `high`, `low`, `close`, `volume`
   - Optionally `timestamp` (used for sorting if present)
2. Save the file as CSV or Parquet. For example:
   ```csv
   timestamp,open,high,low,close,volume
   2023-01-01T00:00:00Z,16500,16620,16480,16580,120.5
   2023-01-01T00:05:00Z,16580,16610,16550,16570,98.2
   ```
3. Place the file anywhere on disk (e.g. `data/btc_5m.csv`).

### 3. Create a training configuration

Training is driven by a YAML configuration. Copy the template below into `config.yml` and adjust the paths and hyperparameters to match your dataset.

```yaml
data:
  path: data/btc_5m.csv        # Path to your CSV/Parquet file
  format: csv                  # csv or parquet (optional when extension matches)
  features: [open, high, low, close, volume]
  normalization: zscore        # or "minmax"
  target: return               # "return" to predict percentage change, "close" for raw price
  target_column: close
  window_size: 50              # Number of past candles per sequence
  dropna: true

model:
  type: lstm                   # lstm or gru
  input_size: 5                # Must match the number of features
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
  output_size: 1
  bidirectional: false

training:
  epochs: 20
  batch_size: 64
  learning_rate: 0.001
  optimizer: adam
  weight_decay: 0.0
  seed: 42
  num_workers: 0
  drop_last: false
  shuffle: true

validation:
  split: 0.2                   # Portion of data for validation
  test_split: 0.0              # Set >0 to hold out a test set
  metrics: [mse, mae, directional_accuracy]

checkpoint:
  dir: checkpoints             # Where checkpoints are stored
  save_best: true              # Save the best validation model
  save_every: null             # Optionally save every N epochs
```

### 4. Launch training

Run the training script and point it to your configuration. Optional command-line flags let you override key hyperparameters without editing the YAML file.

```bash
python train.py --config config.yml --epochs 10 --batch-size 128 --device cuda
```

- If `--config` is omitted the script falls back to the built-in defaults (expects `data/sample.csv`).
- Use `--device cpu` to force CPU training.
- Add `--verbose` to see debug-level logs during preprocessing and training.

During training the script will:
1. Load and normalize your OHLCV data.
2. Create sliding-window sequences and split them into train/validation(/test) sets.
3. Build the requested LSTM/GRU model and begin training.
4. Report losses/metrics each epoch and optionally save checkpoints.

### 5. Inspect results

- Checkpoints are written to the directory specified in `checkpoint.dir` (default: `checkpoints/`).
- Metrics and progress are printed to the console. Redirect the output to a log file if desired:
  ```bash
  python train.py --config config.yml | tee training.log
  ```
- When `validation.test_split` > 0, final test metrics are logged after training completes.

## Next steps

- Tweak the normalization strategy or window size to match your market and time frame.
- Extend the YAML configuration with scheduler options (e.g. `scheduler: {type: steplr, step_size: 10, gamma: 0.5}`).
- Integrate your own downstream evaluation, order execution, or reinforcement learning components using the modular dataset and model utilities.
