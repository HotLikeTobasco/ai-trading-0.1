"""Utilities for loading and preprocessing OHLCV candlestick data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_FEATURE_COLUMNS",
    "PreprocessingResult",
    "load_ohlcv",
    "normalize_candlesticks",
    "apply_normalization",
    "generate_sliding_windows",
    "preprocess_ohlcv",
]

DEFAULT_FEATURE_COLUMNS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")


@dataclass
class PreprocessingResult:
    """Container aggregating preprocessing outputs."""

    sequences: np.ndarray
    targets: np.ndarray
    feature_columns: Tuple[str, ...]
    target_type: str
    normalization_stats: Dict[str, Any]
    raw_dataframe: pd.DataFrame
    context: Dict[str, np.ndarray]


def load_ohlcv(
    file_path: str | Path,
    *,
    fmt: Optional[str] = None,
    columns: Optional[Iterable[str]] = None,
    parse_dates: bool = True,
    sort_by_timestamp: bool = True,
) -> pd.DataFrame:
    """Load OHLCV data from CSV or Parquet files.

    Parameters
    ----------
    file_path:
        Path to the source data file.
    fmt:
        Optional explicit format ("csv" or "parquet"). If omitted the file
        extension is used.
    columns:
        Optional iterable of columns to retain from the loaded file. Missing
        columns raise a ``KeyError``.
    parse_dates:
        Convert the ``timestamp`` column (if present) to ``datetime`` objects.
    sort_by_timestamp:
        Whether to sort the dataframe by ``timestamp`` when that column exists.
    """

    path = Path(file_path)
    if fmt is None:
        fmt = path.suffix.lstrip(".").lower()
    fmt = fmt.lower()

    if fmt == "csv":
        df = pd.read_csv(path)
    elif fmt in {"parquet", "pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format '{fmt}'. Expected 'csv' or 'parquet'.")

    if columns is not None:
        missing = set(columns) - set(df.columns)
        if missing:
            raise KeyError(f"Missing expected columns: {sorted(missing)}")
        df = df.loc[:, list(columns)]

    if parse_dates and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if sort_by_timestamp and "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def normalize_candlesticks(
    df: pd.DataFrame,
    *,
    feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
    method: str = "zscore",
    feature_range: Tuple[float, float] = (0.0, 1.0),
    epsilon: float = 1e-8,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Normalize OHLCV features using z-score or min-max scaling."""

    method = method.lower()
    features = list(feature_columns)
    missing = set(features) - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame is missing columns required for normalization: {sorted(missing)}")

    values = df[features].astype(float)
    stats: Dict[str, Any] = {"method": method, "epsilon": epsilon, "feature_columns": tuple(features)}

    if method == "zscore":
        mean = values.mean()
        std = values.std().replace(0.0, epsilon)
        normalized = (values - mean) / std
        stats.update({"mean": mean, "std": std})
    elif method in {"minmax", "min-max"}:
        min_vals = values.min()
        max_vals = values.max()
        span = (max_vals - min_vals).replace(0.0, epsilon)
        lower, upper = feature_range
        normalized = (values - min_vals) / span
        normalized = normalized * (upper - lower) + lower
        stats.update({"min": min_vals, "max": max_vals, "feature_range": feature_range})
    else:
        raise ValueError("Normalization method must be 'zscore' or 'minmax'.")

    normalized_df = df.copy()
    normalized_df[features] = normalized
    return normalized_df, stats


def apply_normalization(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    *,
    epsilon: Optional[float] = None,
) -> pd.DataFrame:
    """Apply stored normalization statistics to a new dataframe."""

    feature_columns = stats.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    epsilon = epsilon if epsilon is not None else stats.get("epsilon", 1e-8)

    features = list(feature_columns)
    missing = set(features) - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame is missing columns required for normalization: {sorted(missing)}")

    values = df[features].astype(float)
    method = stats.get("method", "zscore").lower()

    if method == "zscore":
        mean = stats["mean"]
        std = stats["std"].replace(0.0, epsilon)
        normalized = (values - mean) / std
    elif method in {"minmax", "min-max"}:
        min_vals = stats["min"]
        max_vals = stats["max"]
        span = (max_vals - min_vals).replace(0.0, epsilon)
        lower, upper = stats.get("feature_range", (0.0, 1.0))
        normalized = (values - min_vals) / span
        normalized = normalized * (upper - lower) + lower
    else:
        raise ValueError(f"Unsupported normalization method '{method}'.")

    normalized_df = df.copy()
    normalized_df[features] = normalized
    return normalized_df


def generate_sliding_windows(
    df: pd.DataFrame,
    *,
    feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
    target_column: str = "close",
    target_type: str = "return",
    window_size: int = 50,
    include_last_close: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Create sliding-window sequences and prediction targets."""

    features = list(feature_columns)
    values = df[features].to_numpy(dtype=np.float32)
    target_series = df[target_column].to_numpy(dtype=np.float32)

    if len(values) <= window_size:
        raise ValueError("Not enough rows to create at least one sliding window.")

    sequences = []
    targets = []
    last_closes = []
    for start_idx in range(len(values) - window_size):
        end_idx = start_idx + window_size
        sequences.append(values[start_idx:end_idx])

        next_close = target_series[end_idx]
        if target_type == "close":
            targets.append(next_close)
        elif target_type == "return":
            prev_close = target_series[end_idx - 1]
            denom = prev_close if prev_close != 0 else 1e-8
            targets.append((next_close - prev_close) / denom)
        else:
            raise ValueError("target_type must be 'close' or 'return'.")

        if include_last_close:
            last_closes.append(target_series[end_idx - 1])

    sequences_arr = np.asarray(sequences, dtype=np.float32)
    targets_arr = np.asarray(targets, dtype=np.float32)
    context: Dict[str, np.ndarray] = {}
    if include_last_close:
        context["last_close"] = np.asarray(last_closes, dtype=np.float32)

    return sequences_arr, targets_arr, context


def preprocess_ohlcv(
    file_path: str | Path,
    *,
    fmt: Optional[str] = None,
    normalization: str = "zscore",
    feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
    target_column: str = "close",
    target_type: str = "return",
    window_size: int = 50,
    dropna: bool = True,
    include_last_close: bool = False,
) -> PreprocessingResult:
    """Full preprocessing pipeline producing model-ready arrays."""

    df = load_ohlcv(file_path, fmt=fmt)
    if dropna:
        df = df.dropna().reset_index(drop=True)

    normalized_df, stats = normalize_candlesticks(
        df,
        feature_columns=feature_columns,
        method=normalization,
    )

    sequences, targets, context = generate_sliding_windows(
        normalized_df,
        feature_columns=feature_columns,
        target_column=target_column,
        target_type=target_type,
        window_size=window_size,
        include_last_close=include_last_close,
    )

    return PreprocessingResult(
        sequences=sequences,
        targets=targets,
        feature_columns=tuple(feature_columns),
        target_type=target_type,
        normalization_stats=stats,
        raw_dataframe=df,
        context=context,
    )
