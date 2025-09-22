"""Configurable RNN model for candlestick forecasting."""
from __future__ import annotations

from typing import Callable, List, Tuple

import torch
from torch import nn

HiddenState = Tuple[torch.Tensor, torch.Tensor] | torch.Tensor
HiddenStateHook = Callable[[torch.Tensor, HiddenState, torch.Tensor], None]


class CandlestickPredictor(nn.Module):
    """Sequence model built on top of an LSTM or GRU backbone."""

    def __init__(
        self,
        *,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        output_size: int = 1,
    ) -> None:
        super().__init__()

        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        rnn_kwargs = dict(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        if self.rnn_type == "lstm":
            self.rnn: nn.Module = nn.LSTM(**rnn_kwargs)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        else:
            raise ValueError("rnn_type must be either 'lstm' or 'gru'.")

        hidden_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_size)
        self.hidden_state_hooks: List[HiddenStateHook] = []

    def add_hidden_state_hook(self, hook: HiddenStateHook) -> None:
        """Register a callback executed after each forward pass."""

        self.hidden_state_hooks.append(hook)

    def clear_hidden_state_hooks(self) -> None:
        """Remove all registered hidden-state callbacks."""

        self.hidden_state_hooks.clear()

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        return_hidden: bool = False,
        return_sequences: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, HiddenState]:
        """Run a forward pass through the predictor."""

        outputs, hidden = self.rnn(inputs)

        for hook in self.hidden_state_hooks:
            hook(outputs, hidden, inputs)

        if return_sequences:
            features = outputs
        else:
            features = outputs[:, -1, :]

        predictions = self.head(self.dropout(features))

        if return_hidden:
            return predictions, hidden
        return predictions


def build_model_from_config(config: dict) -> CandlestickPredictor:
    """Instantiate a predictor from a configuration dictionary."""

    model_params = {
        "input_size": config.get("input_size", 5),
        "hidden_size": config.get("hidden_size", 128),
        "num_layers": config.get("num_layers", 2),
        "dropout": config.get("dropout", 0.2),
        "rnn_type": config.get("type", config.get("rnn_type", "lstm")),
        "bidirectional": config.get("bidirectional", False),
        "output_size": config.get("output_size", 1),
    }
    return CandlestickPredictor(**model_params)
