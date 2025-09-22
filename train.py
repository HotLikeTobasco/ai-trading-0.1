"""Training script for the candlestick predictor."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import torch
from torch import nn

from data.preprocessing import DEFAULT_FEATURE_COLUMNS, preprocess_ohlcv
from datasets.candlestick_dataset import create_dataloaders
from models.candlestick_predictor import CandlestickPredictor, build_model_from_config

try:  # Optional dependency used when loading YAML configurations
    import yaml
except Exception:  # pragma: no cover - gracefully handle optional dependency
    yaml = None


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a candlestick forecasting model.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML configuration file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override the optimizer learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Computation device to use (cpu, cuda, etc.).")
    parser.add_argument("--verbose", action="store_true", help="Increase logging verbosity.")
    return parser.parse_args()


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    default_config: Dict[str, Any] = {
        "data": {
            "path": "data/sample.csv",
            "format": None,
            "features": list(DEFAULT_FEATURE_COLUMNS),
            "normalization": "zscore",
            "target": "return",
            "target_column": "close",
            "window_size": 50,
            "dropna": True,
        },
        "model": {
            "type": "lstm",
            "input_size": len(DEFAULT_FEATURE_COLUMNS),
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "output_size": 1,
            "bidirectional": False,
        },
        "training": {
            "epochs": 20,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "seed": 42,
            "num_workers": 0,
            "drop_last": False,
            "shuffle": True,
        },
        "validation": {
            "split": 0.2,
            "metrics": ["mse", "mae", "directional_accuracy"],
        },
        "checkpoint": {
            "dir": "checkpoints",
            "save_best": True,
            "save_every": None,
        },
    }

    if not config_path:
        return default_config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file '{config_path}' does not exist.")

    if path.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError("Configuration file must be a YAML document (.yml or .yaml).")

    if yaml is None:
        raise ImportError("PyYAML is required to load YAML configuration files.")

    with path.open("r", encoding="utf-8") as handle:
        loaded_config = yaml.safe_load(handle) or {}

    return deep_update(default_config, loaded_config)


def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def prepare_device(requested_device: Optional[str] = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    learning_rate = config.get("learning_rate", 1e-3)
    weight_decay = config.get("weight_decay", 0.0)
    optimizer_name = config.get("optimizer", "adam").lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        momentum = config.get("momentum", 0.0)
        return torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")


def build_scheduler(optimizer: torch.optim.Optimizer, config: Mapping[str, Any]):
    scheduler_cfg = config.get("scheduler")
    if not scheduler_cfg:
        return None

    scheduler_type = scheduler_cfg.get("type", "steplr").lower()
    if scheduler_type == "steplr":
        step_size = scheduler_cfg.get("step_size", 10)
        gamma = scheduler_cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if scheduler_type == "cosineannealing":
        t_max = scheduler_cfg.get("t_max", config.get("epochs", 50))
        eta_min = scheduler_cfg.get("eta_min", 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    raise ValueError(f"Unsupported scheduler type '{scheduler_type}'.")


def unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            sequences, targets, context = batch
            return sequences, targets, context
        if len(batch) == 2:
            sequences, targets = batch
            return sequences, targets, None
    raise ValueError("Unexpected batch format returned by the dataloader.")


def train_one_epoch(
    model: CandlestickPredictor,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        sequences, targets, _ = unpack_batch(batch)
        sequences = sequences.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(sequences)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = sequences.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


def evaluate(
    model: CandlestickPredictor,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics: Iterable[str],
    *,
    target_type: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    context_accumulator: Dict[str, List[torch.Tensor]] = {}

    with torch.no_grad():
        for batch in dataloader:
            sequences, targets, context = unpack_batch(batch)
            sequences = sequences.to(device)
            targets = targets.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, targets)

            batch_size = sequences.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions_list.append(outputs.detach().cpu())
            targets_list.append(targets.detach().cpu())

            if context:
                for key, value in context.items():
                    tensor_value = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
                    context_accumulator.setdefault(key, []).append(tensor_value)

    metrics_result: Dict[str, float] = {}
    if total_samples:
        metrics_result["loss"] = total_loss / total_samples

    if not predictions_list:
        return metrics_result

    predictions_tensor = torch.cat(predictions_list, dim=0)
    targets_tensor = torch.cat(targets_list, dim=0)
    context_tensor = {key: torch.cat(values, dim=0) for key, values in context_accumulator.items()}

    for metric in metrics:
        metric_lower = metric.lower()
        if metric_lower == "mse":
            metrics_result["mse"] = torch.mean((predictions_tensor - targets_tensor) ** 2).item()
        elif metric_lower == "mae":
            metrics_result["mae"] = torch.mean(torch.abs(predictions_tensor - targets_tensor)).item()
        elif metric_lower == "directional_accuracy":
            try:
                metrics_result["directional_accuracy"] = directional_accuracy(
                    predictions_tensor, targets_tensor, target_type, context=context_tensor
                )
            except ValueError as exc:
                logging.warning("Skipping directional accuracy: %s", exc)
        else:
            logging.warning("Unknown metric '%s' requested.", metric)

    return metrics_result


def directional_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_type: str,
    *,
    context: Optional[Mapping[str, torch.Tensor]] = None,
    epsilon: float = 1e-8,
) -> float:
    preds = predictions.view(-1)
    trgs = targets.view(-1)

    if target_type == "return":
        return torch.mean((torch.sign(preds) == torch.sign(trgs)).float()).item()

    if target_type == "close":
        if not context or "last_close" not in context:
            raise ValueError("'last_close' context is required for directional accuracy on close targets.")
        last_close = context["last_close"].view(-1)
        denom = torch.where(torch.abs(last_close) < epsilon, torch.full_like(last_close, epsilon), last_close)
        predicted_return = (preds - last_close) / denom
        actual_return = (trgs - last_close) / denom
        return torch.mean((torch.sign(predicted_return) == torch.sign(actual_return)).float()).item()

    raise ValueError(f"Unsupported target_type '{target_type}' for directional accuracy.")


def save_checkpoint(
    model: CandlestickPredictor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, float],
    config: Mapping[str, Any],
    checkpoint_path: Path,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": dict(metrics),
        "config": dict(config),
    }
    torch.save(payload, checkpoint_path)
    logging.info("Saved checkpoint to %s", checkpoint_path)


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    config = load_config(args.config)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.device is not None:
        config.setdefault("runtime", {})["device"] = args.device

    data_cfg = config["data"]
    validation_cfg = config["validation"]
    training_cfg = config["training"]
    checkpoint_cfg = config.get("checkpoint", {})

    include_last_close = "directional_accuracy" in [metric.lower() for metric in validation_cfg.get("metrics", [])]

    logging.info("Loading and preprocessing data from %s", data_cfg["path"])
    preprocessing_result = preprocess_ohlcv(
        data_cfg["path"],
        fmt=data_cfg.get("format"),
        normalization=data_cfg.get("normalization", "zscore"),
        feature_columns=data_cfg.get("features", DEFAULT_FEATURE_COLUMNS),
        target_column=data_cfg.get("target_column", "close"),
        target_type=data_cfg.get("target", "return"),
        window_size=data_cfg.get("window_size", 50),
        dropna=data_cfg.get("dropna", True),
        include_last_close=include_last_close,
    )

    logging.info(
        "Created %d sequences with window size %d",
        len(preprocessing_result.sequences),
        data_cfg.get("window_size", 50),
    )

    device = prepare_device(config.get("runtime", {}).get("device"))
    logging.info("Using device: %s", device)

    dataloaders = create_dataloaders(
        preprocessing_result.sequences,
        preprocessing_result.targets,
        batch_size=training_cfg.get("batch_size", 64),
        val_split=validation_cfg.get("split", 0.2),
        test_split=validation_cfg.get("test_split", 0.0),
        shuffle=training_cfg.get("shuffle", True),
        num_workers=training_cfg.get("num_workers", 0),
        drop_last=training_cfg.get("drop_last", False),
        seed=training_cfg.get("seed"),
        pin_memory=training_cfg.get("pin_memory", False),
        context=preprocessing_result.context if preprocessing_result.context else None,
    )

    model = build_model_from_config(config["model"])
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = build_optimizer(model, training_cfg)
    scheduler = build_scheduler(optimizer, training_cfg)

    epochs = training_cfg.get("epochs", 20)
    best_val_loss = float("inf")
    best_checkpoint_path: Optional[Path] = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, dataloaders.train, optimizer, criterion, device)
        log_msg = f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.6f}"

        val_metrics = {}
        if dataloaders.val is not None:
            val_metrics = evaluate(
                model,
                dataloaders.val,
                criterion,
                device,
                validation_cfg.get("metrics", []),
                target_type=data_cfg.get("target", "return"),
            )
            if "loss" in val_metrics:
                log_msg += f", val_loss: {val_metrics['loss']:.6f}"
                if val_metrics["loss"] < best_val_loss and checkpoint_cfg.get("save_best", True):
                    best_val_loss = val_metrics["loss"]
                    best_checkpoint_path = Path(checkpoint_cfg.get("dir", "checkpoints")) / "best.pt"
                    save_checkpoint(model, optimizer, epoch, val_metrics, config, best_checkpoint_path)
            for metric_name, value in val_metrics.items():
                if metric_name != "loss":
                    log_msg += f", val_{metric_name}: {value:.6f}"

        logging.info(log_msg)

        if scheduler is not None:
            scheduler.step()

        save_every = checkpoint_cfg.get("save_every")
        if save_every and epoch % save_every == 0:
            path = Path(checkpoint_cfg.get("dir", "checkpoints")) / f"epoch_{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, val_metrics or {"train_loss": train_loss}, config, path)

    if best_checkpoint_path:
        logging.info("Best validation checkpoint saved to %s", best_checkpoint_path)

    if dataloaders.val is None:
        logging.info("No validation split was provided. Final training loss: %.6f", train_loss)

    if dataloaders.test is not None:
        test_metrics = evaluate(
            model,
            dataloaders.test,
            criterion,
            device,
            validation_cfg.get("metrics", []),
            target_type=data_cfg.get("target", "return"),
        )
        logging.info("Test metrics: %s", test_metrics)


if __name__ == "__main__":
    main()
