"""Dataset utilities for candlestick sequence modelling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class CandlestickDataset(Dataset):
    """Torch ``Dataset`` wrapping preprocessed candlestick tensors."""

    def __init__(
        self,
        sequences: torch.Tensor | Sequence,
        targets: torch.Tensor | Sequence,
        *,
        sequence_dtype: torch.dtype = torch.float32,
        target_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        context: Optional[Mapping[str, Sequence]] = None,
    ) -> None:
        sequences_tensor = torch.as_tensor(sequences, dtype=sequence_dtype)
        targets_tensor = torch.as_tensor(targets, dtype=target_dtype)

        if targets_tensor.ndim == 1:
            targets_tensor = targets_tensor.unsqueeze(-1)

        if device is not None:
            sequences_tensor = sequences_tensor.to(device)
            targets_tensor = targets_tensor.to(device)

        self.sequences = sequences_tensor
        self.targets = targets_tensor
        self.context: Optional[Dict[str, torch.Tensor]] = None

        if context:
            context_tensors: Dict[str, torch.Tensor] = {}
            for key, values in context.items():
                tensor = torch.as_tensor(values, dtype=sequence_dtype)
                if tensor.shape[0] != self.sequences.shape[0]:
                    raise ValueError(
                        "Context tensor length does not match the number of sequences."
                    )
                context_tensors[key] = tensor
            if device is not None:
                context_tensors = {key: tensor.to(device) for key, tensor in context_tensors.items()}
            self.context = context_tensors

    def __len__(self) -> int:  # pragma: no cover - simple wrapper
        return self.sequences.size(0)

    def __getitem__(self, index: int):  # pragma: no cover - simple wrapper
        if self.context is None:
            return self.sequences[index], self.targets[index]
        sample_context = {key: tensor[index] for key, tensor in self.context.items()}
        return self.sequences[index], self.targets[index], sample_context


@dataclass
class DataLoaders:
    """Grouped PyTorch dataloaders for convenience."""

    train: DataLoader
    val: Optional[DataLoader] = None
    test: Optional[DataLoader] = None


def _split_lengths(
    total: int,
    *,
    val_split: float,
    test_split: float,
) -> Dict[str, int]:
    if not 0 <= val_split < 1:
        raise ValueError("val_split must be within [0, 1).")
    if not 0 <= test_split < 1:
        raise ValueError("test_split must be within [0, 1).")
    if val_split + test_split >= 1:
        raise ValueError("Validation and test splits must sum to less than 1.")

    val_len = int(total * val_split)
    test_len = int(total * test_split)
    train_len = total - val_len - test_len

    if train_len <= 0:
        raise ValueError("Not enough samples to allocate to the training split.")

    lengths: Dict[str, int] = {"train": train_len}
    if val_len > 0:
        lengths["val"] = val_len
    if test_len > 0:
        lengths["test"] = test_len
    return lengths


def create_dataloaders(
    sequences: torch.Tensor | Sequence,
    targets: torch.Tensor | Sequence,
    *,
    batch_size: int = 64,
    val_split: float = 0.2,
    test_split: float = 0.0,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    seed: Optional[int] = 42,
    pin_memory: bool = False,
    context: Optional[Mapping[str, Sequence]] = None,
) -> DataLoaders:
    """Create train/val(/test) dataloaders from sequences and targets."""

    dataset = CandlestickDataset(sequences, targets, context=context)
    total = len(dataset)

    lengths = _split_lengths(total, val_split=val_split, test_split=test_split)

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    subsets: Dict[str, Subset] = {}
    split_lengths = [lengths[name] for name in ("train", "val", "test") if name in lengths]
    split_names = [name for name in ("train", "val", "test") if name in lengths]
    random_subsets = random_split(dataset, split_lengths, generator=generator)
    for name, subset in zip(split_names, random_subsets):
        subsets[name] = subset

    def make_loader(subset: Dataset, *, shuffle_data: bool) -> DataLoader:
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle_data,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )

    train_loader = make_loader(subsets["train"], shuffle_data=shuffle)
    val_loader = make_loader(subsets["val"], shuffle_data=False) if "val" in subsets else None
    test_loader = make_loader(subsets["test"], shuffle_data=False) if "test" in subsets else None

    return DataLoaders(train=train_loader, val=val_loader, test=test_loader)
