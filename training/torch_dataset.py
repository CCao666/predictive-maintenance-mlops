"""PyTorch dataset for fixed-length C-MAPSS sequences."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class RULSequenceDataset(Dataset):
    """Store sensor sequences and their scalar RUL targets as tensors."""

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        if sequences.ndim != 3:
            raise ValueError(
                "sequences must have shape "
                "(samples, sequence_length, features)"
            )

        if targets.ndim != 1:
            raise ValueError("targets must have shape (samples,)")

        if len(sequences) != len(targets):
            raise ValueError(
                "sequences and targets must contain the same number of samples"
            )

        self.sequences = torch.as_tensor(
            sequences,
            dtype=torch.float32,
        )
        self.targets = torch.as_tensor(
            targets,
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        """Return the number of sequence-target pairs."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one sequence and its scalar RUL target."""
        return self.sequences[index], self.targets[index]

