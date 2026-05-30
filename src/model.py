"""
CodeNet — a shallow autoencoder that maps a code-feature vector to a
compact embedding.

Training uses MSE reconstruction loss (forward() returns both the embedding
and its reconstruction).  For inference, call encode() directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CodeNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        embedding_size: int = 32,
    ) -> None:
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_size, embedding_size),
            nn.Tanh(),          # keeps embeddings in [-1, 1]
        )
        self._decoder = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the embedding (for inference)."""
        return self._encoder(x)

    def forward(self, x: torch.Tensor):
        """Return (embedding, reconstruction) for training."""
        embedding = self._encoder(x)
        reconstruction = self._decoder(embedding)
        return embedding, reconstruction

    def export_onnx(self, path: str) -> None:
        """Export the full autoencoder to ONNX."""
        dummy = torch.randn(1, self._encoder[0].in_features)
        torch.onnx.export(
            self,
            dummy,
            path,
            input_names=['input'],
            output_names=['embedding', 'reconstruction'],
            dynamic_axes={
                'input':          {0: 'batch'},
                'embedding':      {0: 'batch'},
                'reconstruction': {0: 'batch'},
            },
            opset_version=12,
        )


def create_model(input_size: int) -> CodeNet:
    return CodeNet(input_size=input_size)
