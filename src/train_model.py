import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from model import CodeNet

def load_dataset(np_path: str):
    """Load feature dataset saved as .npy and return a TensorDataset."""
    data = np.load(np_path)
    tensor = torch.from_numpy(data).float()
    return torch.utils.data.TensorDataset(tensor, tensor)  # autoencoder target is same as input

def train_model(dataset_path: str, epochs: int = 10, batch_size: int = 32, output_path: str = "model.onnx"):
    """Train an auto‑encoder model on the provided dataset and export to ONNX.

    Args:
        dataset_path: Path to .npy file containing feature vectors.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        output_path: Destination path for exported ONNX model.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset = load_dataset(dataset_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Assume feature dimension from first sample
    sample_input, _ = dataset[0]
    input_dim = sample_input.shape[0]
    model = CodeNet(input_size=input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            embedding, reconstruction = model(batch_x)
            loss = criterion(reconstruction, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}")

    # Export to ONNX
    model.export_onnx(output_path)
    print(f"Model exported to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train ADS CodeDetective model")
    parser.add_argument("dataset", help="Path to .npy dataset file")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("-o", "--output", default="model.onnx", help="Output ONNX model file")
    args = parser.parse_args()
    train_model(args.dataset, epochs=args.epochs, batch_size=args.batch_size, output_path=args.output)
