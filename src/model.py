import torch
import torch.nn as nn

class CodeNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, embedding_size: int = 32):
        super(CodeNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
            nn.Tanh() # Tanh to keep embeddings between -1 and 1
        )
        
        # In a real scenario, we would have a decoder here to train it as an Autoencoder
        # For this MVP, we just focus on the architecture
        
    def forward(self, x):
        return self.encoder(x)

def create_model(input_size: int) -> CodeNet:
    return CodeNet(input_size=input_size)
