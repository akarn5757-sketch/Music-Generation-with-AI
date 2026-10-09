"""
model.py
========
Deep Learning Neural Network Model for Music Generation.
Implements a multi-layer stacked LSTM network with token embeddings,
dropout regularization, and dense projection layers to learn complex
harmonic and temporal music patterns.
"""

import torch
import torch.nn as nn


class MusicLSTM(nn.Module):
    """
    Stacked LSTM architecture for symbolic music sequence prediction.

    Architecture:
      Embedding Layer (vocab_size -> embedding_dim)
      -> Multi-layer LSTM with Dropout
      -> Fully Connected Linear Layer (hidden_dim -> 256)
      -> ReLU Activation & Dropout
      -> Output Classification Layer (256 -> vocab_size)
    """
    def __init__(self, vocab_size: int,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.3):
        super(MusicLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # Regularization & Classification Head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass.
        Args:
          x: Tensor of shape (batch_size, seq_len) with token indices
          hidden: Optional tuple (h_0, c_0) for stateful inference
        Returns:
          logits: Tensor of shape (batch_size, vocab_size) for next-token prediction
                  (or (batch_size, seq_len, vocab_size) if full sequence desired)
          hidden: Updated LSTM hidden state tuple
        """
        # Shape: (batch_size, seq_len, embedding_dim)
        embeds = self.embedding(x)

        # Shape: out = (batch_size, seq_len, hidden_dim)
        out, hidden = self.lstm(embeds, hidden)

        # Take last time step for classification
        last_step = out[:, -1, :]  # Shape: (batch_size, hidden_dim)

        # Pass through dense head
        h = self.dropout(last_step)
        h = self.relu(self.fc1(h))
        h = self.dropout(h)
        logits = self.fc2(h)      # Shape: (batch_size, vocab_size)

        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """
        Initializes hidden and cell states with zeros.
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    def count_parameters(self) -> int:
        """
        Returns the number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    vocab_sz = 120
    model = MusicLSTM(vocab_size=vocab_sz, embedding_dim=128, hidden_dim=256, num_layers=3)
    dummy_input = torch.randint(0, vocab_sz, (16, 32))  # batch 16, seq_len 32
    logits, _ = model(dummy_input)
    print(f"Model Architecture:\n{model}\n")
    print(f"Trainable Parameters: {model.count_parameters():,}")
    print(f"Input shape: {dummy_input.shape} -> Output logits shape: {logits.shape}")
    print("[OK] Model forward pass verified successfully!")

