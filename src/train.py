"""
train.py
========
Training Pipeline for Deep Learning Music Generation Model.
Trains the MusicLSTM network on extracted MIDI note sequences,
evaluates validation loss, saves best model weights, and plots training loss.
"""

import os
import sys
import time
import json
import argparse

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset_builder import ensure_dataset
from src.preprocess import (
    extract_notes_from_dataset,
    build_vocabulary,
    prepare_sequences,
    get_dataloaders
)
from src.model import MusicLSTM
from src.visualizer import plot_training_history


def train_model(epochs: int = 30,
                batch_size: int = 64,
                seq_length: int = 32,
                learning_rate: float = 0.002,
                embedding_dim: int = 128,
                hidden_dim: int = 256,
                num_layers: int = 2,
                model_save_path: str = "models/best_model.pth",
                config_save_path: str = "models/model_config.json"):
    """
    Main training execution function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"  AI Music Generation - Training Pipeline")
    print(f"  Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"============================================================")

    # 1. Dataset verification & Preprocessing
    ensure_dataset()
    notes = extract_notes_from_dataset()
    note2idx, idx2note, vocab = build_vocabulary(notes)
    vocab_size = len(vocab)

    X, y = prepare_sequences(notes, note2idx, seq_length=seq_length)
    train_loader, val_loader = get_dataloaders(X, y, batch_size=batch_size, val_split=0.15)

    # 2. Model Initialization
    model = MusicLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.3
    ).to(device)

    print(f"[+] Initialized MusicLSTM with {model.count_parameters():,} trainable parameters.")

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 4. Training Loop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else avg_train_loss
        history["val_loss"].append(avg_val_loss)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        # Checkpoint saving
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            best_marker = " [* Best Model Saved]"
        else:
            best_marker = ""

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"| Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} "
              f"| LR: {current_lr:.6f} "
              f"| Time: {epoch_time:.2f}s{best_marker}")

    total_training_time = time.time() - start_time
    print(f"\n============================================================")
    print(f"[+] Training completed in {total_training_time:.2f} seconds!")
    print(f"[+] Best Validation Loss: {best_val_loss:.4f}")
    print(f"[+] Model checkpoint saved to: {model_save_path}")

    # Save model hyperparameters configuration
    config = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "seq_length": seq_length,
        "best_val_loss": best_val_loss,
        "epochs": epochs
    }
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[+] Model configuration saved to: {config_save_path}")

    # Plot and save loss history
    plot_training_history(history, save_path="models/training_loss.png")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Music Generation LSTM Model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--seq_length", type=int, default=32, help="Sequence length")
    parser.add_argument("--hidden_dim", type=int, default=256, help="LSTM hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
