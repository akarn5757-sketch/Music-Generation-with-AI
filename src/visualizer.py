"""
visualizer.py
=============
Provides visualization utilities for:
1. Training and validation loss / accuracy curves.
2. Symbolic piano roll representations of generated music compositions.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import music21 as m21


def plot_training_history(history: dict, save_path: str = "models/training_loss.png"):
    """
    Plots training loss and validation loss over epochs.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=150)
    
    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-o", label="Training Loss", linewidth=2, markersize=4)
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], "r--s", label="Validation Loss", linewidth=2, markersize=4)
    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Cross Entropy Loss", fontsize=12, fontweight="bold")
    ax1.set_title("Music Generation LSTM - Training Performance", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Saved training history chart to '{save_path}'.")
    return save_path


def plot_piano_roll(midi_path: str, save_path: str = "output/piano_roll.png"):
    """
    Generates an acoustic piano roll plot (pitch vs. time) from a MIDI file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    score = m21.converter.parse(midi_path)

    notes_data = []
    for element in score.flatten().notes:
        offset = float(element.offset)
        duration = float(element.quarterLength)
        if element.isChord:
            for p in element.pitches:
                notes_data.append((offset, duration, p.midi, p.nameWithOctave))
        elif element.isNote:
            notes_data.append((offset, duration, element.pitch.midi, element.pitch.nameWithOctave))

    if not notes_data:
        print(f"[!] Warning: No notes found in '{midi_path}' for piano roll.")
        return None

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    for offset, duration, midi_val, name in notes_data:
        rect = plt.Rectangle((offset, midi_val - 0.4), duration, 0.8,
                             facecolor="#6366f1", edgecolor="#312e81", alpha=0.85, linewidth=0.8)
        ax.add_patch(rect)

    all_offsets = [n[0] for n in notes_data]
    all_durs = [n[1] for n in notes_data]
    all_midis = [n[2] for n in notes_data]

    max_time = max(o + d for o, d in zip(all_offsets, all_durs)) + 2
    min_midi = max(min(all_midis) - 2, 21)
    max_midi = min(max(all_midis) + 2, 108)

    ax.set_xlim(0, max_time)
    ax.set_ylim(min_midi, max_midi)
    ax.set_xlabel("Time (Quarter Notes / Beats)", fontsize=12, fontweight="bold")
    ax.set_ylabel("MIDI Note Pitch (Semitones)", fontsize=12, fontweight="bold")
    ax.set_title(f"Piano Roll: {os.path.basename(midi_path)}", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Saved piano roll visualization to '{save_path}'.")
    return save_path

