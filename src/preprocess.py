"""
preprocess.py
=============
Extracts musical elements (notes, chords, rests) from MIDI files using music21,
encodes vocabulary, builds sliding-window sequences for training,
and prepares PyTorch DataLoaders.
"""

import os
import glob
import json
import pickle
from typing import List, Tuple, Dict
import music21 as m21
import torch
from torch.utils.data import Dataset, DataLoader


class MusicDataset(Dataset):
    """
    PyTorch Dataset wrapper for sequence-to-next-token pairs.
    """
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def parse_single_midi(file_path: str) -> List[str]:
    """
    Parses a single MIDI file into a sequence of note/chord string tokens.
    """
    notes = []
    try:
        midi = m21.converter.parse(file_path)
        notes_to_parse = None

        try:
            # Try to get parts (e.g. piano right hand or instrument parts)
            parts = m21.instrument.partitionByInstrument(midi)
            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notesAndRests
        except Exception:
            notes_to_parse = midi.flat.notesAndRests

        for element in notes_to_parse:
            if isinstance(element, m21.note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, m21.chord.Chord):
                # Sort pitches from low to high and join with '.'
                notes.append('.'.join(str(p) for p in sorted(element.pitches)))
            elif isinstance(element, m21.note.Rest):
                if element.quarterLength >= 0.5:
                    notes.append("rest")
    except Exception as e:
        print(f"[!] Warning: Error parsing {os.path.basename(file_path)}: {e}")

    return notes


def extract_notes_from_dataset(midi_dir: str = "data/midi_dataset",
                              cache_file: str = "data/processed/notes.pkl",
                              force_reload: bool = False) -> List[str]:
    """
    Parses all MIDI files in the given directory and caches the resulting tokens list.
    """
    if os.path.exists(cache_file) and not force_reload:
        print(f"[*] Loading cached notes from '{cache_file}'...")
        with open(cache_file, "rb") as f:
            notes = pickle.load(f)
        print(f"[+] Loaded {len(notes)} note tokens from cache.")
        return notes

    midi_files = glob.glob(os.path.join(midi_dir, "*.mid")) + glob.glob(os.path.join(midi_dir, "*.midi"))
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in '{midi_dir}'. Run dataset_builder first.")

    print(f"[*] Parsing {len(midi_files)} MIDI files from '{midi_dir}'...")
    notes = []
    for i, file_path in enumerate(midi_files):
        parsed = parse_single_midi(file_path)
        notes.extend(parsed)
        if (i + 1) % 5 == 0 or (i + 1) == len(midi_files):
            print(f"    Processed {i + 1}/{len(midi_files)} files ({len(notes)} tokens gathered)")

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(notes, f)
    print(f"[+] Successfully extracted {len(notes)} tokens and saved to '{cache_file}'.")
    return notes


def build_vocabulary(notes: List[str], vocab_file: str = "data/processed/vocab.json") -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    """
    Extracts sorted unique tokens, creates bi-directional mappings,
    and stores the vocabulary mapping to JSON.
    """
    vocab = sorted(list(set(notes)))
    note2idx = {note: idx for idx, note in enumerate(vocab)}
    idx2note = {idx: note for idx, note in enumerate(vocab)}

    os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
    vocab_payload = {
        "vocab_size": len(vocab),
        "vocab": vocab,
        "note2idx": note2idx,
        "idx2note": {str(k): v for k, v in idx2note.items()}
    }
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab_payload, f, indent=2)

    print(f"[+] Vocabulary built: {len(vocab)} unique tokens stored in '{vocab_file}'.")
    return note2idx, idx2note, vocab


def prepare_sequences(notes: List[str],
                      note2idx: Dict[str, int],
                      seq_length: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates input sequences and target tokens using a sliding window.
    X shape: (num_samples, seq_length)
    y shape: (num_samples,)
    """
    network_input = []
    network_output = []

    for i in range(len(notes) - seq_length):
        seq_in = notes[i:i + seq_length]
        seq_out = notes[i + seq_length]
        network_input.append([note2idx[char] for char in seq_in])
        network_output.append(note2idx[seq_out])

    X = torch.tensor(network_input, dtype=torch.long)
    y = torch.tensor(network_output, dtype=torch.long)

    print(f"[+] Created {len(network_input)} training sequence pairs of length {seq_length}.")
    return X, y


def get_dataloaders(X: torch.Tensor,
                    y: torch.Tensor,
                    batch_size: int = 64,
                    val_split: float = 0.15) -> Tuple[DataLoader, DataLoader]:
    """
    Splits sequences into train and validation sets and returns PyTorch DataLoaders.
    """
    total_samples = len(X)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size

    dataset = MusicDataset(X, y)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[+] DataLoaders ready: Train samples={train_size}, Val samples={val_size}, Batch size={batch_size}.")
    return train_loader, val_loader


if __name__ == "__main__":
    notes = extract_notes_from_dataset()
    note2idx, idx2note, vocab = build_vocabulary(notes)
    X, y = prepare_sequences(notes, note2idx, seq_length=32)
    train_loader, val_loader = get_dataloaders(X, y, batch_size=64)
    print(f"[OK] Preprocessing pipeline verified successfully!")
