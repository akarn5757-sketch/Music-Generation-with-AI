"""
generate.py
===========
Music Generation Engine.
Loads the trained MusicLSTM model and vocabulary, samples new musical note
sequences using temperature-controlled stochastic sampling, converts sequences
to MIDI files, synthesizes WAV audio, and visualizes the generated piece.
"""

import os
import sys
import time
import json
import random
import argparse
from typing import List, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F
import music21 as m21

from src.model import MusicLSTM
from src.audio_synth import midi_to_wav, play_audio
from src.visualizer import plot_piano_roll


def sample_with_temperature(logits: torch.Tensor, temperature: float = 0.8) -> int:
    """
    Samples an index from logits scaled by temperature.
    - Low temp (0.2 - 0.6): High confidence, predictable harmonious notes.
    - Medium temp (0.7 - 1.0): Balanced musical creativity and structure.
    - High temp (> 1.0): Experimental, surprising note selections.
    """
    if temperature <= 1e-4:
        return int(torch.argmax(logits).item())

    # Scale logits by temperature
    scaled_logits = logits / max(temperature, 1e-4)
    probs = F.softmax(scaled_logits, dim=-1).cpu().numpy().squeeze()

    # Numerical stability check
    probs = np.nan_to_num(probs, nan=1.0 / len(probs))
    probs = probs / np.sum(probs)

    sampled_idx = np.random.choice(len(probs), p=probs)
    return int(sampled_idx)


def sequence_to_midi(notes_sequence: List[str],
                     output_path: str = "output/generated_music.mid",
                     bpm: int = 110,
                     instrument_name: str = "Acoustic Grand Piano") -> str:
    """
    Converts a sequence of note, chord, and rest string tokens into a MIDI file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stream = m21.stream.Stream()
    
    # Metadata
    stream.metadata = m21.metadata.Metadata(
        title="AI Generated Composition",
        composer="CodeAlpha AI Music Generator"
    )
    stream.append(m21.tempo.MetronomeMark(number=bpm))
    stream.append(m21.instrument.Piano())

    offset = 0.0

    for pattern in notes_sequence:
        pattern = pattern.strip()
        if not pattern:
            continue

        # 1. Rests
        if pattern.lower() == "rest":
            new_rest = m21.note.Rest(quarterLength=0.5)
            new_rest.offset = offset
            stream.append(new_rest)
            offset += 0.5

        # 2. Chords (notes separated by '.')
        elif "." in pattern or (pattern.isdigit() and len(pattern) > 2):
            chord_notes = []
            parts = pattern.split(".")
            for current_note in parts:
                try:
                    if current_note.isdigit():
                        p = m21.pitch.Pitch(midi=int(current_note))
                    else:
                        p = m21.pitch.Pitch(current_note)
                    new_note = m21.note.Note(p)
                    new_note.storedInstrument = m21.instrument.Piano()
                    chord_notes.append(new_note)
                except Exception:
                    continue

            if chord_notes:
                new_chord = m21.chord.Chord(chord_notes, quarterLength=1.0)
                new_chord.offset = offset
                stream.append(new_chord)
                offset += 0.75

        # 3. Single Note
        else:
            try:
                new_note = m21.note.Note(pattern, quarterLength=0.5)
                new_note.offset = offset
                new_note.storedInstrument = m21.instrument.Piano()
                stream.append(new_note)
                offset += 0.5
            except Exception:
                continue

    stream.write("midi", fp=output_path)
    print(f"[+] MIDI successfully written to: {output_path}")
    return output_path


class MusicGenerator:
    """
    Encapsulates loaded neural network model, vocabulary, and generation pipeline.
    """
    def __init__(self,
                 model_path: str = "models/best_model.pth",
                 config_path: str = "models/model_config.json",
                 vocab_path: str = "data/processed/vocab.json",
                 device: Optional[str] = None):
        
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load vocabulary
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file '{vocab_path}' not found. Please run training first.")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        self.vocab = vocab_data["vocab"]
        self.note2idx = vocab_data["note2idx"]
        self.idx2note = {int(k): v for k, v in vocab_data["idx2note"].items()}
        self.vocab_size = len(self.vocab)

        # Load config
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "embedding_dim": 128,
                "hidden_dim": 256,
                "num_layers": 2,
                "seq_length": 32
            }

        # Initialize model & load state dict
        self.model = MusicLSTM(
            vocab_size=self.vocab_size,
            embedding_dim=self.config.get("embedding_dim", 128),
            hidden_dim=self.config.get("hidden_dim", 256),
            num_layers=self.config.get("num_layers", 2),
            dropout=0.0
        ).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint '{model_path}' not found. Please run training first.")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"[+] Loaded MusicGenerator model from '{model_path}' on device '{self.device}'.")

    def generate(self,
                 num_notes: int = 120,
                 temperature: float = 0.8,
                 primer_seed: Optional[List[str]] = None) -> List[str]:
        """
        Generates a sequence of musical tokens given an initial seed.
        """
        seq_length = self.config.get("seq_length", 32)

        # Build initial seed
        if primer_seed and len(primer_seed) >= seq_length:
            seed_pattern = [self.note2idx.get(n, 0) for n in primer_seed[:seq_length]]
        else:
            # Random seed from vocabulary
            seed_pattern = [random.randint(0, self.vocab_size - 1) for _ in range(seq_length)]

        generated_sequence = []
        current_input = list(seed_pattern)

        with torch.no_grad():
            for _ in range(num_notes):
                # Prepare tensor: (1, seq_length)
                input_tensor = torch.tensor([current_input[-seq_length:]], dtype=torch.long, device=self.device)
                logits, _ = self.model(input_tensor)
                
                # Sample next token with temperature
                next_idx = sample_with_temperature(logits, temperature=temperature)
                next_token = self.idx2note[next_idx]

                generated_sequence.append(next_token)
                current_input.append(next_idx)

        return generated_sequence

    def generate_and_save(self,
                          num_notes: int = 100,
                          temperature: float = 0.8,
                          output_prefix: str = "output/generated_composition",
                          bpm: int = 110,
                          synthesize_audio: bool = True) -> dict:
        """
        Full pipeline: generates sequence, exports MIDI, converts to WAV audio,
        and plots piano roll.
        """
        timestamp = int(time.time())
        midi_path = f"{output_prefix}_{timestamp}.mid"
        wav_path = f"{output_prefix}_{timestamp}.wav"
        roll_path = f"{output_prefix}_{timestamp}_roll.png"

        print(f"[*] Generating {num_notes} notes at temperature={temperature}...")
        notes_seq = self.generate(num_notes=num_notes, temperature=temperature)

        # 1. MIDI Export
        sequence_to_midi(notes_seq, output_path=midi_path, bpm=bpm)

        # 2. Audio Synthesis
        if synthesize_audio:
            print(f"[*] Synthesizing WAV audio...")
            midi_to_wav(midi_path, wav_path, bpm=bpm)
            print(f"[+] Audio synthesized: {wav_path}")

        # 3. Piano roll visualization
        plot_piano_roll(midi_path, save_path=roll_path)

        return {
            "midi_path": midi_path,
            "wav_path": wav_path if synthesize_audio else None,
            "roll_path": roll_path,
            "num_notes": len(notes_seq),
            "sequence": notes_seq
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Music with Trained LSTM Model")
    parser.add_argument("--notes", type=int, default=100, help="Number of notes to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature (0.2 to 1.5)")
    parser.add_argument("--bpm", type=int, default=110, help="Tempo in BPM")
    parser.add_argument("--play", action="store_true", help="Play the generated music immediately")
    args = parser.parse_args()

    gen = MusicGenerator()
    results = gen.generate_and_save(num_notes=args.notes, temperature=args.temp, bpm=args.bpm)

    if args.play and results.get("wav_path"):
        play_audio(results["wav_path"])
