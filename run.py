"""
run.py
======
Unified Master Launcher for AI Music Generation Studio.
CodeAlpha Artificial Intelligence Internship - Task 3.

Usage:
  python run.py --mode web                       # Launch Interactive Web Dashboard
  python run.py --mode generate --notes 100     # Generate new MIDI & WAV music
  python run.py --mode train --epochs 20         # Train the MusicLSTM neural network
  python run.py --mode setup                     # Build / verify MIDI dataset
  python run.py --mode synth --file <path.mid>   # Convert MIDI to WAV audio
"""

import os
import sys
import argparse

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def print_banner():
    banner = """
========================================================================
   ____          _        _    _       _             
  / ___|___   __| | ___  / \\  | |_ __ | |__   __ _   
 | |   / _ \\ / _` |/ _ \\/ _ \\ | | '_ \\| '_ \\ / _` |  
 | |__| (_) | (_| |  __/ ___ \\| | |_) | | | | (_| |  
  \\____\\___/ \\__,_|\\___/_/   \\_\\_| .__/|_| |_|\\__,_|  
                                 |_|                 
      ARTIFICIAL INTELLIGENCE INTERNSHIP - TASK 3
              AI MUSIC GENERATION WITH LSTM
========================================================================
"""
    print(banner)


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="CodeAlpha AI Music Generation Launcher")
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["setup", "train", "generate", "web", "synth"],
        help="Execution mode: setup | train | generate | web | synth"
    )
    # Generation args
    parser.add_argument("--notes", type=int, default=100, help="Number of notes to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Creativity temperature (0.2 - 1.5)")
    parser.add_argument("--bpm", type=int, default=110, help="Tempo in beats per minute")
    parser.add_argument("--play", action="store_true", help="Play audio immediately upon generation")

    # Training args
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")

    # Web args
    parser.add_argument("--port", type=int, default=5000, help="Web server port")

    # Audio synthesis args
    parser.add_argument("--file", type=str, default="", help="Path to MIDI file to synthesize to WAV")

    args = parser.parse_args()

    if args.mode == "setup":
        print("[*] Running dataset setup...")
        from src.dataset_builder import ensure_dataset
        ensure_dataset()
        print("[+] Dataset setup completed successfully.")

    elif args.mode == "train":
        print(f"[*] Starting model training ({args.epochs} epochs)...")
        from src.train import train_model
        train_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)

    elif args.mode == "generate":
        print(f"[*] Generating composition ({args.notes} notes, temp={args.temp}, BPM={args.bpm})...")
        from src.generate import MusicGenerator
        from src.audio_synth import play_audio
        gen = MusicGenerator()
        results = gen.generate_and_save(
            num_notes=args.notes,
            temperature=args.temp,
            bpm=args.bpm,
            synthesize_audio=True
        )
        print("\n[+] Music generation complete!")
        print(f"    - MIDI File:       {results['midi_path']}")
        print(f"    - Synthesized WAV: {results['wav_path']}")
        print(f"    - Piano Roll:      {results['roll_path']}")

        if args.play and results.get("wav_path"):
            play_audio(results["wav_path"])

    elif args.mode == "web":
        print(f"[*] Launching Web Application on http://127.0.0.1:{args.port}...")
        from app import app
        app.run(host="127.0.0.1", port=args.port, debug=False)

    elif args.mode == "synth":
        if not args.file or not os.path.exists(args.file):
            print(f"[!] Please provide a valid MIDI file using --file <path.mid>")
            sys.exit(1)
        from src.audio_synth import midi_to_wav
        out_wav = args.file.rsplit(".", 1)[0] + ".wav"
        print(f"[*] Synthesizing {args.file} -> {out_wav}...")
        midi_to_wav(args.file, out_wav)
        print(f"[+] WAV created: {out_wav}")


if __name__ == "__main__":
    main()

