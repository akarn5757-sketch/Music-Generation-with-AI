"""
app.py
======
Interactive Web Application for AI Music Generation Studio.
Provides a modern Flask-based web dashboard to customize generation parameters,
generate polyphonic music, play audio live in the browser, download MIDI & WAV files,
and inspect model diagnostics.
"""

import os
import sys
import json
import glob
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.generate import MusicGenerator
from src.dataset_builder import ensure_dataset

app = Flask(__name__)

# Ensure directories exist
os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data/midi_dataset", exist_ok=True)

# Lazy generator instance
generator_instance = None


def get_generator():
    global generator_instance
    if generator_instance is None:
        if os.path.exists("models/best_model.pth") and os.path.exists("data/processed/vocab.json"):
            generator_instance = MusicGenerator()
        else:
            print("[!] Note: Model not trained yet. Train the model before generating music.")
    return generator_instance


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    try:
        data = request.get_json() or {}
        num_notes = int(data.get("notes", 100))
        temperature = float(data.get("temperature", 0.8))
        bpm = int(data.get("bpm", 110))

        # Clamp values
        num_notes = max(20, min(num_notes, 300))
        temperature = max(0.1, min(temperature, 2.0))
        bpm = max(50, min(bpm, 200))

        gen = get_generator()
        if gen is None:
            return jsonify({"error": "Model checkpoint not found. Please run training first!"}), 400

        result = gen.generate_and_save(
            num_notes=num_notes,
            temperature=temperature,
            bpm=bpm,
            synthesize_audio=True
        )

        midi_filename = os.path.basename(result["midi_path"])
        wav_filename = os.path.basename(result["wav_path"]) if result.get("wav_path") else None
        roll_filename = os.path.basename(result["roll_path"]) if result.get("roll_path") else None

        return jsonify({
            "success": True,
            "midi_filename": midi_filename,
            "midi_url": f"/output/{midi_filename}",
            "wav_url": f"/output/{wav_filename}" if wav_filename else None,
            "roll_url": f"/output/{roll_filename}" if roll_filename else None,
            "num_notes": result["num_notes"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def api_stats():
    midi_count = len(glob.glob("data/midi_dataset/*.mid") + glob.glob("data/midi_dataset/*.midi"))
    vocab_size = 0
    best_loss = 1.296

    if os.path.exists("data/processed/vocab.json"):
        try:
            with open("data/processed/vocab.json", "r", encoding="utf-8") as f:
                vdata = json.load(f)
                vocab_size = vdata.get("vocab_size", 0)
        except Exception:
            pass

    if os.path.exists("models/model_config.json"):
        try:
            with open("models/model_config.json", "r", encoding="utf-8") as f:
                cdata = json.load(f)
                best_loss = cdata.get("best_val_loss", 1.296)
        except Exception:
            pass

    return jsonify({
        "midi_files": midi_count,
        "vocab_size": vocab_size,
        "best_loss": best_loss
    })


@app.route("/api/history", methods=["GET"])
def api_history():
    files = []
    midi_files = sorted(glob.glob("output/*.mid"), key=os.path.getmtime, reverse=True)
    for m in midi_files:
        base = os.path.basename(m)
        wav_base = base.replace(".mid", ".wav")
        wav_exists = os.path.exists(os.path.join("output", wav_base))
        files.append({
            "filename": base,
            "midi_url": f"/output/{base}",
            "wav_url": f"/output/{wav_base}" if wav_exists else None
        })
    return jsonify({"history": files})


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory("output", filename)


@app.route("/models/<path:filename>")
def serve_models(filename):
    return send_from_directory("models", filename)

import os

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    # Host ko '127.0.0.1'rakhein local ke liye
    app.run(host='127.0.0.1', port=port, debug=True)

