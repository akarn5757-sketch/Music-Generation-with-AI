"""
audio_synth.py
==============
High-fidelity algorithmic synthesizer for converting MIDI files to WAV audio.
Features realistic piano/acoustic timbre synthesis with additive harmonics,
ADSR envelope, and stereo acoustic resonance, enabling immediate in-browser
and local audio playback without needing external soundfonts.
"""

import os
import math
import wave
import struct
import numpy as np
import music21 as m21


def pitch_to_freq(pitch_str_or_midi):
    """
    Converts note pitch name (e.g., 'C4', 'A4') or MIDI number (60, 69) to frequency in Hz.
    A4 = 440 Hz = MIDI 69.
    """
    if isinstance(pitch_str_or_midi, (int, float)):
        midi_num = pitch_str_or_midi
    else:
        p = m21.pitch.Pitch(pitch_str_or_midi)
        midi_num = p.midi
    return 440.0 * (2.0 ** ((midi_num - 69.0) / 12.0))


def generate_piano_note(freq, duration_sec, sample_rate=44100, amplitude=0.4):
    """
    Generates a realistic piano/acoustic tone using additive synthesis
    with overtones and an ADSR envelope.
    """
    total_samples = int(duration_sec * sample_rate)
    if total_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    t = np.linspace(0, duration_sec, total_samples, endpoint=False, dtype=np.float32)

    # Harmonics: fundamental, octave, octave+fifth, 2 octaves, etc.
    # Higher harmonics decay faster
    h1 = np.sin(2.0 * np.pi * freq * t) * np.exp(-t * 2.5)
    h2 = 0.50 * np.sin(2.0 * np.pi * 2 * freq * t) * np.exp(-t * 4.0)
    h3 = 0.25 * np.sin(2.0 * np.pi * 3 * freq * t) * np.exp(-t * 6.5)
    h4 = 0.12 * np.sin(2.0 * np.pi * 4 * freq * t) * np.exp(-t * 9.0)
    h5 = 0.06 * np.sin(2.0 * np.pi * 5 * freq * t) * np.exp(-t * 12.0)
    
    waveform = (h1 + h2 + h3 + h4 + h5).astype(np.float32)

    # ADSR Envelope
    attack_samples = min(int(0.008 * sample_rate), total_samples // 4)
    release_samples = min(int(0.040 * sample_rate), total_samples // 4)

    envelope = np.ones(total_samples, dtype=np.float32)
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
    if release_samples > 0:
        envelope[-release_samples:] *= np.linspace(1.0, 0.0, release_samples)

    return waveform * envelope * amplitude


def midi_to_wav(midi_path: str, wav_path: str, bpm: float = 110.0, sample_rate: int = 44100):
    """
    Parses a MIDI file and synthesizes it into a 16-bit stereo WAV audio file.
    """
    midi_path = os.path.abspath(midi_path)
    wav_path = os.path.abspath(wav_path)
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)

    score = m21.converter.parse(midi_path)
    quarter_sec = 60.0 / bpm

    events = []
    max_end_time = 0.0

    # Extract all notes and chords with exact offset timing
    for element in score.flatten().notesAndRests:
        offset_sec = float(element.offset) * quarter_sec
        duration_sec = float(element.quarterLength) * quarter_sec

        if element.isChord:
            for p in element.pitches:
                freq = pitch_to_freq(p.midi)
                events.append((offset_sec, duration_sec, freq, 0.30))
                max_end_time = max(max_end_time, offset_sec + duration_sec)
        elif element.isNote:
            freq = pitch_to_freq(element.pitch.midi)
            events.append((offset_sec, duration_sec, freq, 0.40))
            max_end_time = max(max_end_time, offset_sec + duration_sec)

    # Extra decay tail
    total_duration = max_end_time + 1.5
    total_samples = int(total_duration * sample_rate)
    audio_left = np.zeros(total_samples, dtype=np.float32)
    audio_right = np.zeros(total_samples, dtype=np.float32)

    for offset_sec, duration_sec, freq, amp in events:
        start_idx = int(offset_sec * sample_rate)
        note_audio = generate_piano_note(freq, duration_sec + 0.3, sample_rate=sample_rate, amplitude=amp)
        end_idx = min(start_idx + len(note_audio), total_samples)
        actual_len = end_idx - start_idx
        if actual_len > 0:
            # Slight stereo panning based on frequency (lower notes left, higher right)
            pan = min(max((math.log2(freq / 261.63) + 1.5) / 3.0, 0.2), 0.8)
            audio_left[start_idx:end_idx] += note_audio[:actual_len] * (1.0 - pan)
            audio_right[start_idx:end_idx] += note_audio[:actual_len] * pan

    # Normalize audio to prevent clipping
    max_peak = max(np.max(np.abs(audio_left)), np.max(np.abs(audio_right)), 1e-6)
    if max_peak > 0.85:
        scaling = 0.85 / max_peak
        audio_left *= scaling
        audio_right *= scaling

    # Convert to 16-bit PCM
    pcm_left = (audio_left * 32767).astype(np.int16)
    pcm_right = (audio_right * 32767).astype(np.int16)

    # Interleave stereo
    stereo = np.empty((total_samples * 2,), dtype=np.int16)
    stereo[0::2] = pcm_left
    stereo[1::2] = pcm_right

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())

    return wav_path


def play_audio(file_path: str):
    """
    Plays audio or MIDI using pygame.mixer.
    """
    import pygame
    pygame.mixer.init()
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        print(f"[*] Playing {file_path}... Press Enter to stop.")
        input()
        pygame.mixer.music.stop()
    except Exception as e:
        print(f"[!] Playback notice: {e}")
    finally:
        pygame.mixer.quit()


if __name__ == "__main__":
    import sys
    test_file = "data/midi_dataset/bach_chorale_C_1.mid"
    out_wav = "output/test_synth.wav"
    if os.path.exists(test_file):
        print(f"[*] Converting {test_file} to {out_wav}...")
        midi_to_wav(test_file, out_wav)
        print(f"[+] Synthesis test succeeded: {out_wav} generated!")

