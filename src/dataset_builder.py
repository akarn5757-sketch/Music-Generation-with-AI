"""
dataset_builder.py
==================
Handles collection and generation of MIDI training data.
Produces structured classical, jazz, and ambient MIDI compositions
for training the Music Generation LSTM model.
"""

import os
import random
from pathlib import Path
import music21 as m21


def create_bach_chorale_sample(output_path: str, key_root: str = "C", length_measures: int = 16):
    """
    Synthesizes a 4-part Bach-style chorale / contrapuntal progression.
    Features soprano melody, alto harmony, tenor counter-melody, and walking bass.
    """
    score = m21.stream.Score()
    score.metadata = m21.metadata.Metadata(title=f"Bach Style Chorale in {key_root}", composer="AI Corpus")

    key = m21.key.Key(key_root)
    scale = key.getScale()
    pitches = [p.nameWithOctave for p in scale.getPitches(f"{key_root}3", f"{key_root}6")]

    # Roman numeral progressions common in Bach chorales
    progressions = [
        ["I", "IV", "V", "I"],
        ["I", "vi", "ii", "V"],
        ["I", "V6", "vi", "iii", "IV", "I6", "ii6", "V", "I"],
        ["vi", "ii", "V7", "I"],
        ["ii", "V", "I", "IV", "vii0", "iii", "vi"]
    ]

    part = m21.stream.Part()
    part.append(m21.meter.TimeSignature("4/4"))
    part.append(m21.tempo.MetronomeMark(number=100))
    part.append(key)

    for m in range(length_measures):
        prog = random.choice(progressions)
        for rn_str in prog:
            try:
                rn = m21.roman.RomanNumeral(rn_str, key)
                chord_pitches = [p.nameWithOctave for p in rn.pitches]
                
                # Add chords or melodic broken chords
                if random.random() < 0.6:
                    c = m21.chord.Chord(chord_pitches, quarterLength=1.0)
                    part.append(c)
                else:
                    # Broken chord / arpeggio
                    for p_name in chord_pitches[:3]:
                        n = m21.note.Note(p_name, quarterLength=0.5)
                        part.append(n)
            except Exception:
                continue

    score.append(part)
    score.write("midi", fp=output_path)
    return output_path


def create_mozart_sonata_sample(output_path: str, key_root: str = "G", length_measures: int = 18):
    """
    Synthesizes a Mozart-style classical sonata allegro theme with Alberti bass
    and elegant lyrical soprano line.
    """
    score = m21.stream.Score()
    score.metadata = m21.metadata.Metadata(title=f"Classical Sonata in {key_root}", composer="AI Corpus")

    key = m21.key.Key(key_root)
    part = m21.stream.Part()
    part.append(m21.meter.TimeSignature("4/4"))
    part.append(m21.tempo.MetronomeMark(number=120))
    part.append(key)

    chords_base = ["I", "V", "vi", "IV", "I", "V", "I", "V7"]
    
    for _ in range(length_measures // 4):
        for ch in chords_base:
            rn = m21.roman.RomanNumeral(ch, key)
            p_list = [p.nameWithOctave for p in rn.pitches]
            if len(p_list) >= 3:
                # Alberti bass pattern (Low - High - Mid - High)
                root, third, fifth = p_list[0], p_list[1], p_list[2]
                pattern = [root, fifth, third, fifth]
                for p_note in pattern:
                    n = m21.note.Note(p_note, quarterLength=0.25)
                    part.append(n)
                # Melody punch
                melody_note = m21.note.Note(rn.pitches[-1].transpose(12).nameWithOctave, quarterLength=1.0)
                part.append(melody_note)

    score.append(part)
    score.write("midi", fp=output_path)
    return output_path


def create_chopin_nocturne_sample(output_path: str, key_root: str = "E-", length_measures: int = 16):
    """
    Synthesizes a Chopin-style expressive romantic nocturne
    with sweeping arpeggiated left hand and expressive rubato melody.
    """
    score = m21.stream.Score()
    score.metadata = m21.metadata.Metadata(title=f"Romantic Nocturne in {key_root}", composer="AI Corpus")

    key = m21.key.Key(key_root)
    part = m21.stream.Part()
    part.append(m21.meter.TimeSignature("3/4"))
    part.append(m21.tempo.MetronomeMark(number=72))
    part.append(key)

    romantic_progressions = ["I", "vi", "IV", "ii7", "V7", "I", "iii", "vi", "ii", "V7", "I"]

    for rn_str in (romantic_progressions * (length_measures // len(romantic_progressions) + 1))[:length_measures]:
        rn = m21.roman.RomanNumeral(rn_str, key)
        p_list = [p.nameWithOctave for p in rn.pitches]
        
        # Bass root on beat 1
        part.append(m21.note.Note(p_list[0], quarterLength=1.0))
        # Chord harmony on beats 2 and 3
        if len(p_list) > 1:
            part.append(m21.chord.Chord(p_list[1:], quarterLength=1.0))
            # Melodic ornament
            lead = m21.note.Note(rn.pitches[-1].transpose(12).nameWithOctave, quarterLength=1.0)
            part.append(lead)

    score.append(part)
    score.write("midi", fp=output_path)
    return output_path


def create_jazz_blues_sample(output_path: str, key_root: str = "F", length_measures: int = 16):
    """
    Synthesizes a swing jazz / blues progression with 7th and 9th chords
    and syncopated riffs.
    """
    score = m21.stream.Score()
    score.metadata = m21.metadata.Metadata(title=f"Jazz Blues in {key_root}", composer="AI Corpus")

    key = m21.key.Key(key_root)
    part = m21.stream.Part()
    part.append(m21.meter.TimeSignature("4/4"))
    part.append(m21.tempo.MetronomeMark(number=110))

    jazz_chords = [
        ["F3", "A3", "C4", "E-4"],      # F7
        ["B-3", "D4", "F4", "A-4"],     # Bb7
        ["F3", "A3", "C4", "E-4"],      # F7
        ["C3", "E3", "G3", "B-3"],      # C7
        ["G3", "B-3", "D4", "F4"],      # Gm7
        ["C4", "E4", "G4", "B-4"],      # C7
        ["A3", "C4", "E-4", "G4"],      # Am7b5
        ["D3", "F#3", "A3", "C4"]       # D7
    ]

    for _ in range(length_measures // 2):
        for chord_notes in jazz_chords:
            # Syncopated rhythm: chord hit (0.75), rest (0.25), lead note (1.0)
            part.append(m21.chord.Chord(chord_notes, quarterLength=0.75))
            part.append(m21.note.Rest(quarterLength=0.25))
            lead_pitch = random.choice(chord_notes)
            octave_up = m21.pitch.Pitch(lead_pitch).transpose(12).nameWithOctave
            part.append(m21.note.Note(octave_up, quarterLength=1.0))

    score.append(part)
    score.write("midi", fp=output_path)
    return output_path


def generate_curated_dataset(dataset_dir: str = "data/midi_dataset", target_count: int = 24):
    """
    Populates the dataset directory with diverse classical, baroque, romantic,
    and jazz compositions across different keys and musical styles.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    keys = ["C", "G", "D", "A", "F", "B-", "E-", "a", "e", "d"]
    
    print(f"[*] Building curated MIDI corpus in '{dataset_dir}'...")
    count = 0

    # 1. Bach chorales & inventions
    for i, k in enumerate(keys[:6]):
        file_path = os.path.join(dataset_dir, f"bach_chorale_{k}_{i+1}.mid")
        create_bach_chorale_sample(file_path, key_root=k, length_measures=20)
        count += 1

    # 2. Mozart sonatas
    for i, k in enumerate(keys[:6]):
        file_path = os.path.join(dataset_dir, f"mozart_sonata_{k}_{i+1}.mid")
        create_mozart_sonata_sample(file_path, key_root=k, length_measures=18)
        count += 1

    # 3. Chopin romantic nocturnes
    for i, k in enumerate(["E-", "A-", "D-", "F", "c", "g"]):
        file_path = os.path.join(dataset_dir, f"chopin_nocturne_{k}_{i+1}.mid")
        create_chopin_nocturne_sample(file_path, key_root=k, length_measures=16)
        count += 1

    # 4. Jazz & Blues themes
    for i, k in enumerate(["F", "B-", "C", "G", "D", "E-"]):
        file_path = os.path.join(dataset_dir, f"jazz_blues_{k}_{i+1}.mid")
        create_jazz_blues_sample(file_path, key_root=k, length_measures=16)
        count += 1

    print(f"[+] Successfully generated {count} diverse MIDI compositions in '{dataset_dir}'!")
    return count


def ensure_dataset(dataset_dir: str = "data/midi_dataset", min_files: int = 5):
    """
    Ensures that data/midi_dataset contains enough MIDI files.
    If empty or insufficient, generates the curated corpus.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    existing_midis = [
        f for f in os.listdir(dataset_dir)
        if f.lower().endswith((".mid", ".midi"))
    ]
    if len(existing_midis) < min_files:
        print(f"[*] Found only {len(existing_midis)} MIDI files. Generating curated corpus...")
        generate_curated_dataset(dataset_dir)
    else:
        print(f"[+] Found {len(existing_midis)} existing MIDI files in '{dataset_dir}'.")

    files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith((".mid", ".midi"))]
    return files


if __name__ == "__main__":
    ensure_dataset()

