# LFCC Feature Extraction Guide — ASVspoof 2019 LA

A complete walkthrough for downloading ASVspoof 2019 data, extracting LFCC features, visualizing spectrograms, and preparing data for model training.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Getting the Data](#2-getting-the-data)
3. [Understanding the Dataset Structure](#3-understanding-the-dataset-structure)
4. [Understanding the Code](#4-understanding-the-code)
5. [Running & Testing](#5-running--testing)
6. [Loading Features for Model Training](#6-loading-features-for-model-training)
7. [Sanity Checks](#7-sanity-checks)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

### Install Dependencies

```bash
# Core dependencies
pip install numpy scipy matplotlib torch torchaudio soundfile

# For HuggingFace data download (recommended)
pip install datasets

# Linux only — soundfile backend
sudo apt-get install libsndfile1
```

### Files You Should Have

You should have received three Python files:

| File | Purpose |
|------|---------|
| `quick_start.py` | Run first — downloads samples, extracts features, generates comparison plots |
| `lfcc_extraction.py` | Core extraction module — single file and batch processing |
| `GUIDE_lfcc_extraction.py` | Detailed reference with all command variations and code snippets |

---

## 2. Getting the Data

You have three options, from fastest to most complete.

### Option A: HuggingFace (Recommended — Fastest)

No manual download needed. The `quick_start.py` script handles everything:

```bash
python quick_start.py
```

This downloads 40 sample files (10 bonafide + 5 per each of the 6 attack types), extracts LFCC from all of them, and generates a comparison visualization.

To download more samples:

```bash
python quick_start.py --n_bonafide 50 --n_spoof 20
```

For the full dataset programmatically:

```python
from datasets import load_dataset

# Streams data — doesn't download all 7GB at once
dataset = load_dataset("LanceaKing/asvspoof2019", "LA", streaming=True)

# Or download fully for faster repeated access
dataset = load_dataset("LanceaKing/asvspoof2019", "LA")
```

### Option B: Kaggle (~4GB Download)

```bash
# Install kaggle CLI
pip install kaggle

# Set up API key:
#   1. Go to kaggle.com → Account → Create New API Token
#   2. Save kaggle.json to ~/.kaggle/kaggle.json

# Download
kaggle datasets download -d awsaf49/asvpoof-2019-dataset
unzip asvpoof-2019-dataset.zip -d asvspoof_data/
```

### Option C: Official Edinburgh DataShare (~7GB)

```bash
# Direct download link for the LA partition
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip
unzip LA.zip
```

---

## 3. Understanding the Dataset Structure

### Folder Layout

After downloading (Option B or C), your directory looks like this:

```
LA/
├── ASVspoof2019_LA_train/
│   └── flac/                    ← 25,380 .flac audio files
├── ASVspoof2019_LA_dev/
│   └── flac/                    ← 24,844 .flac audio files
├── ASVspoof2019_LA_eval/
│   └── flac/                    ← 71,237 .flac audio files
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

### Audio File Format

- Format: FLAC (Free Lossless Audio Codec) — no data loss
- Sample rate: 16,000 Hz
- Channels: Mono (single channel)
- Bit depth: 16-bit
- Duration: Varies (~1–10 seconds per utterance)
- Naming: `LA_T_XXXXXXX.flac` (train), `LA_D_XXXXXXX.flac` (dev), `LA_E_XXXXXXX.flac` (eval)

### Protocol File Format

Each line in the protocol file has 5 columns:

```
LA_0079  LA_T_6483968  -  A04  spoof
```

| Column | Name | Description |
|--------|------|-------------|
| 1 | `SPEAKER_ID` | `LA_0079` — which person's voice |
| 2 | `AUDIO_FILE_ID` | `LA_T_6483968` — maps to `LA_T_6483968.flac` |
| 3 | `-` | Placeholder (ignored) |
| 4 | `SYSTEM_ID` | `A04` = attack system, `-` = bonafide (real speech) |
| 5 | `LABEL` | `bonafide` or `spoof` |

### Attack Types in Training Set

| System ID | Type | Method |
|-----------|------|--------|
| A01 | TTS | Neural waveform model |
| A02 | TTS | Vocoder-based |
| A03 | TTS | Vocoder-based |
| A04 | TTS | Waveform concatenation |
| A05 | VC | Voice conversion |
| A06 | VC | Voice conversion |

### Dataset Split Sizes

| Split | Bonafide | Spoofed | Total | Attacks |
|-------|----------|---------|-------|---------|
| Train | 2,580 | 22,800 | 25,380 | A01–A06 (6 known) |
| Dev | 2,548 | 22,296 | 24,844 | A01–A06 (6 known) |
| Eval | 7,355 | 63,882 | 71,237 | A07–A19 (13, mostly unknown) |

### Creating a Balanced Subset

The 1:10 bonafide-to-spoof ratio is too imbalanced for preliminary experiments. Create a balanced subset:

```python
import random

bonafide = []
spoof_by_attack = {}

with open("ASVspoof2019.LA.cm.train.trn.txt") as f:
    for line in f:
        parts = line.strip().split()
        entry = {
            'speaker': parts[0],
            'audio_id': parts[1],
            'system': parts[3],
            'label': parts[4],
        }
        if entry['label'] == 'bonafide':
            bonafide.append(entry)
        else:
            spoof_by_attack.setdefault(entry['system'], []).append(entry)

# Keep all bonafide (2,580)
subset = bonafide.copy()

# Sample ~416 per attack type (2,500 / 6 ≈ 416)
for system_id, entries in spoof_by_attack.items():
    random.shuffle(entries)
    subset.extend(entries[:416])

random.shuffle(subset)
print(f"Balanced subset: {len(subset)} samples")
# Output: Balanced subset: ~5,076 samples
```

---

## 4. Understanding the Code

### `lfcc_extraction.py` — Code Structure

The file has 5 main sections:

#### Section 1: `extract_lfcc_manual()`

The core function. Implements the full LFCC pipeline from scratch with every step commented:

```
Input:  1D numpy array (64,000 samples for 4 sec at 16kHz)
        ↓
Step 1: Pre-emphasis         → boost high frequencies (y[n] = x[n] - 0.97·x[n-1])
Step 2: Framing              → chop into 20ms overlapping windows (10ms hop)
Step 3: Hamming window       → taper frame edges to reduce spectral leakage
Step 4: FFT                  → time domain → frequency domain (power spectrum)
Step 5: Linear filterbank    → 20 equally-spaced triangular filters (NOT mel-scale)
Step 6: Log compression      → compress dynamic range
Step 7: DCT                  → decorrelate into 20 cepstral coefficients
Step 8: Deltas               → append first and second temporal derivatives
        ↓
Output: 2D numpy array (60 coefficients × ~399 frames)
```

The key difference from MFCC: Step 5 uses a **linear** filterbank (equal spacing) instead of mel-scale (compressed at high frequencies). This preserves high-frequency resolution where spoofing artifacts often appear.

#### Section 2: `extract_lfcc_torchaudio()`

Same extraction using torchaudio's built-in `LFCC` transform. Fewer lines, faster for batch processing, but less transparent.

#### Section 3: Visualization Functions

- `plot_lfcc_spectrogram()` — plots just the LFCC spectrogram
- `plot_full_pipeline()` — shows all stages from waveform to CNN input

#### Section 4: `load_audio()`

Loads `.flac` or `.wav` files. Handles resampling, padding (short files), and truncation (long files) to ensure consistent output size.

#### Section 5: `process_asvspoof_dataset()`

Batch processor. Reads the protocol file, loops over all audio files, extracts LFCC, and saves each as a `.npy` file plus a `labels.json` mapping file.

### Output Shape Explained

For a 4-second utterance:

```
LFCC shape: (60, 399)

Row 0–19:   Static LFCC coefficients (spectral shape snapshot)
Row 20–39:  Delta coefficients (velocity — how fast each LFCC changes)
Row 40–59:  Delta-delta coefficients (acceleration of change)

Columns:    Time frames (one per 10ms → ~400 for 4 seconds)
```

The CNN receives this as a tensor of shape `(1, 60, 399)` — like a single-channel grayscale image where:
- Channels = 1 (grayscale)
- Height = 60 (coefficient index)
- Width = ~399 (time frames)

---

## 5. Running & Testing

### Test 1: Demo Mode (No Data Needed)

Run immediately to verify the code works:

```bash
python lfcc_extraction.py
```

This generates a synthetic speech-like signal and produces:
- `lfcc_pipeline.png` — full pipeline visualization (waveform → spectrogram → CNN input)
- `lfcc_spectrogram.png` — just the spectrogram
- Console output showing the feature shape

### Test 2: Quick Start with Real Data

Download samples and extract features in one command:

```bash
python quick_start.py
```

Output:
- `sample_data/flac/` — downloaded .flac audio files
- `sample_data/protocol.txt` — labels for each file
- `sample_features/` — extracted .npy feature files
- `sample_features/labels.json` — label mapping
- `sample_features/bonafide_vs_spoof_comparison.png` — visual comparison

### Test 3: Single Audio File

```bash
# Basic extraction
python lfcc_extraction.py --audio_path sample_data/flac/LA_T_6483968.flac

# Save features as .npy file
python lfcc_extraction.py \
    --audio_path sample_data/flac/LA_T_6483968.flac \
    --save_npy \
    --output_dir ./my_features

# Use torchaudio method instead of manual
python lfcc_extraction.py \
    --audio_path sample_data/flac/LA_T_6483968.flac \
    --method torchaudio

# Skip delta computation (just 20 static LFCCs)
python lfcc_extraction.py \
    --audio_path sample_data/flac/LA_T_6483968.flac \
    --no_deltas

# Save the plot to a specific path
python lfcc_extraction.py \
    --audio_path sample_data/flac/LA_T_6483968.flac \
    --save_plot my_spectrogram.png
```

### Test 4: Batch Process Full Dataset

```bash
# Process ALL training files (~10-20 min)
python lfcc_extraction.py --batch \
    --audio_dir /path/to/LA/ASVspoof2019_LA_train/flac/ \
    --protocol /path/to/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
    --output_dir ./features/train/

# Process only first 100 files (quick test)
python lfcc_extraction.py --batch \
    --audio_dir /path/to/LA/ASVspoof2019_LA_train/flac/ \
    --protocol /path/to/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
    --output_dir ./features/train/ \
    --max_samples 100

# Process development set
python lfcc_extraction.py --batch \
    --audio_dir /path/to/LA/ASVspoof2019_LA_dev/flac/ \
    --protocol /path/to/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
    --output_dir ./features/dev/
```

After batch processing:

```
features/train/
├── LA_T_1234567.npy      ← shape (60, ~399) each
├── LA_T_1234568.npy
├── LA_T_1234569.npy
├── ...
└── labels.json            ← maps audio_id → label + system_id
```

### Test 5: Compare Bonafide vs Spoof

A great experiment for your mid-report:

```bash
# Generate spectrogram for a bonafide sample
python lfcc_extraction.py \
    --audio_path /path/to/bonafide_file.flac \
    --save_plot bonafide_spectrogram.png

# Generate spectrogram for a spoof sample
python lfcc_extraction.py \
    --audio_path /path/to/spoof_file.flac \
    --save_plot spoof_spectrogram.png
```

Or use `quick_start.py` which automatically generates the comparison for all 6 attack types.

What to look for in the comparison:
- Spoof spectrograms often have **smoother patterns** (less natural micro-variation)
- Some attacks show **bandwidth cutoff** (energy drops above ~4kHz)
- Vocoder-based attacks may show **periodic artifacts** (regular stripe patterns)
- **Delta and delta-delta** sections may reveal unnatural transitions

---

## 6. Loading Features for Model Training

Once features are extracted, load them into a PyTorch Dataset:

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os


class ASVspoofLFCCDataset(Dataset):
    """PyTorch Dataset for loading pre-extracted LFCC features."""

    def __init__(self, feature_dir, max_frames=400):
        with open(os.path.join(feature_dir, "labels.json")) as f:
            self.labels = json.load(f)
        self.feature_dir = feature_dir
        self.max_frames = max_frames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        entry = self.labels[idx]

        # Load LFCC features
        lfcc = np.load(
            os.path.join(self.feature_dir, f"{entry['audio_id']}.npy")
        )  # shape: (60, n_frames)

        # Pad or truncate to fixed width
        n_coeffs, n_frames = lfcc.shape
        if n_frames > self.max_frames:
            lfcc = lfcc[:, :self.max_frames]
        elif n_frames < self.max_frames:
            pad_width = self.max_frames - n_frames
            lfcc = np.pad(lfcc, ((0, 0), (0, pad_width)))

        # Shape: (1, 60, 400) — single-channel "grayscale image"
        tensor = torch.FloatTensor(lfcc).unsqueeze(0)

        # Label: 1 = bonafide, 0 = spoof
        label = torch.LongTensor([entry['label']])[0]

        return tensor, label


# Usage
train_dataset = ASVspoofLFCCDataset("./features/train/")
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

# Each batch
for batch_features, batch_labels in train_loader:
    print(f"Features: {batch_features.shape}")  # (32, 1, 60, 400)
    print(f"Labels:   {batch_labels.shape}")     # (32,)
    break
```

The tensor shape `(batch_size, 1, 60, 400)` is what goes directly into your LCNN or ResNet model.

---

## 7. Sanity Checks

After extracting features, verify they're correct:

```python
import numpy as np
import os

feature_dir = "./features/train/"
npy_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]

for f in npy_files[:5]:
    lfcc = np.load(os.path.join(feature_dir, f))
    print(
        f"{f}: shape={lfcc.shape}, "
        f"min={lfcc.min():.3f}, max={lfcc.max():.3f}, "
        f"mean={lfcc.mean():.3f}, std={lfcc.std():.3f}"
    )
```

**What to verify:**

| Check | Expected | Problem if Wrong |
|-------|----------|------------------|
| Shape height | 60 (or 20 without deltas) | Delta computation failed |
| Shape width | ~200–600 (varies by duration) | Audio loading or framing issue |
| Values NOT all zeros | `abs(lfcc).sum() > 0` | Audio file is silent or corrupted |
| Values NOT all same | `std > 0.01` | Windowing or FFT failed |
| Reasonable range | min > -20, max < 20 | Pre-emphasis or log step issue |
| Static > delta-delta | First 20 rows have larger magnitude | Delta computation order wrong |

---

## 8. Troubleshooting

### "No module named 'soundfile'"

```bash
pip install soundfile
# On Linux also:
sudo apt-get install libsndfile1
```

### "RuntimeError: Couldn't find appropriate backend"

torchaudio needs soundfile as a backend for .flac files:

```bash
pip install soundfile
```

### Features Are All Zeros

Check that the audio file isn't silent or corrupted:

```python
import torchaudio
waveform, sr = torchaudio.load("file.flac")
print(f"Max amplitude: {waveform.abs().max()}")
# Should print something > 0
```

### Shape Mismatch When Feeding to Model

Make sure you're padding/truncating to the same `max_frames` for all files. The height (60) is always fixed, but the width (n_frames) varies with audio duration. The `ASVspoofLFCCDataset` class above handles this automatically.

### Out of Memory During Batch Processing

LFCC extraction is CPU-only (no GPU needed). If you're running out of RAM, process in smaller chunks:

```bash
python lfcc_extraction.py --batch --max_samples 5000 ...
# Then run again for the next 5000, etc.
```

### torchaudio vs Manual Give Slightly Different Results

This is expected. Minor differences come from different windowing implementations and floating-point ordering. Both are valid — the model will learn to work with either. Just be consistent: don't extract training data with one method and evaluation data with another.

---

## Quick Reference — Command Cheat Sheet

```bash
# Demo (no data needed)
python lfcc_extraction.py

# Quick start (downloads samples + extracts + visualizes)
python quick_start.py

# Single file
python lfcc_extraction.py --audio_path file.flac

# Single file with all options
python lfcc_extraction.py \
    --audio_path file.flac \
    --method torchaudio \
    --save_npy \
    --output_dir ./features \
    --save_plot my_plot.png

# Batch processing
python lfcc_extraction.py --batch \
    --audio_dir /path/to/flac/ \
    --protocol /path/to/protocol.txt \
    --output_dir ./features/train/ \
    --max_samples 5000 \
    --method manual
```
