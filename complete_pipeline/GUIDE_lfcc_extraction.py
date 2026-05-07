"""
==============================================================================
STEP-BY-STEP GUIDE: LFCC Extraction for ASVspoof 2019
==============================================================================

This guide walks you through:
  1. Getting sample data (3 options)
  2. Understanding the code structure
  3. Running and testing
  4. Interpreting results

==============================================================================
STEP 1: GET THE DATA
==============================================================================

You have 3 options, from fastest to most complete:

------------------------------------------------------------------------
OPTION A: HuggingFace (RECOMMENDED — fastest, no manual download)
------------------------------------------------------------------------

    pip install datasets soundfile

    # Python script to download a few samples:
    from datasets import load_dataset

    # This downloads only the LA (Logical Access) partition
    # streaming=True means it won't download the entire 7GB at once
    dataset = load_dataset("LanceaKing/asvspoof2019", "LA", streaming=True)
    
    # Get first 10 training samples
    import soundfile as sf
    import os
    
    os.makedirs("sample_data/flac", exist_ok=True)
    
    for i, sample in enumerate(dataset["train"]):
        if i >= 10:
            break
        
        audio = sample["audio"]
        filename = sample["audio_file_name"]
        label = "bonafide" if sample["key"] == 0 else "spoof"
        system = sample["system_id"]
        
        # Save as .flac
        sf.write(
            f"sample_data/flac/{filename}.flac",
            audio["array"],
            audio["sampling_rate"]
        )
        
        print(f"  {filename} | {label:8s} | system: {system}")
    
    print(f"Saved 10 samples to sample_data/flac/")


------------------------------------------------------------------------
OPTION B: Kaggle (medium — ~4GB download, easier than Edinburgh)
------------------------------------------------------------------------

    # Install kaggle CLI
    pip install kaggle
    
    # Set up API key: 
    #   1. Go to kaggle.com → Account → Create New API Token
    #   2. Save kaggle.json to ~/.kaggle/kaggle.json
    
    # Download
    kaggle datasets download -d awsaf49/asvpoof-2019-dataset
    unzip asvpoof-2019-dataset.zip -d asvspoof_data/

    # After unzipping, your folder structure will look like:
    # asvspoof_data/
    # └── LA/
    #     ├── ASVspoof2019_LA_train/
    #     │   └── flac/              ← 25,380 .flac files
    #     ├── ASVspoof2019_LA_dev/
    #     │   └── flac/              ← 24,844 .flac files
    #     ├── ASVspoof2019_LA_eval/
    #     │   └── flac/              ← 71,237 .flac files
    #     └── ASVspoof2019_LA_cm_protocols/
    #         ├── ASVspoof2019.LA.cm.train.trn.txt
    #         ├── ASVspoof2019.LA.cm.dev.trl.txt
    #         └── ASVspoof2019.LA.cm.eval.trl.txt


------------------------------------------------------------------------
OPTION C: Official Edinburgh DataShare (full, ~7GB)
------------------------------------------------------------------------

    # Download from: https://datashare.ed.ac.uk/handle/10283/3336
    # Click "LA.zip" (Logical Access partition)
    # Or direct download:
    wget https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip
    unzip LA.zip
    
    # Same folder structure as Option B above


==============================================================================
STEP 2: UNDERSTAND THE PROTOCOL FILE
==============================================================================

The protocol file tells you which audio is bonafide and which is spoofed.
Each line looks like this:

    LA_0079  LA_T_6483968  -  A04  spoof

    Column 1: SPEAKER_ID    → LA_0079 (which person's voice)
    Column 2: AUDIO_FILE_ID → LA_T_6483968 (maps to LA_T_6483968.flac)
    Column 3: -             → ignored (placeholder)
    Column 4: SYSTEM_ID     → A04 (which TTS/VC system made this)
                               "-" means bonafide (no system, real speech)
                               A01-A06 are the 6 known attack types
    Column 5: LABEL         → "bonafide" or "spoof"

To create a balanced subset (as we discussed — ~2580 bonafide + ~2500 spoof
stratified by attack type), use this script:

    import random
    
    bonafide = []
    spoof_by_attack = {}  # {system_id: [entries]}
    
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
    
    # Take all bonafide (2580)
    subset = bonafide.copy()
    
    # Take ~416 per attack type (2500 / 6 attacks ≈ 416)
    for system_id, entries in spoof_by_attack.items():
        random.shuffle(entries)
        subset.extend(entries[:416])
    
    random.shuffle(subset)
    print(f"Balanced subset: {len(subset)} samples")
    print(f"  Bonafide: {sum(1 for s in subset if s['label']=='bonafide')}")
    print(f"  Spoof:    {sum(1 for s in subset if s['label']=='spoof')}")


==============================================================================
STEP 3: UNDERSTAND THE CODE STRUCTURE
==============================================================================

The lfcc_extraction.py file has 5 main sections:

┌─────────────────────────────────────────────────────────────────┐
│  SECTION 1: extract_lfcc_manual()                               │
│  ─────────────────────────────────                              │
│  The core function. Takes a 1D numpy array of audio samples     │
│  and returns a 2D LFCC matrix. Every step is commented.         │
│                                                                 │
│  Input:  waveform (64000 samples for 4 sec at 16kHz)           │
│  Output: lfcc matrix (60 x ~399)                                │
│                                                                 │
│  Pipeline inside this function:                                 │
│    waveform                                                     │
│      → pre-emphasis (boost high freq)                           │
│      → framing (chop into 20ms overlapping windows)             │
│      → hamming window (taper edges)                             │
│      → FFT (time → frequency domain)                            │
│      → linear filterbank (20 equally-spaced triangular filters) │
│      → log compression                                          │
│      → DCT (decorrelate → 20 cepstral coefficients)            │
│      → deltas + delta-deltas (temporal derivatives)             │
│    = 2D matrix (60 rows × ~399 columns)                        │
├─────────────────────────────────────────────────────────────────┤
│  SECTION 2: extract_lfcc_torchaudio()                           │
│  ─────────────────────────────────────                          │
│  Same thing but uses torchaudio's built-in LFCC transform.     │
│  Fewer lines, faster, but less transparent.                     │
│  Use this for batch processing.                                 │
├─────────────────────────────────────────────────────────────────┤
│  SECTION 3: plot_lfcc_spectrogram() / plot_full_pipeline()      │
│  ─────────────────────────────────────────────────              │
│  Visualization functions. plot_full_pipeline() shows all        │
│  stages from waveform → spectrogram → CNN input.                │
├─────────────────────────────────────────────────────────────────┤
│  SECTION 4: load_audio()                                        │
│  ────────────────────────                                       │
│  Loads .flac or .wav files. Handles:                            │
│    - Resampling if SR ≠ 16kHz                                  │
│    - Padding short files (< 4 sec) with zeros                  │
│    - Truncating long files (> 4 sec)                            │
│  This ensures every audio file produces the same-size output.   │
├─────────────────────────────────────────────────────────────────┤
│  SECTION 5: process_asvspoof_dataset()                          │
│  ─────────────────────────────────────                          │
│  Batch processor. Reads the protocol file, loops over all       │
│  audio files, extracts LFCC, saves each as a .npy file.        │
│  Also saves labels.json mapping audio_id → label.               │
└─────────────────────────────────────────────────────────────────┘


==============================================================================
STEP 4: INSTALL DEPENDENCIES
==============================================================================

    pip install numpy scipy matplotlib torch torchaudio soundfile

    # If using HuggingFace for data:
    pip install datasets

    # If you get errors with soundfile on Linux:
    sudo apt-get install libsndfile1


==============================================================================
STEP 5: RUN AND TEST
==============================================================================

------------------------------------------------------------------------
TEST 1: Demo mode (no data needed — runs immediately)
------------------------------------------------------------------------

    python lfcc_extraction.py

    # This generates a synthetic signal and extracts LFCC from it.
    # Output:
    #   - Prints shape info: (60, 399) = 60 coefficients × 399 frames
    #   - Saves lfcc_pipeline.png showing the full pipeline visualization
    #   - Saves lfcc_spectrogram.png showing just the spectrogram


------------------------------------------------------------------------
TEST 2: Single real audio file
------------------------------------------------------------------------

    # After downloading sample data (Option A above):
    python lfcc_extraction.py --audio_path sample_data/flac/LA_T_6483968.flac

    # To also save the features as .npy:
    python lfcc_extraction.py \
        --audio_path sample_data/flac/LA_T_6483968.flac \
        --save_npy \
        --output_dir ./my_features

    # To use torchaudio instead of manual extraction:
    python lfcc_extraction.py \
        --audio_path sample_data/flac/LA_T_6483968.flac \
        --method torchaudio

    # To skip deltas (just 20 static LFCCs, not 60):
    python lfcc_extraction.py \
        --audio_path sample_data/flac/LA_T_6483968.flac \
        --no_deltas


------------------------------------------------------------------------
TEST 3: Compare bonafide vs spoof visually
------------------------------------------------------------------------

    # This is a great experiment for your mid-report!
    # Extract LFCC from one bonafide and one spoof file, then compare.

    python lfcc_extraction.py \
        --audio_path sample_data/flac/LA_T_1234567.flac \
        --save_plot bonafide_example.png

    python lfcc_extraction.py \
        --audio_path sample_data/flac/LA_T_9876543.flac \
        --save_plot spoof_example.png

    # Look at the two images side by side — you may see:
    #   - Spoof has smoother spectrogram (less natural variation)
    #   - Spoof may show bandwidth cutoff (energy drops above ~4kHz)
    #   - Delta/delta-delta may show unnatural transitions


------------------------------------------------------------------------
TEST 4: Batch process the entire training set
------------------------------------------------------------------------

    # Process ALL training files (takes ~10-20 min depending on hardware):
    python lfcc_extraction.py --batch \
        --audio_dir /path/to/LA/ASVspoof2019_LA_train/flac/ \
        --protocol /path/to/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
        --output_dir ./features/train/

    # Process only first 100 files (quick test):
    python lfcc_extraction.py --batch \
        --audio_dir /path/to/LA/ASVspoof2019_LA_train/flac/ \
        --protocol /path/to/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
        --output_dir ./features/train/ \
        --max_samples 100

    # Process development set:
    python lfcc_extraction.py --batch \
        --audio_dir /path/to/LA/ASVspoof2019_LA_dev/flac/ \
        --protocol /path/to/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
        --output_dir ./features/dev/

    # After batch processing, your output directory contains:
    # features/train/
    # ├── LA_T_1234567.npy    ← shape (60, 399) each
    # ├── LA_T_1234568.npy
    # ├── LA_T_1234569.npy
    # ├── ...
    # └── labels.json          ← maps audio_id → label + system_id


==============================================================================
STEP 6: LOAD EXTRACTED FEATURES FOR MODEL TRAINING
==============================================================================

Once features are extracted, here's how your teammate (or you) loads them
into a PyTorch Dataset for model training:

    import torch
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import json
    import os

    class ASVspoofLFCCDataset(Dataset):
        def __init__(self, feature_dir, max_frames=400):
            # Load labels
            with open(os.path.join(feature_dir, "labels.json")) as f:
                self.labels = json.load(f)
            
            self.feature_dir = feature_dir
            self.max_frames = max_frames
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            entry = self.labels[idx]
            
            # Load LFCC features
            feature_path = os.path.join(
                self.feature_dir, f"{entry['audio_id']}.npy"
            )
            lfcc = np.load(feature_path)  # shape: (60, n_frames)
            
            # Pad or truncate to fixed width
            n_coeffs, n_frames = lfcc.shape
            if n_frames > self.max_frames:
                lfcc = lfcc[:, :self.max_frames]
            elif n_frames < self.max_frames:
                pad_width = self.max_frames - n_frames
                lfcc = np.pad(lfcc, ((0, 0), (0, pad_width)))
            
            # Convert to tensor
            # Shape: (1, 60, 400) — like a single-channel grayscale image
            tensor = torch.FloatTensor(lfcc).unsqueeze(0)
            
            # Label: 1 = bonafide, 0 = spoof
            label = torch.LongTensor([entry['label']])[0]
            
            return tensor, label
    
    # Usage:
    train_dataset = ASVspoofLFCCDataset("./features/train/")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )
    
    # Each batch:
    for batch_features, batch_labels in train_loader:
        print(f"Features: {batch_features.shape}")  # (32, 1, 60, 400)
        print(f"Labels:   {batch_labels.shape}")     # (32,)
        break


==============================================================================
STEP 7: VERIFY YOUR EXTRACTION IS CORRECT
==============================================================================

Run this quick sanity check after extracting features:

    import numpy as np
    import os
    
    feature_dir = "./features/train/"
    
    # Load a few random features
    npy_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
    
    for f in npy_files[:5]:
        lfcc = np.load(os.path.join(feature_dir, f))
        print(f"{f}: shape={lfcc.shape}, "
              f"min={lfcc.min():.3f}, max={lfcc.max():.3f}, "
              f"mean={lfcc.mean():.3f}")
    
    # Expected output:
    # LA_T_1234567.npy: shape=(60, 399), min=-8.234, max=5.123, mean=-1.456
    #
    # What to check:
    #   ✓ Shape is (60, N) where N is ~200-600 depending on utterance length
    #   ✓ Values are NOT all zeros (would mean audio loading failed)
    #   ✓ Values are NOT all the same (would mean windowing/FFT failed)
    #   ✓ Min/max are in reasonable range (typically -10 to +10)
    #   ✓ First 20 rows (static) have larger magnitude than last 20 (delta-delta)


==============================================================================
COMMON ISSUES AND FIXES
==============================================================================

ISSUE: "No module named 'soundfile'"
FIX:   pip install soundfile
       # On Linux also: sudo apt-get install libsndfile1

ISSUE: "RuntimeError: Couldn't find appropriate backend"
FIX:   pip install soundfile
       # torchaudio needs soundfile as backend for .flac files

ISSUE: Features are all zeros
FIX:   Check that the audio file isn't silent/corrupted:
       python -c "import torchaudio; w,sr = torchaudio.load('file.flac'); print(w.abs().max())"
       # Should print something > 0

ISSUE: Shape mismatch when feeding to model
FIX:   Make sure you're padding/truncating to the same max_frames
       for all files. The height (60) is always fixed, but the
       width (n_frames) varies with audio duration.

ISSUE: Out of memory during batch processing
FIX:   LFCC extraction is CPU-only, not GPU. If you're running out
       of RAM, process in chunks using --max_samples

"""
