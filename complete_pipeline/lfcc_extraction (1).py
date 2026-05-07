"""
LFCC Feature Extraction & Spectrogram Visualization
=====================================================
Extracts Linear Frequency Cepstral Coefficients (LFCCs) from audio files
and visualizes them as 2D spectrograms — the exact input format for 
CNN-based anti-spoofing models (e.g., LCNN on ASVspoof 2019 LA).

Usage:
    python lfcc_extraction.py --audio_path /path/to/audio.flac
    python lfcc_extraction.py --audio_path /path/to/audio.flac --save_npy --output_dir ./features

Requirements:
    pip install torch torchaudio numpy matplotlib librosa soundfile
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torchaudio
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# =============================================================================
# Core LFCC Extraction (from scratch — no library dependency)
# =============================================================================

def extract_lfcc_manual(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_filters: int = 20,
    n_ceps: int = 20,
    frame_length_ms: float = 20.0,
    frame_shift_ms: float = 10.0,
    n_fft: int = 512,
    pre_emphasis: float = 0.97,
    include_deltas: bool = True,
):
    """
    Extract LFCC features from raw waveform — step by step.
    
    This implements the exact pipeline used in ASVspoof 2019 baselines:
    1. Pre-emphasis
    2. Framing (overlapping windows)
    3. Hamming window
    4. FFT → power spectrum
    5. Linear filterbank (NOT mel — this is what makes LFCC different from MFCC)
    6. Log compression
    7. DCT → cepstral coefficients
    8. Delta + delta-delta (optional)
    
    Parameters
    ----------
    waveform : np.ndarray
        1D array of audio samples (mono, float32, range [-1, 1])
    sample_rate : int
        Sample rate in Hz (16000 for ASVspoof 2019)
    n_filters : int
        Number of linear filterbank channels (default: 20)
    n_ceps : int
        Number of cepstral coefficients to keep (default: 20)
    frame_length_ms : float
        Frame length in milliseconds (default: 20ms)
    frame_shift_ms : float
        Frame shift / hop in milliseconds (default: 10ms)
    n_fft : int
        FFT size (default: 512)
    pre_emphasis : float
        Pre-emphasis filter coefficient (default: 0.97)
    include_deltas : bool
        Whether to append delta and delta-delta features (default: True)
    
    Returns
    -------
    lfcc : np.ndarray
        2D array of shape (n_coeffs, n_frames)
        - Without deltas: (20, ~400) for a 4-sec utterance
        - With deltas: (60, ~400) for a 4-sec utterance
    """
    
    # -------------------------------------------------------------------------
    # Step 1: Pre-emphasis — boost high frequencies
    # y[n] = x[n] - alpha * x[n-1]
    # This compensates for the natural spectral tilt of speech where
    # low frequencies dominate. Spoofing artifacts often hide in 
    # high frequencies, so this step is important.
    # -------------------------------------------------------------------------
    emphasized = np.append(waveform[0], waveform[1:] - pre_emphasis * waveform[:-1])
    
    # -------------------------------------------------------------------------
    # Step 2: Framing — chop signal into overlapping windows
    # A 4-second clip at 16kHz = 64000 samples
    # With 20ms frames and 10ms hop: ~400 frames
    # -------------------------------------------------------------------------
    frame_length = int(sample_rate * frame_length_ms / 1000)  # 320 samples
    frame_shift = int(sample_rate * frame_shift_ms / 1000)    # 160 samples
    
    n_frames = 1 + (len(emphasized) - frame_length) // frame_shift
    
    # Create frame indices using stride trick for efficiency
    indices = (
        np.arange(frame_length)[None, :] + 
        np.arange(n_frames)[:, None] * frame_shift
    )
    frames = emphasized[indices]  # shape: (n_frames, frame_length)
    
    # -------------------------------------------------------------------------
    # Step 3: Hamming window — taper frame edges to reduce spectral leakage
    # Without windowing, the abrupt edges of each frame create artificial
    # high-frequency artifacts in the FFT output.
    # -------------------------------------------------------------------------
    hamming = np.hamming(frame_length)
    frames = frames * hamming
    
    # -------------------------------------------------------------------------
    # Step 4: FFT → power spectrum
    # Transform each frame from time domain to frequency domain.
    # We only need the positive frequencies (first n_fft/2 + 1 bins).
    # -------------------------------------------------------------------------
    mag_spectrum = np.abs(np.fft.rfft(frames, n=n_fft))  # (n_frames, n_fft/2+1)
    power_spectrum = (1.0 / n_fft) * (mag_spectrum ** 2)
    
    # -------------------------------------------------------------------------
    # Step 5: LINEAR filterbank — THIS IS THE KEY DIFFERENCE FROM MFCC
    # 
    # MFCC uses mel-scale filterbank (compressed at high frequencies)
    # LFCC uses LINEAR filterbank (equal spacing across ALL frequencies)
    # 
    # This preserves high-frequency resolution where spoofing artifacts
    # often appear (4-8kHz range), which mel-scale would compress away.
    # -------------------------------------------------------------------------
    low_freq = 0
    high_freq = sample_rate / 2  # Nyquist frequency (8000 Hz for 16kHz SR)
    
    # Linearly spaced filter center frequencies
    linear_points = np.linspace(low_freq, high_freq, n_filters + 2)
    
    # Convert Hz to FFT bin indices
    bin_points = np.floor((n_fft + 1) * linear_points / sample_rate).astype(int)
    
    # Build triangular filterbank matrix
    filterbank = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        # Rising slope of triangle
        for j in range(bin_points[i], bin_points[i + 1]):
            filterbank[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
        # Falling slope of triangle
        for j in range(bin_points[i + 1], bin_points[i + 2]):
            filterbank[i, j] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])
    
    # Apply filterbank to power spectrum
    filter_energies = np.dot(power_spectrum, filterbank.T)  # (n_frames, n_filters)
    
    # -------------------------------------------------------------------------
    # Step 6: Log compression
    # Compresses dynamic range. Also makes the features more Gaussian-like,
    # which helps downstream classifiers.
    # -------------------------------------------------------------------------
    filter_energies = np.where(filter_energies == 0, np.finfo(float).eps, filter_energies)
    log_energies = np.log(filter_energies)
    
    # -------------------------------------------------------------------------
    # Step 7: DCT (Discrete Cosine Transform)
    # Decorrelates the log filterbank energies into cepstral coefficients.
    # Lower coefficients = broad spectral shape (vocal tract)
    # Higher coefficients = fine spectral detail (where artifacts hide)
    # -------------------------------------------------------------------------
    from scipy.fft import dct
    cepstral = dct(log_energies, type=2, axis=1, norm='ortho')
    lfcc_static = cepstral[:, :n_ceps]  # Keep first n_ceps coefficients
    
    # Transpose to (n_ceps, n_frames) — standard format for CNN input
    lfcc_static = lfcc_static.T
    
    # -------------------------------------------------------------------------
    # Step 8: Delta and delta-delta (temporal derivatives)
    # Delta = how fast each coefficient is changing over time
    # Delta-delta = acceleration of change
    # Natural speech has smooth, predictable transitions.
    # Synthesized speech can have abrupt or unnaturally smooth transitions.
    # -------------------------------------------------------------------------
    if include_deltas:
        delta = compute_deltas(lfcc_static, width=2)
        delta_delta = compute_deltas(delta, width=2)
        lfcc = np.concatenate([lfcc_static, delta, delta_delta], axis=0)
    else:
        lfcc = lfcc_static
    
    return lfcc


def compute_deltas(features: np.ndarray, width: int = 2) -> np.ndarray:
    """
    Compute delta (first derivative) features using regression formula.
    
    For each frame t, delta is computed as:
        delta[t] = sum_{n=1}^{W} n * (c[t+n] - c[t-n]) / (2 * sum_{n=1}^{W} n^2)
    
    Parameters
    ----------
    features : np.ndarray, shape (n_features, n_frames)
    width : int
        Number of frames on each side to use for regression
    
    Returns
    -------
    deltas : np.ndarray, same shape as input
    """
    n_features, n_frames = features.shape
    denominator = 2 * sum(n ** 2 for n in range(1, width + 1))
    
    # Pad edges by repeating first/last frame
    padded = np.pad(features, ((0, 0), (width, width)), mode='edge')
    
    deltas = np.zeros_like(features)
    for t in range(n_frames):
        for n in range(1, width + 1):
            deltas[:, t] += n * (padded[:, t + width + n] - padded[:, t + width - n])
        deltas[:, t] /= denominator
    
    return deltas


# =============================================================================
# Alternative: LFCC extraction using torchaudio (if available)
# =============================================================================

def extract_lfcc_torchaudio(
    waveform_tensor: "torch.Tensor",
    sample_rate: int = 16000,
    n_lfcc: int = 20,
    n_filter: int = 20,
    n_fft: int = 512,
    include_deltas: bool = True,
) -> np.ndarray:
    """
    Extract LFCC using torchaudio's built-in LFCC transform.
    Simpler but less transparent than the manual version.
    """
    lfcc_transform = torchaudio.transforms.LFCC(
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
        n_filter=n_filter,
        n_fft=n_fft,
        speckwargs={
            "hop_length": int(sample_rate * 0.01),   # 10ms hop
            "win_length": int(sample_rate * 0.02),    # 20ms window
            "window_fn": torch.hamming_window,
        },
    )
    
    lfcc = lfcc_transform(waveform_tensor)  # (1, n_lfcc, n_frames)
    lfcc = lfcc.squeeze(0).numpy()          # (n_lfcc, n_frames)
    
    if include_deltas:
        delta_transform = torchaudio.transforms.ComputeDeltas()
        lfcc_tensor = torch.from_numpy(lfcc).unsqueeze(0)
        delta = delta_transform(lfcc_tensor).squeeze(0).numpy()
        delta_delta = delta_transform(torch.from_numpy(delta).unsqueeze(0)).squeeze(0).numpy()
        lfcc = np.concatenate([lfcc, delta, delta_delta], axis=0)
    
    return lfcc


# =============================================================================
# Visualization
# =============================================================================

def plot_lfcc_spectrogram(
    lfcc: np.ndarray,
    sample_rate: int = 16000,
    hop_ms: float = 10.0,
    title: str = "LFCC Spectrogram",
    save_path: str = None,
):
    """
    Visualize LFCC features as a 2D spectrogram — exactly what the CNN sees.
    
    Parameters
    ----------
    lfcc : np.ndarray, shape (n_coeffs, n_frames)
        LFCC features (with or without deltas)
    """
    n_coeffs, n_frames = lfcc.shape
    has_deltas = n_coeffs > 20
    n_ceps = n_coeffs // 3 if has_deltas else n_coeffs
    
    # Time axis in seconds
    duration = n_frames * hop_ms / 1000
    
    fig, axes = plt.subplots(
        3 if has_deltas else 1, 1,
        figsize=(14, 10 if has_deltas else 4),
        sharex=True,
    )
    
    if not has_deltas:
        axes = [axes]
    
    sections = [
        ("Static LFCC (c0–c19)", lfcc[:n_ceps]),
    ]
    if has_deltas:
        sections.append(("Delta (velocity)", lfcc[n_ceps:2*n_ceps]))
        sections.append(("Delta-delta (acceleration)", lfcc[2*n_ceps:]))
    
    cmaps = ['viridis', 'magma', 'inferno']
    
    for idx, (label, data) in enumerate(sections):
        ax = axes[idx]
        im = ax.imshow(
            data,
            aspect='auto',
            origin='lower',
            extent=[0, duration, 0, data.shape[0]],
            cmap=cmaps[idx],
            interpolation='nearest',
        )
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_yticks(np.arange(0, data.shape[0], 5))
        plt.colorbar(im, ax=ax, shrink=0.8, label='Coefficient value')
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=12)
    
    fig.suptitle(
        f'{title}\n'
        f'Shape: ({n_coeffs}, {n_frames}) — '
        f'{"with" if has_deltas else "without"} deltas — '
        f'Duration: {duration:.2f}s',
        fontsize=13,
        fontweight='bold',
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Spectrogram saved to: {save_path}")
    
    plt.show()
    plt.close()


def plot_full_pipeline(
    waveform: np.ndarray,
    lfcc: np.ndarray,
    sample_rate: int = 16000,
    hop_ms: float = 10.0,
    save_path: str = None,
):
    """
    Visualize the complete pipeline: waveform → LFCC spectrogram → CNN input.
    Shows what happens at each stage.
    """
    n_coeffs, n_frames = lfcc.shape
    n_ceps = n_coeffs // 3 if n_coeffs > 20 else n_coeffs
    duration = len(waveform) / sample_rate
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), 
                              gridspec_kw={'height_ratios': [1, 1.5, 1.5, 1.5, 1.5]})
    
    # 1. Raw waveform
    time_axis = np.linspace(0, duration, len(waveform))
    axes[0].plot(time_axis, waveform, linewidth=0.3, color='#1D9E75')
    axes[0].set_ylabel('Amplitude', fontweight='bold')
    axes[0].set_title('Step 1: Raw waveform', fontweight='bold', loc='left')
    axes[0].set_xlim(0, duration)
    
    # 2. Power spectrogram (intermediate — what FFT produces)
    frame_length = int(sample_rate * 0.02)
    frame_shift = int(sample_rate * 0.01)
    from scipy.signal import spectrogram as scipy_spectrogram
    f, t, Sxx = scipy_spectrogram(waveform, fs=sample_rate, 
                                    nperseg=frame_length, noverlap=frame_length-frame_shift,
                                    nfft=512)
    axes[1].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), cmap='viridis', shading='gouraud')
    axes[1].set_ylabel('Frequency (Hz)', fontweight='bold')
    axes[1].set_title('Step 5: Power spectrum (after FFT)', fontweight='bold', loc='left')
    axes[1].set_ylim(0, sample_rate / 2)
    
    # 3. Static LFCC
    lfcc_time = np.linspace(0, duration, n_frames)
    im3 = axes[2].imshow(lfcc[:n_ceps], aspect='auto', origin='lower',
                          extent=[0, duration, 0, n_ceps], cmap='viridis')
    axes[2].set_ylabel('LFCC index', fontweight='bold')
    axes[2].set_title('Step 7: Static LFCC coefficients (after DCT)', fontweight='bold', loc='left')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    # 4. Delta
    if n_coeffs > 20:
        im4 = axes[3].imshow(lfcc[n_ceps:2*n_ceps], aspect='auto', origin='lower',
                              extent=[0, duration, 0, n_ceps], cmap='magma')
        axes[3].set_ylabel('Delta index', fontweight='bold')
        axes[3].set_title('Step 8a: Delta (first derivative over time)', fontweight='bold', loc='left')
        plt.colorbar(im4, ax=axes[3], shrink=0.8)
        
        # 5. Full stacked LFCC (what CNN actually sees)
        im5 = axes[4].imshow(lfcc, aspect='auto', origin='lower',
                              extent=[0, duration, 0, n_coeffs], cmap='viridis')
        axes[4].set_ylabel('All coefficients', fontweight='bold')
        axes[4].set_title(
            f'Final CNN input: shape (1, {n_coeffs}, {n_frames})', 
            fontweight='bold', loc='left', color='#D85A30'
        )
        axes[4].axhline(y=n_ceps, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
        axes[4].axhline(y=2*n_ceps, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
        axes[4].text(duration * 0.02, n_ceps/2, 'Static', color='white', fontsize=9, va='center')
        axes[4].text(duration * 0.02, n_ceps*1.5, 'Delta', color='white', fontsize=9, va='center')
        axes[4].text(duration * 0.02, n_ceps*2.5, 'Delta-delta', color='white', fontsize=9, va='center')
        plt.colorbar(im5, ax=axes[4], shrink=0.8)
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=12)
    
    fig.suptitle('LFCC Extraction Pipeline — From Waveform to CNN Input',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pipeline visualization saved to: {save_path}")
    
    plt.show()
    plt.close()


# =============================================================================
# Audio loading utilities
# =============================================================================

def load_audio(audio_path: str, target_sr: int = 16000, max_duration: float = 4.0):
    """
    Load audio file and normalize to fixed length.
    
    For ASVspoof 2019 LA:
    - Files are FLAC format, 16kHz, mono
    - We pad/truncate to a fixed length for consistent CNN input size
    """
    if HAS_TORCH:
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        waveform_np = waveform.squeeze(0).numpy()
        waveform_tensor = waveform
    elif HAS_LIBROSA:
        waveform_np, sr = librosa.load(audio_path, sr=target_sr)
        waveform_tensor = None
    else:
        raise ImportError("Install either torchaudio or librosa")
    
    # Pad or truncate to fixed length
    target_length = int(target_sr * max_duration)
    
    if len(waveform_np) > target_length:
        waveform_np = waveform_np[:target_length]
    elif len(waveform_np) < target_length:
        waveform_np = np.pad(waveform_np, (0, target_length - len(waveform_np)))
    
    if HAS_TORCH:
        waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0)  # (1, samples)
    
    return waveform_np, waveform_tensor, target_sr


# =============================================================================
# Batch processing for ASVspoof dataset
# =============================================================================

def process_asvspoof_dataset(
    audio_dir: str,
    protocol_file: str,
    output_dir: str,
    max_samples: int = None,
    method: str = "manual",
):
    """
    Batch extract LFCC features from ASVspoof 2019 LA dataset.
    
    Parameters
    ----------
    audio_dir : str
        Path to directory containing .flac files
        e.g., "/data/ASVspoof2019/LA/ASVspoof2019_LA_train/flac/"
    protocol_file : str
        Path to protocol file
        e.g., "/data/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    output_dir : str
        Where to save extracted .npy feature files
    max_samples : int, optional
        Limit number of samples to process (for testing)
    method : str
        "manual" (from scratch) or "torchaudio" (using torchaudio.transforms.LFCC)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse protocol file
    # Format: SPEAKER_ID AUDIO_FILE_ID - SYSTEM_ID LABEL
    # Example: LA_0079 LA_T_6483968 - A04 spoof
    entries = []
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            speaker_id = parts[0]
            audio_id = parts[1]
            system_id = parts[3]
            label = parts[4]  # "bonafide" or "spoof"
            entries.append({
                'speaker_id': speaker_id,
                'audio_id': audio_id,
                'system_id': system_id,
                'label': label,
            })
    
    if max_samples:
        entries = entries[:max_samples]
    
    print(f"Processing {len(entries)} audio files...")
    print(f"Method: {method}")
    print(f"Output directory: {output_dir}")
    
    labels = []
    
    for i, entry in enumerate(entries):
        audio_path = os.path.join(audio_dir, f"{entry['audio_id']}.flac")
        
        if not os.path.exists(audio_path):
            print(f"  Warning: {audio_path} not found, skipping")
            continue
        
        # Load audio
        waveform_np, waveform_tensor, sr = load_audio(audio_path)
        
        # Extract LFCC
        if method == "torchaudio" and HAS_TORCH:
            lfcc = extract_lfcc_torchaudio(waveform_tensor, sample_rate=sr)
        else:
            lfcc = extract_lfcc_manual(waveform_np, sample_rate=sr)
        
        # Save features as .npy
        feature_path = os.path.join(output_dir, f"{entry['audio_id']}.npy")
        np.save(feature_path, lfcc)
        
        # Track labels
        labels.append({
            'audio_id': entry['audio_id'],
            'label': 1 if entry['label'] == 'bonafide' else 0,
            'system_id': entry['system_id'],
            'speaker_id': entry['speaker_id'],
        })
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(entries)} files")
    
    # Save labels
    import json
    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"\nDone! Features saved to {output_dir}")
    print(f"Labels saved to {labels_path}")
    print(f"Feature shape per file: {lfcc.shape}")
    
    return labels


# =============================================================================
# Main — Demo with synthetic audio if no file provided
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LFCC Feature Extraction for ASVspoof 2019")
    parser.add_argument("--audio_path", type=str, default=None,
                        help="Path to audio file (.flac or .wav)")
    parser.add_argument("--method", type=str, default="manual", choices=["manual", "torchaudio"],
                        help="Extraction method: 'manual' (from scratch) or 'torchaudio'")
    parser.add_argument("--save_npy", action="store_true",
                        help="Save extracted features as .npy file")
    parser.add_argument("--output_dir", type=str, default="./lfcc_features",
                        help="Directory to save features")
    parser.add_argument("--no_deltas", action="store_true",
                        help="Don't compute delta and delta-delta features")
    parser.add_argument("--save_plot", type=str, default=None,
                        help="Path to save spectrogram plot (e.g., lfcc_plot.png)")
    
    # Batch processing arguments
    parser.add_argument("--batch", action="store_true",
                        help="Batch process ASVspoof dataset")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Directory containing .flac files (for batch)")
    parser.add_argument("--protocol", type=str, default=None,
                        help="Path to protocol file (for batch)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process in batch mode")
    
    args = parser.parse_args()
    
    include_deltas = not args.no_deltas
    
    # ---- Batch mode ----
    if args.batch:
        if not args.audio_dir or not args.protocol:
            print("Error: --audio_dir and --protocol required for batch mode")
            return
        process_asvspoof_dataset(
            audio_dir=args.audio_dir,
            protocol_file=args.protocol,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            method=args.method,
        )
        return
    
    # ---- Single file or demo mode ----
    if args.audio_path:
        print(f"Loading audio: {args.audio_path}")
        waveform_np, waveform_tensor, sr = load_audio(args.audio_path)
    else:
        # Generate synthetic demo audio (speech-like signal)
        print("No audio file provided — generating synthetic demo signal")
        sr = 16000
        duration = 4.0
        t = np.linspace(0, duration, int(sr * duration))
        # Mix of frequencies to simulate speech harmonics
        waveform_np = (
            0.3 * np.sin(2 * np.pi * 150 * t) +       # fundamental
            0.2 * np.sin(2 * np.pi * 300 * t) +       # 2nd harmonic
            0.1 * np.sin(2 * np.pi * 450 * t) +       # 3rd harmonic
            0.05 * np.sin(2 * np.pi * 1200 * t) +     # formant-like
            0.03 * np.sin(2 * np.pi * 2500 * t) +     # higher formant
            0.02 * np.random.randn(len(t))             # noise
        ).astype(np.float32)
        waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0) if HAS_TORCH else None
    
    print(f"Audio duration: {len(waveform_np)/sr:.2f}s | Samples: {len(waveform_np)} | SR: {sr}Hz")
    
    # Extract LFCC
    print(f"\nExtracting LFCC features (method: {args.method})...")
    
    if args.method == "torchaudio" and HAS_TORCH:
        lfcc = extract_lfcc_torchaudio(waveform_tensor, sample_rate=sr, include_deltas=include_deltas)
        print("  Used: torchaudio.transforms.LFCC")
    else:
        lfcc = extract_lfcc_manual(waveform_np, sample_rate=sr, include_deltas=include_deltas)
        print("  Used: Manual implementation (from scratch)")
    
    print(f"\nLFCC shape: {lfcc.shape}")
    print(f"  - Coefficients (height): {lfcc.shape[0]}")
    print(f"  - Time frames (width):   {lfcc.shape[1]}")
    print(f"  - CNN input tensor: (1, {lfcc.shape[0]}, {lfcc.shape[1]})")
    
    if include_deltas:
        n_ceps = lfcc.shape[0] // 3
        print(f"  - Static LFCCs:   rows 0–{n_ceps-1}")
        print(f"  - Deltas:         rows {n_ceps}–{2*n_ceps-1}")
        print(f"  - Delta-deltas:   rows {2*n_ceps}–{3*n_ceps-1}")
    
    # Save features
    if args.save_npy:
        os.makedirs(args.output_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(args.audio_path or "demo"))[0]
        npy_path = os.path.join(args.output_dir, f"{filename}_lfcc.npy")
        np.save(npy_path, lfcc)
        print(f"\nFeatures saved: {npy_path}")
    
    # Visualize
    print("\nGenerating visualizations...")
    
    # Full pipeline visualization
    plot_full_pipeline(
        waveform_np, lfcc, sample_rate=sr,
        save_path=args.save_plot or "lfcc_pipeline.png"
    )
    
    # Just the spectrogram
    plot_lfcc_spectrogram(
        lfcc, sample_rate=sr,
        title="LFCC Spectrogram — CNN Input",
        save_path="lfcc_spectrogram.png" if not args.save_plot else None
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
