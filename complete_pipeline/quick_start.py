"""
Quick Start: Download ASVspoof 2019 samples + extract LFCC
=============================================================
Run this first to get sample data and verify everything works.

    pip install datasets soundfile numpy scipy matplotlib torch torchaudio
    python quick_start.py
"""

import os
import numpy as np
import json

# =============================================================================
# Step 1: Download sample data from HuggingFace
# =============================================================================

def download_samples(n_bonafide=10, n_spoof_per_attack=5, output_dir="sample_data"):
    """Download a small balanced sample from HuggingFace."""
    
    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError:
        print("ERROR: Install required packages:")
        print("  pip install datasets soundfile")
        return None
    
    print("=" * 60)
    print("STEP 1: Downloading samples from HuggingFace...")
    print("=" * 60)
    
    flac_dir = os.path.join(output_dir, "flac")
    os.makedirs(flac_dir, exist_ok=True)
    
    dataset = load_dataset("LanceaKing/asvspoof2019", "LA", split="train", streaming=True)
    
    bonafide_count = 0
    spoof_counts = {}  # {system_id: count}
    protocol_lines = []
    
    total_target = n_bonafide + n_spoof_per_attack * 6  # 6 attack types
    saved_count = 0
    
    for sample in dataset:
        audio_id = sample["audio_file_name"]
        speaker_id = sample["speaker_id"]
        system_id = sample["system_id"]
        is_bonafide = sample["key"] == 0
        label = "bonafide" if is_bonafide else "spoof"
        
        # Decide whether to keep this sample
        keep = False
        if is_bonafide and bonafide_count < n_bonafide:
            keep = True
            bonafide_count += 1
        elif not is_bonafide:
            count = spoof_counts.get(system_id, 0)
            if count < n_spoof_per_attack:
                keep = True
                spoof_counts[system_id] = count + 1
        
        if not keep:
            # Check if we have enough
            if bonafide_count >= n_bonafide:
                all_done = all(
                    spoof_counts.get(f"A0{i}", 0) >= n_spoof_per_attack
                    for i in range(1, 7)
                )
                if all_done:
                    break
            continue
        
        # Save audio
        audio = sample["audio"]
        sf.write(
            os.path.join(flac_dir, f"{audio_id}.flac"),
            audio["array"],
            audio["sampling_rate"],
        )
        
        # Build protocol line (same format as official ASVspoof protocol)
        system_for_protocol = "-" if is_bonafide else system_id
        protocol_lines.append(
            f"{speaker_id} {audio_id} - {system_for_protocol} {label}"
        )
        
        saved_count += 1
        if saved_count % 10 == 0:
            print(f"  Downloaded {saved_count} samples...")
    
    # Save protocol file
    protocol_path = os.path.join(output_dir, "protocol.txt")
    with open(protocol_path, "w") as f:
        f.write("\n".join(protocol_lines))
    
    print(f"\nDownload complete!")
    print(f"  Audio files:   {flac_dir}/ ({saved_count} files)")
    print(f"  Protocol file: {protocol_path}")
    print(f"  Bonafide: {bonafide_count}")
    for sys_id in sorted(spoof_counts.keys()):
        print(f"  Spoof {sys_id}: {spoof_counts[sys_id]}")
    
    return output_dir


# =============================================================================
# Step 2: Extract LFCC from all samples
# =============================================================================

def extract_all_features(data_dir="sample_data", output_dir="sample_features"):
    """Extract LFCC features from downloaded samples."""
    
    # Import our extraction module
    import sys
    # Try importing from same directory or from outputs
    try:
        from lfcc_extraction import extract_lfcc_manual, load_audio, compute_deltas
    except ImportError:
        # Add the directory containing lfcc_extraction.py to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from lfcc_extraction import extract_lfcc_manual, load_audio, compute_deltas
    
    print("\n" + "=" * 60)
    print("STEP 2: Extracting LFCC features...")
    print("=" * 60)
    
    flac_dir = os.path.join(data_dir, "flac")
    protocol_path = os.path.join(data_dir, "protocol.txt")
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse protocol
    entries = []
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                entries.append({
                    'speaker_id': parts[0],
                    'audio_id': parts[1],
                    'system_id': parts[3],
                    'label': parts[4],
                })
    
    labels = []
    shapes = []
    
    for i, entry in enumerate(entries):
        audio_path = os.path.join(flac_dir, f"{entry['audio_id']}.flac")
        
        if not os.path.exists(audio_path):
            print(f"  Skipping {entry['audio_id']} — file not found")
            continue
        
        # Load and extract
        waveform_np, _, sr = load_audio(audio_path)
        lfcc = extract_lfcc_manual(waveform_np, sample_rate=sr, include_deltas=True)
        
        # Save
        np.save(os.path.join(output_dir, f"{entry['audio_id']}.npy"), lfcc)
        
        labels.append({
            'audio_id': entry['audio_id'],
            'label': 1 if entry['label'] == 'bonafide' else 0,
            'system_id': entry['system_id'],
            'speaker_id': entry['speaker_id'],
        })
        shapes.append(lfcc.shape)
    
    # Save labels
    with open(os.path.join(output_dir, "labels.json"), 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"\nExtraction complete!")
    print(f"  Features saved to: {output_dir}/")
    print(f"  Total files: {len(labels)}")
    print(f"  Feature shape: {shapes[0]} (coefficients × frames)")
    print(f"  Bonafide: {sum(1 for l in labels if l['label'] == 1)}")
    print(f"  Spoof:    {sum(1 for l in labels if l['label'] == 0)}")
    
    return output_dir


# =============================================================================
# Step 3: Visualize bonafide vs spoof comparison
# =============================================================================

def compare_bonafide_vs_spoof(feature_dir="sample_features"):
    """Create side-by-side comparison of bonafide and spoof spectrograms."""
    
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 60)
    print("STEP 3: Comparing bonafide vs spoof spectrograms...")
    print("=" * 60)
    
    # Load labels
    with open(os.path.join(feature_dir, "labels.json")) as f:
        labels = json.load(f)
    
    # Find one bonafide and one of each attack type
    bonafide_sample = None
    spoof_samples = {}  # {system_id: audio_id}
    
    for entry in labels:
        if entry['label'] == 1 and bonafide_sample is None:
            bonafide_sample = entry
        elif entry['label'] == 0 and entry['system_id'] not in spoof_samples:
            spoof_samples[entry['system_id']] = entry
    
    # Plot comparison
    n_plots = 1 + len(spoof_samples)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), sharex=True)
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot bonafide
    lfcc = np.load(os.path.join(feature_dir, f"{bonafide_sample['audio_id']}.npy"))
    n_ceps = lfcc.shape[0] // 3
    axes[0].imshow(lfcc[:n_ceps], aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_ylabel('LFCC coeff', fontsize=10)
    axes[0].set_title(
        f"BONAFIDE — {bonafide_sample['audio_id']} (speaker: {bonafide_sample['speaker_id']})",
        fontweight='bold', color='green', loc='left'
    )
    
    # Plot each spoof type
    for idx, (system_id, entry) in enumerate(sorted(spoof_samples.items()), 1):
        lfcc = np.load(os.path.join(feature_dir, f"{entry['audio_id']}.npy"))
        axes[idx].imshow(lfcc[:n_ceps], aspect='auto', origin='lower', cmap='viridis')
        axes[idx].set_ylabel('LFCC coeff', fontsize=10)
        
        # Label TTS vs VC
        attack_type = "TTS" if system_id in ["A01", "A02", "A03", "A04"] else "VC"
        axes[idx].set_title(
            f"SPOOF — {system_id} ({attack_type}) — {entry['audio_id']}",
            fontweight='bold', color='red', loc='left'
        )
    
    axes[-1].set_xlabel('Time frames')
    fig.suptitle(
        'Bonafide vs Spoof LFCC Spectrograms — Can you spot the differences?',
        fontsize=14, fontweight='bold', y=1.01
    )
    
    plt.tight_layout()
    save_path = os.path.join(feature_dir, "bonafide_vs_spoof_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison saved to: {save_path}")
    print("\nLook for these differences in the spoof spectrograms:")
    print("  • Smoother patterns (less natural micro-variation)")
    print("  • Bandwidth cutoff (energy drops above certain frequency)")
    print("  • Periodic artifacts (regular patterns from vocoder)")
    print("  • Unnatural transitions between frames")


# =============================================================================
# Step 4: Sanity check — verify features are correct
# =============================================================================

def sanity_check(feature_dir="sample_features"):
    """Verify extracted features look reasonable."""
    
    print("\n" + "=" * 60)
    print("STEP 4: Sanity check...")
    print("=" * 60)
    
    npy_files = sorted([f for f in os.listdir(feature_dir) if f.endswith('.npy')])
    
    all_ok = True
    for f in npy_files[:5]:
        lfcc = np.load(os.path.join(feature_dir, f))
        n_coeffs, n_frames = lfcc.shape
        
        checks = {
            'shape_height': n_coeffs == 60,
            'shape_width': 100 < n_frames < 800,
            'not_all_zeros': np.abs(lfcc).sum() > 0,
            'not_all_same': lfcc.std() > 0.01,
            'reasonable_range': -20 < lfcc.min() and lfcc.max() < 20,
        }
        
        status = "✓" if all(checks.values()) else "✗"
        print(f"  {status} {f}: shape={lfcc.shape}, "
              f"min={lfcc.min():.2f}, max={lfcc.max():.2f}, "
              f"mean={lfcc.mean():.2f}, std={lfcc.std():.2f}")
        
        if not all(checks.values()):
            all_ok = False
            for check, passed in checks.items():
                if not passed:
                    print(f"    FAILED: {check}")
    
    if all_ok:
        print("\n  All checks passed! Features are ready for model training.")
    else:
        print("\n  Some checks failed — review the extraction pipeline.")
    
    return all_ok


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download if you already have data")
    parser.add_argument("--data_dir", default="sample_data",
                        help="Where to store/find audio files")
    parser.add_argument("--feature_dir", default="sample_features",
                        help="Where to store extracted features")
    parser.add_argument("--n_bonafide", type=int, default=10,
                        help="Number of bonafide samples to download")
    parser.add_argument("--n_spoof", type=int, default=5,
                        help="Number of spoof samples per attack type")
    
    args = parser.parse_args()
    
    # Step 1: Download
    if not args.skip_download:
        download_samples(
            n_bonafide=args.n_bonafide,
            n_spoof_per_attack=args.n_spoof,
            output_dir=args.data_dir,
        )
    
    # Step 2: Extract features
    extract_all_features(
        data_dir=args.data_dir,
        output_dir=args.feature_dir,
    )
    
    # Step 3: Visualize comparison
    compare_bonafide_vs_spoof(feature_dir=args.feature_dir)
    
    # Step 4: Sanity check
    sanity_check(feature_dir=args.feature_dir)
    
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Look at {args.feature_dir}/bonafide_vs_spoof_comparison.png")
    print(f"  2. Use the features in {args.feature_dir}/ for model training")
    print(f"  3. Scale up: run lfcc_extraction.py --batch on the full dataset")
    print(f"\nFor model training, load features like this:")
    print(f"  lfcc = np.load('{args.feature_dir}/LA_T_XXXXX.npy')  # shape (60, ~399)")
    print(f"  tensor = torch.FloatTensor(lfcc).unsqueeze(0)         # shape (1, 60, ~399)")
