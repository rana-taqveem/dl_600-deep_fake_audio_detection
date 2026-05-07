"""
==============================================================================
POC: End-to-End Anti-Spoofing — Small Subset
==============================================================================
Downloads a small balanced subset from HuggingFace, extracts LFCC features,
trains a lightweight LCNN, and reports EER — all in one script.

Target: Preliminary results for mid-report in under 30 minutes.

Usage:
    pip install datasets soundfile numpy scipy matplotlib torch torchaudio scikit-learn
    python poc_antispoofing.py

Subset sizes (configurable):
    Train: 50 bonafide + 50 spoof (stratified across 6 attacks) = 100
    Dev:   30 bonafide + 30 spoof = 60

Total: ~160 audio files. Downloads in ~2 minutes, trains in ~5 minutes on CPU.
==============================================================================
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# CONFIG — Change these to scale up
# =============================================================================

DEFAULT_CONFIG = {
    # Data
    "n_bonafide_train": 50,        # bonafide samples for training
    "n_spoof_per_attack_train": 9, # spoof per attack type (6 types × 9 = 54)
    "n_bonafide_dev": 30,          # bonafide samples for validation
    "n_spoof_per_attack_dev": 5,   # spoof per attack type for dev
    "max_audio_sec": 4.0,          # pad/truncate audio to this length

    # LFCC
    "sample_rate": 16000,
    "n_lfcc": 20,
    "n_filters": 20,
    "n_fft": 512,
    "frame_ms": 20.0,
    "hop_ms": 10.0,
    "max_frames": 400,             # pad/truncate feature frames

    # Training
    "batch_size": 16,
    "epochs": 20,
    "lr": 0.0001,
    "weight_decay": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Paths
    "data_dir": "poc_data",
    "output_dir": "poc_results",
}


# =============================================================================
# STEP 1: Download small subset from HuggingFace
# =============================================================================

def download_subset(config, split="train"):
    """Download a small balanced subset of ASVspoof 2019 LA."""

    from datasets import load_dataset
    import soundfile as sf

    if split == "train":
        n_bonafide = config["n_bonafide_train"]
        n_spoof = config["n_spoof_per_attack_train"]
        hf_split = "train"
    else:
        n_bonafide = config["n_bonafide_dev"]
        n_spoof = config["n_spoof_per_attack_dev"]
        hf_split = "validation"

    flac_dir = os.path.join(config["data_dir"], split, "flac")
    os.makedirs(flac_dir, exist_ok=True)

    # Check if already downloaded
    existing = [f for f in os.listdir(flac_dir) if f.endswith(".flac")]
    if len(existing) > 10:
        print(f"  [{split}] Found {len(existing)} existing files, skipping download")
        return

    print(f"  [{split}] Downloading from HuggingFace (target: {n_bonafide} bonafide + {n_spoof}×6 spoof)...")

    dataset = load_dataset("LanceaKing/asvspoof2019", "LA", split=hf_split, streaming=True)

    bonafide_count = 0
    spoof_counts = defaultdict(int)
    protocol_lines = []
    saved = 0

    for sample in dataset:
        audio_id = sample["audio_file_name"]
        speaker_id = sample["speaker_id"]
        system_id = sample["system_id"]
        is_bonafide = sample["key"] == 0
        label = "bonafide" if is_bonafide else "spoof"

        keep = False
        if is_bonafide and bonafide_count < n_bonafide:
            keep = True
            bonafide_count += 1
        elif not is_bonafide and spoof_counts[system_id] < n_spoof:
            keep = True
            spoof_counts[system_id] += 1

        if not keep:
            # Check completion
            done = bonafide_count >= n_bonafide and all(
                spoof_counts.get(f"A0{i}", 0) >= n_spoof for i in range(1, 7)
            )
            if done:
                break
            continue

        # Save audio
        audio = sample["audio"]
        sf.write(
            os.path.join(flac_dir, f"{audio_id}.flac"),
            audio["array"],
            audio["sampling_rate"],
        )

        sys_for_protocol = "-" if is_bonafide else system_id
        protocol_lines.append(f"{speaker_id} {audio_id} - {sys_for_protocol} {label}")
        saved += 1

    # Save protocol
    protocol_path = os.path.join(config["data_dir"], split, "protocol.txt")
    with open(protocol_path, "w") as f:
        f.write("\n".join(protocol_lines))

    print(f"  [{split}] Downloaded {saved} files ({bonafide_count} bonafide, "
          f"{sum(spoof_counts.values())} spoof)")
    for sys_id in sorted(spoof_counts):
        print(f"    {sys_id}: {spoof_counts[sys_id]}")


# =============================================================================
# STEP 2: LFCC extraction (compact version)
# =============================================================================

def extract_lfcc(waveform, config):
    """Extract LFCC features from a 1D waveform array."""

    sr = config["sample_rate"]
    n_filters = config["n_filters"]
    n_ceps = config["n_lfcc"]
    n_fft = config["n_fft"]
    frame_len = int(sr * config["frame_ms"] / 1000)
    frame_hop = int(sr * config["hop_ms"] / 1000)

    # Pre-emphasis
    emphasized = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

    # Framing
    n_frames = 1 + (len(emphasized) - frame_len) // frame_hop
    if n_frames < 1:
        n_frames = 1
        emphasized = np.pad(emphasized, (0, frame_len - len(emphasized)))

    indices = np.arange(frame_len)[None, :] + np.arange(n_frames)[:, None] * frame_hop
    frames = emphasized[indices] * np.hamming(frame_len)

    # FFT → power spectrum
    power = (1.0 / n_fft) * np.abs(np.fft.rfft(frames, n=n_fft)) ** 2

    # Linear filterbank
    freq_points = np.linspace(0, sr / 2, n_filters + 2)
    bin_points = np.floor((n_fft + 1) * freq_points / sr).astype(int)
    fbank = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        for j in range(bin_points[i], bin_points[i + 1]):
            fbank[i, j] = (j - bin_points[i]) / max(1, bin_points[i + 1] - bin_points[i])
        for j in range(bin_points[i + 1], bin_points[i + 2]):
            fbank[i, j] = (bin_points[i + 2] - j) / max(1, bin_points[i + 2] - bin_points[i + 1])

    # Log filterbank energies → DCT
    energies = np.dot(power, fbank.T)
    energies = np.where(energies == 0, np.finfo(float).eps, energies)
    log_energies = np.log(energies)

    from scipy.fft import dct
    cepstral = dct(log_energies, type=2, axis=1, norm="ortho")[:, :n_ceps].T

    # Deltas
    def _delta(feat, w=2):
        denom = 2 * sum(n**2 for n in range(1, w + 1))
        padded = np.pad(feat, ((0, 0), (w, w)), mode="edge")
        d = np.zeros_like(feat)
        for t in range(feat.shape[1]):
            for n in range(1, w + 1):
                d[:, t] += n * (padded[:, t + w + n] - padded[:, t + w - n])
            d[:, t] /= denom
        return d

    delta = _delta(cepstral)
    delta_delta = _delta(delta)

    return np.concatenate([cepstral, delta, delta_delta], axis=0)  # (60, n_frames)


def process_split(config, split):
    """Extract LFCC for all files in a split."""

    import torchaudio

    flac_dir = os.path.join(config["data_dir"], split, "flac")
    protocol_path = os.path.join(config["data_dir"], split, "protocol.txt")
    feat_dir = os.path.join(config["data_dir"], split, "features")
    os.makedirs(feat_dir, exist_ok=True)

    # Check if already extracted
    existing_npy = [f for f in os.listdir(feat_dir) if f.endswith(".npy")]
    if len(existing_npy) > 10:
        print(f"  [{split}] Found {len(existing_npy)} existing feature files, skipping extraction")
        return

    entries = []
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                entries.append({
                    "speaker_id": parts[0],
                    "audio_id": parts[1],
                    "system_id": parts[3],
                    "label": parts[4],
                })

    labels = []
    target_len = int(config["sample_rate"] * config["max_audio_sec"])

    for entry in entries:
        path = os.path.join(flac_dir, f"{entry['audio_id']}.flac")
        if not os.path.exists(path):
            continue

        waveform, sr = torchaudio.load(path)
        wav = waveform.squeeze(0).numpy()

        # Pad/truncate
        if len(wav) > target_len:
            wav = wav[:target_len]
        elif len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))

        lfcc = extract_lfcc(wav, config)

        np.save(os.path.join(feat_dir, f"{entry['audio_id']}.npy"), lfcc)
        labels.append({
            "audio_id": entry["audio_id"],
            "label": 1 if entry["label"] == "bonafide" else 0,
            "system_id": entry["system_id"],
        })

    with open(os.path.join(feat_dir, "labels.json"), "w") as f:
        json.dump(labels, f)

    print(f"  [{split}] Extracted {len(labels)} features → {feat_dir}/")


# =============================================================================
# STEP 3: PyTorch Dataset
# =============================================================================

class LFCCDataset(Dataset):
    def __init__(self, feat_dir, max_frames=400):
        with open(os.path.join(feat_dir, "labels.json")) as f:
            self.labels = json.load(f)
        self.feat_dir = feat_dir
        self.max_frames = max_frames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        entry = self.labels[idx]
        lfcc = np.load(os.path.join(self.feat_dir, f"{entry['audio_id']}.npy"))

        # Pad/truncate width
        _, n_frames = lfcc.shape
        if n_frames > self.max_frames:
            lfcc = lfcc[:, :self.max_frames]
        elif n_frames < self.max_frames:
            lfcc = np.pad(lfcc, ((0, 0), (0, self.max_frames - n_frames)))

        tensor = torch.FloatTensor(lfcc).unsqueeze(0)  # (1, 60, max_frames)
        label = entry["label"]
        system_id = entry.get("system_id", "-")

        return tensor, label, system_id


# =============================================================================
# STEP 4: Lightweight LCNN model
# =============================================================================

class MaxFeatureMap(nn.Module):
    """Max Feature Map activation — splits channels in half, takes element-wise max."""
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return torch.max(a, b)


class LightCNN(nn.Module):
    """
    Lightweight CNN for anti-spoofing.
    Architecture: 5 conv blocks with MFM activation → global avg pool → FC classifier.
    ~200K parameters — trains fast even on CPU.
    """
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (1, 60, 400) → (16, 60, 400)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            MaxFeatureMap(),  # → (16, 60, 400)
            nn.MaxPool2d(2),  # → (16, 30, 200)

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            MaxFeatureMap(),  # → (16, 30, 200)
            nn.MaxPool2d(2),  # → (16, 15, 100)

            # Block 3
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            MaxFeatureMap(),  # → (32, 15, 100)
            nn.MaxPool2d(2),  # → (32, 7, 50)

            # Block 4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            MaxFeatureMap(),  # → (32, 7, 50)

            # Block 5
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            MaxFeatureMap(),  # → (32, 7, 50)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)  # → (32, 1, 1)

        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x


# =============================================================================
# STEP 5: Training loop
# =============================================================================

def train_model(config):
    """Train the LCNN and return training history."""

    device = config["device"]
    print(f"  Device: {device}")

    # Datasets
    train_ds = LFCCDataset(
        os.path.join(config["data_dir"], "train", "features"),
        max_frames=config["max_frames"],
    )
    dev_ds = LFCCDataset(
        os.path.join(config["data_dir"], "dev", "features"),
        max_frames=config["max_frames"],
    )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=config["batch_size"], shuffle=False)

    print(f"  Train: {len(train_ds)} samples ({sum(1 for l in train_ds.labels if l['label']==1)} bonafide, "
          f"{sum(1 for l in train_ds.labels if l['label']==0)} spoof)")
    print(f"  Dev:   {len(dev_ds)} samples ({sum(1 for l in dev_ds.labels if l['label']==1)} bonafide, "
          f"{sum(1 for l in dev_ds.labels if l['label']==0)} spoof)")

    # Model
    model = LightCNN(in_channels=1, num_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # Loss with class weights (handle slight imbalance)
    n_bonafide = sum(1 for l in train_ds.labels if l["label"] == 1)
    n_spoof = sum(1 for l in train_ds.labels if l["label"] == 0)
    weight = torch.FloatTensor([n_bonafide / (n_bonafide + n_spoof),
                                 n_spoof / (n_bonafide + n_spoof)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    history = {"train_loss": [], "dev_loss": [], "dev_acc": [], "dev_eer": []}

    for epoch in range(config["epochs"]):
        # --- Train ---
        model.train()
        train_loss = 0
        for features, labels, _ in train_loader:
            features = features.to(device)
            labels = torch.LongTensor(labels).to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Evaluate ---
        model.eval()
        dev_loss = 0
        all_scores = []
        all_labels = []
        all_systems = []

        with torch.no_grad():
            for features, labels, systems in dev_loader:
                features = features.to(device)
                labels_tensor = torch.LongTensor(labels).to(device)

                outputs = model(features)
                loss = criterion(outputs, labels_tensor)
                dev_loss += loss.item()

                # Softmax score for bonafide class (index 1)
                scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_scores.extend(scores)
                all_labels.extend(labels)
                all_systems.extend(systems)

        dev_loss /= len(dev_loader)

        # Compute EER
        eer = compute_eer(np.array(all_labels), np.array(all_scores))

        # Accuracy
        preds = (np.array(all_scores) > 0.5).astype(int)
        acc = (preds == np.array(all_labels)).mean() * 100

        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_acc"].append(acc)
        history["dev_eer"].append(eer)

        print(f"  Epoch {epoch+1:2d}/{config['epochs']} | "
              f"Train loss: {train_loss:.4f} | "
              f"Dev loss: {dev_loss:.4f} | "
              f"Dev acc: {acc:.1f}% | "
              f"Dev EER: {eer:.2f}%")

    # Save model
    os.makedirs(config["output_dir"], exist_ok=True)
    model_path = os.path.join(config["output_dir"], "lcnn_poc.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    return model, history, all_scores, all_labels, all_systems


# =============================================================================
# STEP 6: Evaluation metrics
# =============================================================================

def compute_eer(labels, scores):
    """
    Compute Equal Error Rate (EER).
    EER is the point where False Acceptance Rate = False Rejection Rate.
    This is the standard metric for ASVspoof.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Find where FPR and FNR cross
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2 * 100  # as percentage

    return eer


def per_attack_analysis(scores, labels, systems):
    """Break down performance by attack type."""

    results = {}

    # Bonafide scores
    bonafide_scores = [s for s, l in zip(scores, labels) if l == 1]

    # Per-attack scores
    attack_scores = defaultdict(list)
    for s, l, sys_id in zip(scores, labels, systems):
        if l == 0:
            attack_scores[sys_id].append(s)

    for attack_id in sorted(attack_scores.keys()):
        attack_s = attack_scores[attack_id]
        combined_labels = [1] * len(bonafide_scores) + [0] * len(attack_s)
        combined_scores = bonafide_scores + attack_s

        if len(set(combined_labels)) < 2:
            results[attack_id] = {"eer": float("nan"), "n_samples": len(attack_s)}
            continue

        eer = compute_eer(np.array(combined_labels), np.array(combined_scores))
        results[attack_id] = {
            "eer": eer,
            "n_samples": len(attack_s),
            "mean_score": np.mean(attack_s),
        }

    return results


# =============================================================================
# STEP 7: Generate report plots
# =============================================================================

def generate_plots(config, history, scores, labels, systems):
    """Generate visualizations for the mid-report."""

    import matplotlib.pyplot as plt

    os.makedirs(config["output_dir"], exist_ok=True)

    # --- Plot 1: Training curves ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-", label="Train loss", linewidth=1.5)
    ax1.plot(epochs, history["dev_loss"], "r-", label="Dev loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & validation loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["dev_eer"], "g-", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("EER (%)")
    ax2.set_title("Development set EER")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 2: Score distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))

    bonafide_scores = [s for s, l in zip(scores, labels) if l == 1]
    spoof_scores = [s for s, l in zip(scores, labels) if l == 0]

    ax.hist(bonafide_scores, bins=20, alpha=0.6, label="Bonafide", color="green", density=True)
    ax.hist(spoof_scores, bins=20, alpha=0.6, label="Spoof", color="red", density=True)
    ax.set_xlabel("Bonafide score (higher = more likely genuine)")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution — bonafide vs spoof")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "score_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 3: Per-attack EER (if enough data) ---
    attack_results = per_attack_analysis(scores, labels, systems)
    if attack_results:
        fig, ax = plt.subplots(figsize=(8, 4))

        attacks = sorted(attack_results.keys())
        eers = [attack_results[a]["eer"] for a in attacks]
        colors = ["#1D9E75" if a.startswith("-") else
                  "#534AB7" if a in ["A01", "A02", "A03", "A04"] else
                  "#D85A30" for a in attacks]
        labels_text = [f"{a}\n({'TTS' if a in ['A01','A02','A03','A04'] else 'VC'})"
                       if a != "-" else "bonafide" for a in attacks]

        bars = ax.bar(labels_text, eers, color=colors, alpha=0.8)
        ax.set_ylabel("EER (%)")
        ax.set_title("Per-attack EER on development set")
        ax.grid(True, alpha=0.3, axis="y")

        for bar, eer_val in zip(bars, eers):
            if not np.isnan(eer_val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{eer_val:.1f}%", ha="center", va="bottom", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(config["output_dir"], "per_attack_eer.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Plots saved to {config['output_dir']}/")


# =============================================================================
# STEP 8: Print final summary
# =============================================================================

def print_summary(config, history, scores, labels, systems):
    """Print a clean summary suitable for the mid-report."""

    best_epoch = np.argmin(history["dev_eer"])
    best_eer = history["dev_eer"][best_epoch]
    best_acc = history["dev_acc"][best_epoch]

    attack_results = per_attack_analysis(scores, labels, systems)

    print("\n" + "=" * 60)
    print("  PRELIMINARY RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  Model:          Lightweight CNN (LCNN) with MFM activation")
    print(f"  Features:       LFCC (60-dim: 20 static + 20 delta + 20 delta-delta)")
    print(f"  Training set:   {config['n_bonafide_train']} bonafide + "
          f"{config['n_spoof_per_attack_train']*6} spoof = "
          f"{config['n_bonafide_train'] + config['n_spoof_per_attack_train']*6} samples")
    print(f"  Dev set:        {config['n_bonafide_dev']} bonafide + "
          f"{config['n_spoof_per_attack_dev']*6} spoof = "
          f"{config['n_bonafide_dev'] + config['n_spoof_per_attack_dev']*6} samples")
    print(f"  Epochs:         {config['epochs']}")
    print(f"  Device:         {config['device']}")
    print(f"\n  Best Dev EER:   {best_eer:.2f}% (epoch {best_epoch + 1})")
    print(f"  Best Dev Acc:   {best_acc:.1f}%")

    if attack_results:
        print(f"\n  Per-attack EER:")
        for attack_id in sorted(attack_results.keys()):
            r = attack_results[attack_id]
            attack_type = "TTS" if attack_id in ["A01", "A02", "A03", "A04"] else "VC"
            print(f"    {attack_id} ({attack_type}): EER = {r['eer']:.2f}%  (n={r['n_samples']})")

    print(f"\n  Output files:")
    print(f"    {config['output_dir']}/lcnn_poc.pth           — trained model")
    print(f"    {config['output_dir']}/training_curves.png    — loss & EER plots")
    print(f"    {config['output_dir']}/score_distribution.png — bonafide vs spoof scores")
    print(f"    {config['output_dir']}/per_attack_eer.png     — per-attack breakdown")
    print(f"    {config['output_dir']}/results.json           — all results as JSON")

    # Save JSON results
    results = {
        "config": {k: str(v) for k, v in config.items()},
        "best_eer": float(best_eer),
        "best_acc": float(best_acc),
        "best_epoch": int(best_epoch + 1),
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
        "per_attack": {k: {kk: float(vv) if not isinstance(vv, int) else vv
                           for kk, vv in v.items()}
                       for k, v in attack_results.items()},
    }
    with open(os.path.join(config["output_dir"], "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("  NOTE FOR MID-REPORT")
    print("=" * 60)
    print(f"""
  These are preliminary results on a very small subset
  ({config['n_bonafide_train'] + config['n_spoof_per_attack_train']*6} training samples).
  In your report, mention:

  1. This is a proof-of-concept to validate the pipeline
  2. The full training set has 25,380 samples (vs {config['n_bonafide_train'] + config['n_spoof_per_attack_train']*6} used here)
  3. You plan to scale up with:
     - Full balanced dataset (~5,000 samples)
     - SSL-based approach (wav2vec 2.0 / WavLM)
     - Data augmentation
  4. Current EER of {best_eer:.2f}% is expected to improve
     significantly with more data and a stronger model
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="POC Anti-Spoofing — Small Subset")
    parser.add_argument("--n_train", type=int, default=None,
                        help="Override bonafide training samples (default: 50)")
    parser.add_argument("--n_dev", type=int, default=None,
                        help="Override bonafide dev samples (default: 30)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs (default: 20)")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download if data already exists")
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip feature extraction if already done")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.n_train:
        config["n_bonafide_train"] = args.n_train
        config["n_spoof_per_attack_train"] = max(1, args.n_train // 6)
    if args.n_dev:
        config["n_bonafide_dev"] = args.n_dev
        config["n_spoof_per_attack_dev"] = max(1, args.n_dev // 6)
    if args.epochs:
        config["epochs"] = args.epochs

    total_start = time.time()

    # --- Step 1: Download ---
    print("\n" + "=" * 60)
    print("STEP 1: Downloading data")
    print("=" * 60)
    if not args.skip_download:
        download_subset(config, split="train")
        download_subset(config, split="dev")
    else:
        print("  Skipped (--skip_download)")

    # --- Step 2: Extract LFCC ---
    print("\n" + "=" * 60)
    print("STEP 2: Extracting LFCC features")
    print("=" * 60)
    if not args.skip_extract:
        process_split(config, "train")
        process_split(config, "dev")
    else:
        print("  Skipped (--skip_extract)")

    # --- Step 3: Train ---
    print("\n" + "=" * 60)
    print("STEP 3: Training LCNN")
    print("=" * 60)
    model, history, scores, labels, systems = train_model(config)

    # --- Step 4: Visualize ---
    print("\n" + "=" * 60)
    print("STEP 4: Generating report plots")
    print("=" * 60)
    generate_plots(config, history, scores, labels, systems)

    # --- Step 5: Summary ---
    print_summary(config, history, scores, labels, systems)

    elapsed = time.time() - total_start
    print(f"\n  Total time: {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")


if __name__ == "__main__":
    main()
