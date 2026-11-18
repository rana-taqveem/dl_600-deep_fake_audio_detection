# Deepfake Audio Detection: Real vs. Synthetically Cloned Voices

A comparative deep learning study evaluating two anti-spoofing architectures on the **ASVspoof 2019 Logical Access (LA)** benchmark to distinguish genuine speech from synthetically cloned voices across 13 unseen spoofing attacks.

## Authors
* **Fatima Khalid** (`25280015@lums.edu.pk`)
* **Rana Taqveem ul Hassan** (`25280030@lums.edu.pk`)
* **Muhammad Sami ul Basit** (`25280025@lums.edu.pk`)

*Syed Babar Ali School of Science and Engineering, LUMS, Lahore, Pakistan*

---

## Overview
Automatic Speaker Verification (ASV) systems are increasingly vulnerable to modern text-to-speech (TTS) and voice conversion spoofing attacks. The core challenge in audio anti-spoofing is generalizing to **unseen attack types** rather than memorizing known synthesis methods. 

This repository implements and compares two distinct pipelines trained on the full ASVspoof 2019 LA corpus (seen attacks `A01–A06`) and evaluated on 13 unseen evaluation attacks (`A07–A19`):
1. **LFCC + LCNN:** A classical cepstral feature extraction pipeline paired with a Lightweight Convolutional Neural Network using Max Feature Map (MFM) activations.
2. **Modified RawNet2:** An end-to-end raw waveform architecture incorporating frozen parallel sinc filter banks (Mel, linear, and inverse-Mel), multi-scale temporal convolutions, residual blocks with Feature Map Scaling (FMS), and a two-layer GRU.

---

## Repository Structure
```text
├── complete_pipeline/   # End-to-end training, evaluation, and metric computation scripts
├── models/  # LFCC+LCNN and modified RawNet2 model architectures
└── poc/   # Proof-of-concept experiments and working prototype notebooks
```

---

## Model Architectures

### 1. LFCC + LCNN
* **Input:** 4-second normalized audio segments (64,000 samples at 16 kHz).
* **Features:** 60-dimensional Linear Frequency Cepstral Coefficients (20 static + 20 delta + 20 delta-delta) extracted using a 20 ms Hamming window, 10 ms hop, 512-point FFT, and 20 linearly spaced triangular filters (0–8 kHz).
* **Backbone:** 5-block Lightweight CNN with competitive Max Feature Map (MFM) suppression, Batch Normalization, MaxPool2d, and Global Average Pooling (~54K parameters).
* **Training:** Adam optimizer (`lr=1e-4`), batch size 32, 30 epochs, and class-weighted `CrossEntropyLoss` to address the 1:8.7 class imbalance.

### 2. Modified RawNet2
* **Input:** 2-second raw 16 kHz PCM waveforms (32,000 samples).
* **Frontend:** Three frozen parallel sinc filter banks (Mel-scale, linear, and inverse-Mel, 80 filters each) to capture low-frequency formants and high-frequency vocoder artifacts without overfitting.
* **Backbone:** Multi-scale 1D convolutions (kernel sizes 5, 9, and 15), three residual blocks (`192→64→64→64`) with Feature Map Scaling (FMS) channel attention, and a 2-layer GRU with 128 hidden units (~560K parameters).
* **Training:** Adam optimizer (`lr=1e-4`, weight decay `1e-4`), `ReduceLROnPlateau`, FP16 mixed precision, Gaussian noise (30 dB), and ±10% time-shift augmentation.

---

## Experimental Results (ASVspoof 2019 LA)

Contrary to expectations, the lightweight **LFCC + LCNN** pipeline outperformed the end-to-end **RawNet2** model across every metric while using nearly 10x fewer parameters:

| Metric | LFCC + LCNN | Modified RawNet2 |
| :--- | :---: | :---: |
| **Dev EER (Seen A01–A06)** | **0.59%** | 7.24% |
| **Test EER (Unseen A07–A19)** | **4.46%** | 14.27% |
| **Test Accuracy** | **91.50%** | 69.44% |
| **min t-DCF** | **0.10882** | — |
| **Parameters** | **~54K** | ~560K |
| **Best Detected Attack** | A09 (0.00% EER) | A07 (5.13% EER) |
| **Most Challenging Attack** | A17 (19.64% EER) | A17 (31.62% EER) |

### Key Takeaways
* **Generalization:** Handcrafted cepstral features combined with class-weighted loss and longer 4-second temporal context transferred significantly better to unseen synthesis algorithms.
* **Hardest Attack (`A17`):** Both architectures struggled most on attack `A17`, highlighting the need for self-supervised speech representations (such as `wav2vec 2.0` or `WavLM`) in future work.
