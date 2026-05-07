# POC: End-to-End Anti-Spoofing on a Small Subset

A single script that downloads a tiny balanced subset, extracts LFCC features, trains an LCNN, and gives you EER numbers — all in under 30 minutes on a laptop.

---

## Why a Small Subset First?

You need preliminary results for your mid-report by May 8th. Running the full pipeline on 25,000+ samples takes hours. Instead, this POC uses ~160 samples total (100 train + 60 dev) to validate that every piece of the pipeline works end-to-end. Once you confirm the pipeline is correct, you scale up by changing two numbers.

---

## Quick Start

### 1. Install dependencies

```bash
pip install datasets soundfile numpy scipy matplotlib torch torchaudio scikit-learn
```

### 2. Run the POC

```bash
python poc_antispoofing.py
```

That's it. One command. It will:

1. Download 100 training samples from HuggingFace (50 bonafide + ~54 spoof, stratified across all 6 attack types)
2. Download 60 dev samples (30 bonafide + 30 spoof, stratified)
3. Extract LFCC features from all of them
4. Train a lightweight LCNN for 20 epochs
5. Compute EER on the dev set
6. Generate 3 plots for your report
7. Print a clean results summary

Expected runtime: **5–15 minutes** on a laptop (CPU only).

---

## What It Produces

After running, you'll find these in `poc_results/`:

| File | What It Is | Use In Report |
|------|-----------|---------------|
| `training_curves.png` | Loss and EER over epochs | Shows model is learning, not just memorizing |
| `score_distribution.png` | Histogram of bonafide vs spoof scores | Shows separation between classes |
| `per_attack_eer.png` | EER broken down by attack type (A01–A06) | Shows which attacks are harder to detect |
| `lcnn_poc.pth` | Saved model weights | Can reload for further experiments |
| `results.json` | All numbers in machine-readable format | Easy to reference when writing |

---

## Scaling Up

The POC uses ~160 samples by default. To scale up toward your full preliminary results:

### Medium run (~500 samples, ~20 min)

```bash
python poc_antispoofing.py --n_train 200 --n_dev 50 --epochs 30
```

### Full balanced subset (~5,000 samples, ~1-2 hours)

```bash
python poc_antispoofing.py --n_train 2580 --n_dev 2548 --epochs 50
```

### Skip steps you've already done

```bash
# Already downloaded data, just re-train
python poc_antispoofing.py --skip_download --epochs 30

# Already downloaded and extracted, just re-train
python poc_antispoofing.py --skip_download --skip_extract --epochs 50
```

---

## Understanding the Output

### Console Output

During training you'll see output like this:

```
  Epoch  1/20 | Train loss: 0.7234 | Dev loss: 0.6891 | Dev acc: 55.0% | Dev EER: 43.21%
  Epoch  5/20 | Train loss: 0.4512 | Dev loss: 0.5123 | Dev acc: 72.3% | Dev EER: 28.45%
  Epoch 10/20 | Train loss: 0.2134 | Dev loss: 0.3456 | Dev acc: 83.7% | Dev EER: 18.90%
  Epoch 20/20 | Train loss: 0.0891 | Dev loss: 0.2789 | Dev acc: 91.2% | Dev EER: 10.34%
```

What each metric means:

- **Train loss** — should decrease steadily. If it doesn't, learning rate is too low.
- **Dev loss** — should decrease then plateau. If it increases while train loss drops, you're overfitting.
- **Dev acc** — classification accuracy. Useful but not the primary metric.
- **Dev EER** — Equal Error Rate. This is the standard ASVspoof metric. Lower is better. It's the point where False Acceptance Rate equals False Rejection Rate.

### Expected Results on Small Subset

With only ~100 training samples, expect:

- Dev EER: **15–35%** (this is normal for such a tiny dataset)
- Dev accuracy: **65–85%**

These numbers will improve dramatically when you scale to the full balanced subset (~5,000 samples), where you should see EER of **5–15%** with this same LCNN architecture.

### What to Write in Your Mid-Report

The final summary section of the script output includes a template. Key points:

1. State that these are preliminary results on a proof-of-concept subset
2. Report the EER and accuracy numbers you got
3. Include the training curves plot (shows learning progress)
4. Include the per-attack EER chart (shows which attacks are harder)
5. Mention that you plan to scale up with the full dataset and a stronger architecture (SSL-based approach with wav2vec 2.0 / WavLM)

---

## Architecture Details (For Your Report)

The POC uses a Lightweight CNN (LCNN) with the following structure:

```
Input: (batch, 1, 60, 400) — single-channel LFCC spectrogram

Conv Block 1: Conv2d(1→32) → BatchNorm → MFM → MaxPool2d(2)    → (batch, 16, 30, 200)
Conv Block 2: Conv2d(16→32) → BatchNorm → MFM → MaxPool2d(2)   → (batch, 16, 15, 100)
Conv Block 3: Conv2d(16→64) → BatchNorm → MFM → MaxPool2d(2)   → (batch, 32, 7, 50)
Conv Block 4: Conv2d(32→64) → BatchNorm → MFM                  → (batch, 32, 7, 50)
Conv Block 5: Conv2d(32→64) → BatchNorm → MFM                  → (batch, 32, 7, 50)

Global Average Pool → (batch, 32)
FC(32→64) → ReLU → Dropout(0.3)
FC(64→2) → Softmax

Output: 2 class scores (bonafide vs spoof)
```

**MFM (Max Feature Map)**: Instead of ReLU, splits feature maps into two halves and takes the element-wise maximum. This competitive gating forces the network to keep only the most discriminative features — important when training data is limited.

**Total parameters**: ~200K (very lightweight, trains fast)

---

## Folder Structure After Running

```
poc_data/
├── train/
│   ├── flac/           ← downloaded .flac audio files
│   ├── protocol.txt    ← labels for each file
│   └── features/       ← extracted .npy LFCC features + labels.json
└── dev/
    ├── flac/
    ├── protocol.txt
    └── features/

poc_results/
├── lcnn_poc.pth              ← saved model
├── training_curves.png       ← loss + EER over epochs
├── score_distribution.png    ← bonafide vs spoof histogram
├── per_attack_eer.png        ← EER per attack type
└── results.json              ← all numbers
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'datasets'"

```bash
pip install datasets
```

### "LibsndfileError" or soundfile issues

```bash
pip install soundfile
# Linux:
sudo apt-get install libsndfile1
```

### Download hangs or is slow

HuggingFace streaming can be slow on some networks. Try:

```bash
# Set a timeout
export HF_HUB_DOWNLOAD_TIMEOUT=300

# Or download the full dataset once and use --skip_download next time
python poc_antispoofing.py  # first run downloads
python poc_antispoofing.py --skip_download --epochs 30  # subsequent runs skip it
```

### CUDA out of memory

The model is tiny (~200K params), so this shouldn't happen. But if it does:

```bash
# Force CPU
CUDA_VISIBLE_DEVICES="" python poc_antispoofing.py
```

### EER is 50% (random chance)

This means the model isn't learning. Try:
- More training samples: `--n_train 200`
- More epochs: `--epochs 50`
- Check that your data has both bonafide and spoof samples (look at the download step output)
