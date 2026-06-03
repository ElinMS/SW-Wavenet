"""
SW-WaveNet Evaluation Pipeline
===============================
Evaluates the trained SW-WaveNet model on test data containing both
normal and anomalous sounds.

Metrics (paper specification):
    - AUC:  Area Under the ROC Curve
    - pAUC: Partial AUC over low FPR range [0, 0.1]

Also generates:
    - t-SNE visualization of representation vectors
    - Per-machine-ID and overall results

Usage:
    python3 evaluate.py
"""

import os
import re
import glob
import time
import torch
import torch.nn.functional as F
import librosa
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavegram_net import WavegramNet
from wavenet_model import SWWaveNetClassifier


# ======================================================================
# Configuration
# ======================================================================

TEST_DIR       = "/home/teaching/Elin/fan_preprocessed/test"
CHECKPOINT_DIR = "/home/teaching/Elin/SW-Wavenet/checkpoints"
OUTPUT_DIR     = "/home/teaching/Elin/SW-Wavenet/evaluation"
SAMPLE_RATE    = 16000
N_MELS         = 128
N_FFT          = 1024
HOP_LENGTH     = 512
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# Helper Functions
# ======================================================================

def parse_test_file(filepath):
    """
    Parse test filename to extract label and machine ID.

    DCASE 2020 naming convention:
        normal_id_XX_YYYYYYYY.pt   -> is_anomaly = False
        anomaly_id_XX_YYYYYYYY.pt  -> is_anomaly = True

    Returns:
        is_anomaly: bool
        machine_id: str (e.g., '00', '02', '04', '06')
    """
    basename = os.path.basename(filepath)
    is_anomaly = basename.startswith("anomaly")

    match = re.search(r'_id_(\d+)_', basename)
    if match:
        machine_id = match.group(1)
    else:
        raise ValueError(f"Cannot parse machine ID from: {basename}")

    return is_anomaly, machine_id


def extract_features(pt_path):
    """
    Extract log-mel spectrogram and raw waveform from preprocessed file.

    Returns:
        spectrogram: [1, 1, Time, 128]  -- log-mel spectrogram
        waveform:    [1, 1, Samples]    -- raw audio
    """
    data = torch.load(pt_path, map_location='cpu', weights_only=True)
    
    # Preprocessed tensor:
    # spectrogram: [Time, 128] -> [1, 1, Time, 128]
    # waveform: [1, Samples] -> [1, 1, Samples]
    
    spectrogram = data['spectrogram'].unsqueeze(0).unsqueeze(0)
    waveform = data['waveform'].unsqueeze(0)

    return spectrogram, waveform


def compute_pauc(y_true, y_score, max_fpr=0.1):
    """
    Compute partial AUC over FPR range [0, max_fpr].

    Paper definition: "pAUC is defined as the AUC over a low
    false-positive-rate (FPR) range [0, p] and p = 0.1"

    The result is normalized to [0, 1] by dividing by max_fpr.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Clip to FPR <= max_fpr region
    stop_idx = np.searchsorted(fpr, max_fpr, side='right')
    fpr_partial = np.concatenate([fpr[:stop_idx], [max_fpr]])

    # Interpolate TPR at max_fpr
    tpr_at_max = np.interp(max_fpr, fpr, tpr)
    tpr_partial = np.concatenate([tpr[:stop_idx], [tpr_at_max]])

    # Compute area and normalize to [0, 1]
    pauc = np.trapz(tpr_partial, fpr_partial) / max_fpr

    return pauc


# ======================================================================
# Main Evaluation
# ======================================================================

def main():
    print("=" * 70)
    print("  SW-WaveNet -- Evaluation Pipeline")
    print("=" * 70)
    print(f"  Device: {DEVICE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -- 1. Discover test files ------------------------------------------
    print("\n[1/5] Loading test files...")
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.pt")))
    if not test_files:
        print(f"  [ERROR] No .pt files found in {TEST_DIR}")
        return

    # Count normal vs anomaly
    n_normal_total = sum(1 for f in test_files if os.path.basename(f).startswith("normal"))
    n_anomaly_total = len(test_files) - n_normal_total
    print(f"  Test directory : {TEST_DIR}")
    print(f"  Total files    : {len(test_files)}")
    print(f"  Normal         : {n_normal_total}")
    print(f"  Anomaly        : {n_anomaly_total}")

    # -- 2. Load trained model -------------------------------------------
    print("\n[2/5] Loading trained model...")
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
        print(f"  Run train.py first!")
        return

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    id_to_label = checkpoint['id_to_label']
    label_to_id = checkpoint['label_to_id']
    num_classes = len(id_to_label)

    wavegram_net = WavegramNet()
    model = SWWaveNetClassifier(
        wavegram_net=wavegram_net,
        num_classes=num_classes,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"  Epoch          : {checkpoint.get('epoch', '?')}")
    print(f"  Val loss       : {checkpoint.get('val_loss', '?')}")
    print(f"  Val acc        : {checkpoint.get('val_acc', '?')}")
    print(f"  Classes        : {num_classes} ({id_to_label})")
    print(f"  Parameters     : {total_params:,}")

    # -- 3. Compute anomaly scores ---------------------------------------
    print("\n[3/5] Computing anomaly scores...")
    print("-" * 70)

    results = {}  # machine_id -> {'labels': [], 'scores': [], 'reprs': []}
    t0 = time.time()
    errors = []

    for idx, audio_path in enumerate(test_files):
        try:
            is_anomaly, machine_id = parse_test_file(audio_path)

            if machine_id not in results:
                results[machine_id] = {'labels': [], 'scores': [], 'reprs': []}

            # Extract features
            spectrogram, waveform = extract_features(audio_path)
            spectrogram = spectrogram.to(DEVICE)
            waveform = waveform.to(DEVICE)

            # Compute anomaly score and representation
            with torch.no_grad():
                score = model.anomaly_score(spectrogram, waveform)
                repr_vec = model.get_representation(spectrogram, waveform)

            results[machine_id]['labels'].append(1 if is_anomaly else 0)
            results[machine_id]['scores'].append(score.item())
            results[machine_id]['reprs'].append(repr_vec.cpu().numpy().flatten())

            # Progress
            if (idx + 1) % 50 == 0 or idx == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (len(test_files) - idx - 1) / rate if rate > 0 else 0
                label_str = "ANOMALY" if is_anomaly else "normal"
                print(f"  [{idx+1:>4}/{len(test_files)}]  "
                      f"ID={machine_id}  {label_str:<7}  "
                      f"score={score.item():.4f}  ETA={eta:.0f}s")

        except Exception as e:
            errors.append((os.path.basename(audio_path), str(e)))
            print(f"  [{idx+1:>4}/{len(test_files)}]  ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\n  Processed {len(test_files) - len(errors)}/{len(test_files)} files "
          f"in {elapsed:.1f}s ({elapsed/len(test_files):.2f}s/file)")
    if errors:
        print(f"  Errors: {len(errors)}")

    # -- 4. Calculate AUC and pAUC ---------------------------------------
    print("\n[4/5] Calculating metrics...")
    print("=" * 70)
    print(f"  {'Machine ID':<14} {'AUC':>10} {'pAUC':>10} {'Normal':>8} {'Anomaly':>8}")
    print("-" * 70)

    all_labels = []
    all_scores = []
    auc_list = []
    pauc_list = []

    for mid in sorted(results.keys()):
        labels = np.array(results[mid]['labels'])
        scores = np.array(results[mid]['scores'])

        n_normal = int(np.sum(labels == 0))
        n_anomaly = int(np.sum(labels == 1))

        all_labels.extend(labels)
        all_scores.extend(scores)

        if n_anomaly > 0 and n_normal > 0:
            auc = roc_auc_score(labels, scores)
            pauc = compute_pauc(labels, scores, max_fpr=0.1)
        else:
            auc = float('nan')
            pauc = float('nan')
            print(f"  [WARNING] ID {mid}: cannot compute metrics "
                  f"(normal={n_normal}, anomaly={n_anomaly})")

        auc_list.append(auc)
        pauc_list.append(pauc)

        print(f"  Fan ID {mid:<6} {auc:>10.4f} {pauc:>10.4f} {n_normal:>8d} {n_anomaly:>8d}")

    # Overall
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    avg_auc = np.nanmean(auc_list)
    avg_pauc = np.nanmean(pauc_list)

    if np.sum(all_labels == 1) > 0 and np.sum(all_labels == 0) > 0:
        overall_auc = roc_auc_score(all_labels, all_scores)
        overall_pauc = compute_pauc(all_labels, all_scores, max_fpr=0.1)
    else:
        overall_auc = float('nan')
        overall_pauc = float('nan')

    print("-" * 70)
    print(f"  {'Average':<14} {avg_auc:>10.4f} {avg_pauc:>10.4f}")
    print(f"  {'Overall':<14} {overall_auc:>10.4f} {overall_pauc:>10.4f}")
    print("=" * 70)

    # -- Save results to text file --
    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write("SW-WaveNet Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Test dir  : {TEST_DIR}\n")
        f.write(f"Device    : {DEVICE}\n\n")
        f.write(f"{'Machine ID':<14} {'AUC':>10} {'pAUC':>10}\n")
        f.write("-" * 40 + "\n")
        for mid, auc, pauc in zip(sorted(results.keys()), auc_list, pauc_list):
            f.write(f"Fan ID {mid:<6} {auc:>10.4f} {pauc:>10.4f}\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Average':<14} {avg_auc:>10.4f} {avg_pauc:>10.4f}\n")
        f.write(f"{'Overall':<14} {overall_auc:>10.4f} {overall_pauc:>10.4f}\n")
    print(f"\n  Results saved to: {results_path}")

    # -- 5. t-SNE Visualization ------------------------------------------
    print("\n[5/5] Generating t-SNE visualization...")

    try:
        from sklearn.manifold import TSNE

        # Collect all representations
        all_reprs = []
        all_tsne_labels = []
        all_tsne_mids = []

        for mid in sorted(results.keys()):
            reprs = np.array(results[mid]['reprs'])
            labels = np.array(results[mid]['labels'])
            all_reprs.append(reprs)
            all_tsne_labels.extend(labels)
            all_tsne_mids.extend([mid] * len(labels))

        all_reprs = np.vstack(all_reprs)
        all_tsne_labels = np.array(all_tsne_labels)
        all_tsne_mids = np.array(all_tsne_mids)

        print(f"  Representations: {all_reprs.shape}")
        print(f"  Running t-SNE...")

        perplexity = min(30, len(all_reprs) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings = tsne.fit_transform(all_reprs)

        # -- Plot 1: Per machine ID --
        unique_mids = sorted(set(all_tsne_mids))
        n_mids = len(unique_mids)
        fig, axes = plt.subplots(1, n_mids, figsize=(5 * n_mids, 4.5))
        if n_mids == 1:
            axes = [axes]

        for ax, mid in zip(axes, unique_mids):
            mask = all_tsne_mids == mid
            emb = embeddings[mask]
            lab = all_tsne_labels[mask]

            normal_mask = lab == 0
            anomaly_mask = lab == 1

            ax.scatter(emb[normal_mask, 0], emb[normal_mask, 1],
                       c='#2196F3', alpha=0.6, s=15, label='Normal', edgecolors='none')
            ax.scatter(emb[anomaly_mask, 0], emb[anomaly_mask, 1],
                       c='#F44336', alpha=0.6, s=15, label='Anomaly', edgecolors='none')
            ax.set_title(f'Fan ID {mid}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle('t-SNE: SW-WaveNet Representation Vectors (per Machine ID)',
                      fontsize=13, fontweight='bold')
        plt.tight_layout()
        tsne_per_id_path = os.path.join(OUTPUT_DIR, "tsne_per_machine_id.png")
        plt.savefig(tsne_per_id_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {tsne_per_id_path}")

        # -- Plot 2: Combined view --
        fig, ax = plt.subplots(figsize=(8, 6))
        normal_mask = all_tsne_labels == 0
        anomaly_mask = all_tsne_labels == 1

        ax.scatter(embeddings[normal_mask, 0], embeddings[normal_mask, 1],
                   c='#2196F3', alpha=0.5, s=15, label='Normal', edgecolors='none')
        ax.scatter(embeddings[anomaly_mask, 0], embeddings[anomaly_mask, 1],
                   c='#F44336', alpha=0.5, s=15, label='Anomaly', edgecolors='none')
        ax.set_title('t-SNE: SW-WaveNet (All Machine IDs Combined)',
                      fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        tsne_all_path = os.path.join(OUTPUT_DIR, "tsne_combined.png")
        plt.savefig(tsne_all_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {tsne_all_path}")

    except Exception as e:
        print(f"  t-SNE visualization failed: {e}")

    # -- Summary ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Average AUC  : {avg_auc:.4f}")
    print(f"  Average pAUC : {avg_pauc:.4f}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print(f"    -- evaluation_results.txt")
    print(f"    -- tsne_per_machine_id.png")
    print(f"    -- tsne_combined.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
