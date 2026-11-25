# HRDF

Dual-branch late fusion of a radiomics pathway and an AxialGate3D pathway.

```
p = 0.7 * p_rad + 0.3 * p_deep
```

- Radiomics pathway: PyRadiomics and habitat features are extracted inside the lesion ROI on T2, DWI, and ADC. After scaling, mutual-information selection (k = 50), and PCA (85% variance retained), a class-balanced logistic regression produces `p_rad`.
- Deep pathway: the three-channel lesion patch, registered to T2, is resampled to `3 × 64 × 64 × 12`. AxialGate3D produces `p_deep`.
- Fusion: the out-of-fold probabilities are combined with weights 0.7 (radiomics) and 0.3 (deep).
- Decision threshold: accuracy and F1 use the Youden cutoff. On each training fold it is chosen from that fold's training probabilities. Pooled OOF summaries use the Youden point of the reported curve.

Images and trained weights are not included in this repository.

## Setup

```bash
pip install -r requirements.txt -c constraints.txt
```

The deep pathway requires PyTorch. Use `--device cuda` when an NVIDIA GPU is available.

## Radiomics pathway

`--internal-root` is the directory of original cases. `--align-root` is where registered volumes are written. If a case has not been registered yet, the pipeline registers DWI and ADC to T2.

```bash
python -m radiomics_pipeline.run_pipeline \
  --internal-root /path/to/Internal \
  --align-root /path/to/Internal_registered \
  --roi-mask-root /path/to/Internal \
  --labels-csv /path/to/labels.csv \
  --lesion-only \
  --min-roi-voxels 800 \
  --output-dir radiomics_out
```

The label CSV needs `subject_id` (or `number`) and `label`.

## Deep pathway

```bash
python -m radiomics_pipeline.deep_cv \
  --registered-root /path/to/Internal_registered \
  --roi-mask-root /path/to/Internal \
  --labels-csv /path/to/labels.csv \
  --lesion-only \
  --min-roi-voxels 800 \
  --output-dir deep_out \
  --cache-npy deep_cache \
  --device cuda
```

Training runs for 100 epochs by default. Early stopping starts after 25 epochs and uses a holdout split inside the training fold (patience 15). Each fold is saved to `deep_out/checkpoints/fold_XX.pt`.

## Fusion

Radiomics out-of-fold predictions use the same folds stored in the deep-pathway checkpoints.

```bash
python late_fusion.py \
  -i radiomics_out \
  --deep-checkpoint-dir deep_out/checkpoints \
  --deep-cache deep_cache \
  --labels-csv /path/to/labels.csv \
  -o fusion_out
```

`fusion_out/oof_predictions.csv` contains both branch probabilities and the fused probability. `fusion_summary.json` contains the summary metrics.
