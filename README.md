# HRDF

双分支晚期融合：影像组学通路与 AxialGate3D 深度通路。

```
p = 0.7 * p_rad + 0.3 * p_deep
```

- 组学通路：在病灶 ROI 内提取 T2、DWI、ADC 的 PyRadiomics 特征与生境特征，经标准化、互信息筛选（k = 50）和 PCA（保留 85% 方差）后，用类别平衡的逻辑回归得到 `p_rad`。
- 深度通路：把配准到 T2 的三通道病灶块重采样为 `3 × 64 × 64 × 12`，由 AxialGate3D 得到 `p_deep`。
- 融合：两条通路的折外概率按 0.7 / 0.3 加权。

影像和训练权重不在本仓库里。

## 环境

```bash
pip install -r requirements.txt -c constraints.txt
```

深度分支需要 PyTorch。有 NVIDIA GPU 时使用 `--device cuda`。

## 组学分支

`--internal-root` 是原始病例目录，`--align-root` 是配准结果目录。目录里还没有配准文件时，流水线会把 DWI、ADC 配准到 T2。

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

标签 CSV 需要 `subject_id`（或 `number`）和 `label` 两列。

## 深度分支

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

默认训练 100 个 epoch，最少 25 个 epoch 后按验证折内 holdout 早停（耐心 15）。每折权重写到 `deep_out/checkpoints/fold_XX.pt`。

## 融合

组学 OOF 使用深度分支 checkpoint 里保存的同一套折划分。

```bash
python late_fusion.py \
  -i radiomics_out \
  --deep-checkpoint-dir deep_out/checkpoints \
  --deep-cache deep_cache \
  --labels-csv /path/to/labels.csv \
  -o fusion_out
```

`fusion_out/oof_predictions.csv` 含两支概率和融合概率，`fusion_summary.json` 是汇总指标。
