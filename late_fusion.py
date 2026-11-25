#!/usr/bin/env python3
"""双分支晚期融合：p = w * p_rad + (1 - w) * p_deep，默认 w = 0.7。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from radiomics_pipeline.deep_cv import (
    N_INPUT_CHANNELS,
    _cache_name,
    _ensure_channels,
    _require_torch,
    build_3d_model,
)
from radiomics_pipeline.id_list import load_subject_ids
from radiomics_pipeline.model_cv import _nan_safe_matrix, collect_oof_probs, youden_threshold
from radiomics_pipeline.run_pipeline import load_label_csv


def _log(msg: str) -> None:
    print(msg, flush=True)


def _sid(value: object) -> str:
    text = str(value)
    return text.zfill(3) if text.isdigit() else text


def load_fold_splits_from_checkpoints(checkpoint_dir: Path) -> dict[int, list[str]]:
    """从 AxialGate checkpoint 读取各折验证集 subject_id。"""
    torch, _, _, _ = _require_torch()
    splits: dict[int, list[str]] = {}
    for pt in sorted(checkpoint_dir.glob("fold_*.pt")):
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        fold = int(payload.get("fold", len(splits) + 1))
        fold_idx = fold - 1 if fold >= 1 else fold
        val_ids = payload.get("val_subject_ids") or []
        splits[fold_idx] = [_sid(s) for s in val_ids]
    if not splits:
        raise FileNotFoundError(f"未找到有效 checkpoint: {checkpoint_dir}")
    return splits


def collect_deep_oof_from_checkpoints(
    checkpoint_dir: Path,
    cache_npy: Path,
    *,
    spatial: tuple[int, int, int] = (64, 64, 12),
    mask_tag: str = "lesion_t2t",
    device: str = "cpu",
) -> pd.DataFrame:
    """用已训练的 AxialGate 权重在对应验证折上推理。"""
    torch, _, _, _ = _require_torch()
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    records: list[dict] = []

    for pt in sorted(checkpoint_dir.glob("fold_*.pt")):
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        fold = int(payload.get("fold", 0))
        fold_idx = fold - 1 if fold >= 1 else fold
        val_ids = payload.get("val_subject_ids") or []
        model_name = payload.get("model_name") or payload.get("model") or "axialgate"
        if model_name != "axialgate":
            raise ValueError(f"{pt.name} 不是 AxialGate3D 权重: {model_name}")

        model = build_3d_model(model_name, N_INPUT_CHANNELS, 2)
        model.load_state_dict(payload["state_dict"])
        model.to(dev)
        model.eval()

        for sid in val_ids:
            sid = _sid(sid)
            cache_path = cache_npy / _cache_name(sid, spatial, mask_tag=mask_tag)
            if not cache_path.is_file():
                raise FileNotFoundError(f"缺少深度分支缓存: {cache_path}")
            vol = _ensure_channels(np.load(cache_path))
            with torch.no_grad():
                logit = model(torch.from_numpy(vol).unsqueeze(0).to(dev))
                prob = torch.softmax(logit, dim=1)[0, 1].item()
            records.append(
                {
                    "subject_id": sid,
                    "oof_prob_deep": float(prob),
                    "fold": fold_idx,
                }
            )

    if not records:
        raise RuntimeError("未能从 checkpoint 收集深度分支 OOF")
    return pd.DataFrame(records)


def evaluate_binary(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    thr = youden_threshold(y_true, prob)
    pred = (prob >= thr).astype(int)
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, prob))
    return {
        "roc_auc": auc,
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "youden_threshold": thr,
    }


def fold_auc(folds: pd.Series, y_true: pd.Series, prob: pd.Series) -> dict:
    scores: list[float] = []
    for _, idx in folds.groupby(folds).groups.items():
        y = y_true.loc[idx].astype(int).to_numpy()
        p = prob.loc[idx].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        scores.append(float(roc_auc_score(y, p)))
    if not scores:
        return {"roc_auc_mean": float("nan"), "roc_auc_std": float("nan"), "fold_roc_auc": []}
    return {
        "roc_auc_mean": float(np.mean(scores)),
        "roc_auc_std": float(np.std(scores)),
        "fold_roc_auc": scores,
    }


def main() -> None:
    default_labels = _root.parent / "MRI-DN" / "MRI" / "Internal" / "PTEN label.csv"
    ap = argparse.ArgumentParser(description="组学分支与 AxialGate3D 的加权晚期融合")
    ap.add_argument("-i", "--radiomics-dir", type=Path, required=True)
    ap.add_argument("--deep-checkpoint-dir", type=Path, required=True)
    ap.add_argument("--deep-cache", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, default=default_labels)
    ap.add_argument("-o", "--output-dir", type=Path, default=_root / "fusion_out")
    ap.add_argument("--ids-file", type=Path, default=None)
    ap.add_argument("--select-k", type=int, default=50)
    ap.add_argument("--pca-variance", type=float, default=0.85)
    ap.add_argument("--k-fold", type=int, default=5)
    ap.add_argument("--weight-rad", type=float, default=0.7)
    ap.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--mask-tag", type=str, default="lesion_t2t")
    ap.add_argument(
        "--spatial",
        type=int,
        nargs=3,
        default=(64, 64, 12),
        metavar=("Z", "Y", "X"),
    )
    args = ap.parse_args()
    if not 0.0 <= args.weight_rad <= 1.0:
        raise SystemExit("--weight-rad 需要在 0 到 1 之间")

    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    labels = load_label_csv(args.labels_csv)
    labels["subject_id"] = labels["subject_id"].map(_sid)

    feat = pd.read_csv(args.radiomics_dir / "feature_matrix.csv")
    feat["subject_id"] = feat["subject_id"].map(_sid)
    if args.ids_file is not None:
        allow = set(load_subject_ids(args.ids_file))
        feat = feat[feat["subject_id"].isin(allow)]

    sids = sorted(feat["subject_id"].unique().tolist())
    sub = feat.set_index("subject_id").loc[sids]
    X = _nan_safe_matrix(sub.select_dtypes(include=["float", "int", "int64", "float64"]))
    y = labels.set_index("subject_id").loc[sids]["label"].astype(int)

    _log(f"样本数: {len(sids)}  特征维: {X.shape[1]}")
    fold_splits = load_fold_splits_from_checkpoints(args.deep_checkpoint_dir)
    rad_oof = collect_oof_probs(
        X,
        y,
        sids,
        n_splits=args.k_fold,
        select_k=args.select_k,
        pca_variance=args.pca_variance,
        val_ids_by_fold=fold_splits,
    ).rename(columns={"oof_prob": "oof_prob_rad"})
    deep_oof = collect_deep_oof_from_checkpoints(
        args.deep_checkpoint_dir,
        args.deep_cache,
        spatial=tuple(args.spatial),
        mask_tag=args.mask_tag,
        device=args.device,
    )

    merged = rad_oof.merge(deep_oof, on=["subject_id", "fold"], how="inner")
    if len(merged) != len(sids):
        missing = sorted(set(sids) - set(merged["subject_id"]))
        raise RuntimeError(f"OOF 合并后样本不完整，缺失: {missing[:10]}")

    w = float(args.weight_rad)
    merged["oof_prob_fuse"] = w * merged["oof_prob_rad"] + (1.0 - w) * merged["oof_prob_deep"]
    merged = merged.sort_values("subject_id").reset_index(drop=True)
    merged.to_csv(outdir / "oof_predictions.csv", index=False)

    y_arr = merged["y_true"].astype(int).to_numpy()
    summary = {
        "method": "weighted_late_fusion",
        "decision_threshold": "youden",
        "weight_rad": w,
        "weight_deep": 1.0 - w,
        "n_samples": int(len(merged)),
        "select_k": args.select_k,
        "pca_variance": args.pca_variance,
        "radiomics_oof": evaluate_binary(y_arr, merged["oof_prob_rad"].to_numpy()),
        "deep_oof": evaluate_binary(y_arr, merged["oof_prob_deep"].to_numpy()),
        "fusion_oof": evaluate_binary(y_arr, merged["oof_prob_fuse"].to_numpy()),
        "fusion_folds": fold_auc(merged["fold"], merged["y_true"], merged["oof_prob_fuse"]),
    }
    with open(outdir / "fusion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fusion = summary["fusion_oof"]
    folds = summary["fusion_folds"]
    _log(
        f"融合 OOF AUC={fusion['roc_auc']:.4f}  "
        f"折均值={folds['roc_auc_mean']:.4f}±{folds['roc_auc_std']:.4f}"
    )
    _log(f"结果: {outdir / 'fusion_summary.json'}")


if __name__ == "__main__":
    main()
