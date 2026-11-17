"""端到端：对齐 → 归一化/自动前景 → 生境 K-means + PyRadiomics → K 折 + 逻辑回归。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd

from .align_data import (
    case_subject_id,
    ensure_case_registered,
    needs_registration,
)
from .case_features import extract_one_case
from .cohort_qc import filter_subject_ids
from .feature_ranking import rank_features_mutual_info
from .id_list import load_subject_ids
from .model_cv import run_stratified_kfold


def _list_case_dirs(internal_root: Path) -> list[Path]:
    internal_root = internal_root.resolve()
    out: list[Path] = []
    for p in sorted(internal_root.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        out.append(p)
    return out


def load_label_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "number" in df.columns and "label" in df.columns:
        num = df["number"]
        sid = []
        for x in num:
            if pd.isna(x):
                sid.append("")
            elif isinstance(x, (int, float)) and float(x).is_integer():
                sid.append(f"{int(float(x)):03d}")
            else:
                s = str(x).strip()
                sid.append(s.zfill(3) if s.isdigit() else s)
        df = df.assign(subject_id=sid, label=df["label"])
    elif "subject_id" not in df.columns or "label" not in df.columns:
        raise ValueError("标签 CSV 需包含 (number, label) 或 (subject_id, label)")
    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    cols = ["subject_id", "label"]
    if "cohort" in df.columns:
        cols.append("cohort")
    return df[cols].dropna(subset=["subject_id"])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_internal = root.parent / "MRI-DN" / "MRI" / "Internal"
    default_labels = default_internal / "PTEN label.csv"

    parser = argparse.ArgumentParser(
        description="影像组学+生境流水线：对齐→特征→K折逻辑回归",
    )
    parser.add_argument(
        "--internal-root",
        type=Path,
        default=default_internal,
        help="含各病例子文件夹的 Internal 根目录",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=default_labels,
        help="含 subject_id(或 number), label 的 CSV",
    )
    parser.add_argument(
        "--align-root",
        type=Path,
        default=root / "aligned_cache",
        help="配准输出根目录（每病例子目录）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "pipeline_out",
        help="特征表与 CV 结果输出目录",
    )
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--n-habitat-clusters", type=int, default=4)
    parser.add_argument(
        "--habitat-mode",
        type=str,
        default="summary",
        choices=("summary", "per_region", "per_region_compact", "both"),
        help="summary=轻量生境; per_region=全套生境组学; "
        "per_region_compact=仅一阶+形状生境组学(CH选K同前); both=summary+全套",
    )
    parser.add_argument("--habitat-k-min", type=int, default=3)
    parser.add_argument("--habitat-k-max", type=int, default=4)
    parser.add_argument(
        "--habitat-min-voxels",
        type=int,
        default=30,
        help="每个生境最少体素数，过小则跳过该子区组学",
    )
    parser.add_argument(
        "--habitat-fixed-k",
        type=int,
        default=None,
        help="若指定则不做 CH 选 K，强制该 K（消融用）",
    )
    parser.add_argument("--select-k", type=int, default=50)
    parser.add_argument(
        "--no-feature-ranking",
        action="store_true",
        help="不写出 feature_importance_mi.csv",
    )
    parser.add_argument("--pca-variance", type=float, default=0.85)
    parser.add_argument("--reg-iterations", type=int, default=400)
    parser.add_argument("--affine-refine", action="store_true")
    parser.add_argument("--no-hist-match", action="store_true")
    parser.add_argument(
        "--roi-mask-root",
        type=Path,
        default=None,
        help="Internal 原始目录（含 *_T2_T / *_T2_P），启用后仅在 ROI 内提特征",
    )
    parser.add_argument(
        "--min-roi-voxels",
        type=int,
        default=0,
        help="病灶 ROI 最少体素数（需配合 --roi-mask-root）",
    )
    parser.add_argument(
        "--exclude-ids",
        type=str,
        default="",
        help="逗号分隔排除病例，如 554",
    )
    parser.add_argument(
        "--lesion-only",
        action="store_true",
        help="仅使用 *_T2_T 病灶掩膜（不用 T2_P）；无 T2_T 的病例剔除",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="仅处理列表中的 subject_id（.txt 每行一个或 selected_cohort.json）",
    )
    args = parser.parse_args()

    labels = load_label_csv(args.labels_csv)
    label_set = set(labels["subject_id"].tolist())

    case_dirs = _list_case_dirs(args.internal_root)
    if args.max_cases is not None:
        case_dirs = case_dirs[: args.max_cases]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.align_root.mkdir(parents=True, exist_ok=True)

    roi_root = args.roi_mask_root.expanduser().resolve() if args.roi_mask_root else None
    exclude = {x.strip().zfill(3) for x in args.exclude_ids.split(",") if x.strip()}
    todo_sids = [case_subject_id(d) for d in case_dirs if case_subject_id(d) in label_set]
    if args.min_roi_voxels > 0 or exclude or args.lesion_only:
        kept, dropped = filter_subject_ids(
            todo_sids,
            registered_root=args.align_root,
            roi_mask_root=roi_root,
            min_roi_voxels=args.min_roi_voxels,
            exclude_ids=exclude or None,
            lesion_only=args.lesion_only,
        )
        todo_sids = kept
        qc_path = args.output_dir / "qc_excluded.json"
        with open(qc_path, "w", encoding="utf-8") as f:
            json.dump({"excluded": dropped, "n_kept": len(kept)}, f, indent=2, ensure_ascii=False)
        print(f"质控: 剔除 {len(dropped)} 例，保留 {len(kept)} 例 -> {qc_path}", flush=True)
    if args.ids_file is not None:
        allow = set(load_subject_ids(args.ids_file))
        todo_sids = [s for s in todo_sids if s in allow]
        print(f"--ids-file: 仅处理 {len(todo_sids)} 例", flush=True)
    todo = [d for d in case_dirs if case_subject_id(d) in todo_sids]
    n_todo = len(todo)
    if roi_root:
        mode = "纯病灶 T2_T" if args.lesion_only else "T2_T 优先（可回退 T2_P）"
        print(f"ROI 掩膜: {roi_root}  ({mode})", flush=True)
    print(f"待处理 {n_todo} 例（含高阶 PyRadiomics + 生境）...", flush=True)

    rows: list[dict] = []
    for i, case_dir in enumerate(todo, start=1):
        sid = case_subject_id(case_dir)
        align_dir = args.align_root / sid
        align_dir.mkdir(parents=True, exist_ok=True)
        try:
            if needs_registration(case_dir, align_dir):
                ensure_case_registered(
                    case_dir,
                    align_dir,
                    iterations=args.reg_iterations,
                    hist_match=not args.no_hist_match,
                    affine_refine=args.affine_refine,
                    save_transforms=False,
                )
            print(f"[{i}/{n_todo}] {sid} 特征提取中...", flush=True)
            feat_row = extract_one_case(
                case_dir,
                align_dir,
                n_habitat_clusters=args.n_habitat_clusters,
                roi_mask_root=roi_root,
                lesion_only=args.lesion_only,
                habitat_mode=args.habitat_mode,
                habitat_k_min=args.habitat_k_min,
                habitat_k_max=args.habitat_k_max,
                habitat_min_voxels=args.habitat_min_voxels,
                habitat_fixed_k=args.habitat_fixed_k,
            )
            n_rad = sum(1 for k in feat_row if str(k).startswith("rad_"))
            n_hab = sum(1 for k in feat_row if str(k).startswith("hab") and "_rad_" in str(k))
            sk = feat_row.get("habitat_selected_k", "")
            print(
                f"[{i}/{n_todo}] {sid} 完成  rad={n_rad} hab_rad={n_hab} K={sk}",
                flush=True,
            )
            rows.append(feat_row)
        except Exception as e:
            print(f"[{i}/{n_todo}] [skip] {sid}: {e}", file=sys.stderr, flush=True)

    if not rows:
        print("无有效特征行，请检查路径与标签。", file=sys.stderr)
        sys.exit(1)

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(args.output_dir / "feature_matrix.csv", index=False)

    y = labels.set_index("subject_id")["label"]
    feat_df = feat_df.set_index("subject_id")
    common = feat_df.index.intersection(y.index)
    X = feat_df.loc[common].select_dtypes(include=["float", "int", "int64", "float64"])
    y = y.loc[common]
    min_class = int(y.value_counts().min())
    eff_splits = min(args.k_fold, max(2, min_class))
    if eff_splits < args.k_fold:
        print(
            f"注意: 最少类样本数为 {min_class}，已将折数从 {args.k_fold} 调整为 {eff_splits}。",
            file=sys.stderr,
        )

    cv_summary = run_stratified_kfold(
        X,
        y,
        n_splits=eff_splits,
        select_k=args.select_k,
        pca_variance=args.pca_variance,
    )
    with open(args.output_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2, ensure_ascii=False)

    if not args.no_feature_ranking:
        n_rad = sum(1 for c in X.columns if str(c).startswith("rad_") and not str(c).startswith("hab"))
        n_hab_sum = sum(1 for c in X.columns if str(c).startswith("habitat_"))
        n_hab_rad = sum(1 for c in X.columns if str(c).startswith("hab") and "_rad_" in str(c))
        rank_df = rank_features_mutual_info(X, y)
        rank_path = args.output_dir / "feature_importance_mi.csv"
        rank_df.to_csv(rank_path, index=False)
        cv_summary["n_features_total"] = int(X.shape[1])
        cv_summary["n_radiomics"] = n_rad
        cv_summary["n_habitat"] = n_hab_sum
        cv_summary["n_habitat_radiomics"] = n_hab_rad
        cv_summary["habitat_mode"] = args.habitat_mode
        cv_summary["habitat_k_range"] = [args.habitat_k_min, args.habitat_k_max]
        if "habitat_selected_k" in feat_df.columns:
            vc = feat_df["habitat_selected_k"].astype(float).astype(int).value_counts().sort_index()
            cv_summary["habitat_k_counts"] = {str(int(k)): int(v) for k, v in vc.items()}
        with open(args.output_dir / "cv_summary.json", "w", encoding="utf-8") as f:
            json.dump(cv_summary, f, indent=2, ensure_ascii=False)
        print(
            f"特征数: 总计 {X.shape[1]} (整灶 rad_* {n_rad}, "
            f"生境摘要 {n_hab_sum}, 生境组学 hab*_rad_* {n_hab_rad})"
        )
        print(f"互信息排序: {rank_path}")

    print(
        f"样本数={cv_summary['n_samples']}  "
        f"ROC-AUC={cv_summary['roc_auc_mean']:.3f}±{cv_summary['roc_auc_std']:.3f}  "
        f"Acc={cv_summary['accuracy_mean']:.3f}  F1={cv_summary['f1_mean']:.3f}"
    )
    print(f"特征表: {args.output_dir / 'feature_matrix.csv'}")
    print(f"CV 结果: {args.output_dir / 'cv_summary.json'}")


if __name__ == "__main__":
    main()
