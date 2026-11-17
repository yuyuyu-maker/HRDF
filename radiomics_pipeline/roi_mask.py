"""病灶 / 前列腺 ROI 掩膜（T2 网格，来自 Internal 原始目录）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .preprocess import load_sitk

ROI_SUFFIX_PRIORITY = ("T2_T", "T2_P")


def find_lesion_roi_path(
    case_internal_dir: Path,
    subject_id: str,
    *,
    lesion_only: bool = False,
) -> Path | None:
    """
    查找 ROI 掩膜路径。
    ``lesion_only=True``：仅 ``*_T2_T``（纯病灶，不用 T2_P）。
    否则优先 T2_T，其次 T2_P。
    """
    case_internal_dir = Path(case_internal_dir)
    suffixes = ("T2_T",) if lesion_only else ROI_SUFFIX_PRIORITY
    for suffix in suffixes:
        for ext in (".nii.gz", ".nii"):
            p = case_internal_dir / f"{subject_id}_{suffix}{ext}"
            if p.is_file():
                return p
    return None


def load_lesion_mask_sitk(roi_path: Path | str) -> sitk.Image:
    """任意非零标签视为 ROI。"""
    img = load_sitk(roi_path)
    arr = sitk.GetArrayFromImage(img)
    binary = (np.asarray(arr) > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(binary)
    out.CopyInformation(img)
    return sitk.Cast(out, sitk.sitkUInt8)


def lesion_mask_numpy(roi_path: Path | str) -> np.ndarray:
    return sitk.GetArrayFromImage(load_lesion_mask_sitk(roi_path)).astype(bool)
