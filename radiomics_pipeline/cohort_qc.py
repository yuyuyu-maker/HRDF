"""入组质控：配准文件、病灶 ROI 体积、排除列表。"""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from .roi_mask import find_lesion_roi_path, lesion_mask_numpy


def roi_voxel_count(
    roi_mask_root: Path,
    subject_id: str,
    *,
    lesion_only: bool = False,
) -> int:
    roi_path = find_lesion_roi_path(
        roi_mask_root / subject_id, subject_id, lesion_only=lesion_only
    )
    if roi_path is None:
        return 0
    return int(lesion_mask_numpy(roi_path).sum())


def check_case_qc(
    subject_id: str,
    *,
    registered_case_dir: Path,
    roi_mask_root: Path | None,
    min_roi_voxels: int = 0,
    require_dwi_reg: bool = True,
    exclude_ids: set[str] | None = None,
    lesion_only: bool = False,
) -> tuple[bool, str]:
    """返回 (通过, 原因)。"""
    sid = subject_id.strip().zfill(3) if subject_id.isdigit() else subject_id
    if exclude_ids and sid in exclude_ids:
        return False, "exclude_ids"

    dwi = registered_case_dir / f"{sid}_DWI_reg2T2.nii.gz"
    if require_dwi_reg and not dwi.is_file():
        return False, "missing DWI_reg2T2"

    t2 = list(registered_case_dir.glob(f"{sid}_T2.nii*"))
    if not t2:
        return False, "missing T2"

    if roi_mask_root is not None:
        roi_path = find_lesion_roi_path(
            roi_mask_root / sid, sid, lesion_only=lesion_only
        )
        if roi_path is None:
            return False, "missing_T2_T" if lesion_only else "missing_ROI"
        if min_roi_voxels > 0:
            n = int(lesion_mask_numpy(roi_path).sum())
            if n < min_roi_voxels:
                return False, f"roi_voxels={n}<{min_roi_voxels}"

    return True, "ok"


def filter_subject_ids(
    subject_ids: list[str],
    *,
    registered_root: Path,
    roi_mask_root: Path | None,
    min_roi_voxels: int = 0,
    require_dwi_reg: bool = True,
    exclude_ids: set[str] | None = None,
    lesion_only: bool = False,
) -> tuple[list[str], list[dict]]:
    """返回 (通过列表, 剔除记录)。"""
    registered_root = registered_root.resolve()
    if roi_mask_root:
        roi_mask_root = roi_mask_root.resolve()

    kept: list[str] = []
    dropped: list[dict] = []
    for sid in subject_ids:
        sid = str(sid).strip().zfill(3) if str(sid).strip().isdigit() else str(sid).strip()
        ok, reason = check_case_qc(
            sid,
            registered_case_dir=registered_root / sid,
            roi_mask_root=roi_mask_root,
            min_roi_voxels=min_roi_voxels,
            require_dwi_reg=require_dwi_reg,
            exclude_ids=exclude_ids,
            lesion_only=lesion_only,
        )
        if ok:
            kept.append(sid)
        else:
            dropped.append({"subject_id": sid, "reason": reason})
    return kept, dropped
