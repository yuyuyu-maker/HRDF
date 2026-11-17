"""单病例：对齐路径 + 生境划分 + PyRadiomics（整灶 / 分生境）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from mri_case_reader import discover_nifti_stems

from . import habitat, preprocess, radiomics_fe, roi_mask
from .align_data import case_subject_id


def _numpy_mask_to_sitk(mask_zyx: np.ndarray, ref: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(mask_zyx.astype(np.uint8))
    out.CopyInformation(ref)
    return sitk.Cast(out, sitk.sitkUInt8)


def extract_one_case(
    case_dir: Path,
    align_dir: Path,
    *,
    n_habitat_clusters: int = 4,
    roi_mask_root: Path | None = None,
    lesion_only: bool = False,
    habitat_mode: str = "summary",
    habitat_k_min: int = 3,
    habitat_k_max: int = 4,
    habitat_min_voxels: int = 30,
    habitat_fixed_k: int | None = None,
) -> dict[str, float | str]:
    """
    ``habitat_mode``:
      - ``summary``: 轻量生境摘要 + 整灶全套 radiomics
      - ``per_region``: CH 选 K → 每生境全套 radiomics + 整灶全套
      - ``per_region_compact``: CH 选 K（与之前一致）→ 每生境仅一阶+形状
        （对齐文献精简特征，不含纹理）+ 整灶一阶+形状对照 + 轻量摘要
      - ``both``: summary + per_region 全套
    """
    case_dir = Path(case_dir).resolve()
    align_dir = Path(align_dir).resolve()
    stems = discover_nifti_stems(case_dir)
    sid = case_subject_id(case_dir)
    if "T2" not in stems:
        raise ValueError(f"病例无 T2: {case_dir}")
    t2_path = Path(stems["T2"])
    dwi_reg = align_dir / f"{sid}_DWI_reg2T2.nii.gz"
    adc_reg = align_dir / f"{sid}_ADC_reg2T2.nii.gz"
    use_adc = adc_reg.is_file()
    adc_arg: Path | None = adc_reg if use_adc else None

    roi_path = None
    if roi_mask_root is not None:
        roi_path = roi_mask.find_lesion_roi_path(
            roi_mask_root / sid, sid, lesion_only=lesion_only
        )
        if roi_path is None:
            hint = "T2_T" if lesion_only else "T2_T/T2_P"
            raise FileNotFoundError(f"无 {hint} ROI: {roi_mask_root / sid}")

    roi_zyx = roi_mask.lesion_mask_numpy(roi_path) if roi_path else None
    stack, mask, _ch = preprocess.build_normalized_stack(
        t2_path,
        dwi_reg,
        adc_arg,
        use_adc=use_adc,
        roi_mask_zyx=roi_zyx,
    )

    if roi_path is not None:
        mask_sitk = roi_mask.load_lesion_mask_sitk(roi_path)
    else:
        mask_sitk = preprocess.get_foreground_mask_sitk(t2_path)

    paths: dict[str, Path] = {"T2": t2_path, "DWI_reg": dwi_reg}
    if use_adc:
        paths["ADC_reg"] = adc_reg

    row: dict[str, float | str] = {"subject_id": sid}
    mode = (habitat_mode or "summary").lower().strip()
    max_hab = max(int(habitat_k_max), int(n_habitat_clusters), 2)
    compact = mode == "per_region_compact"
    whole_classes = (
        radiomics_fe.COMPACT_FEATURE_CLASSES
        if compact
        else radiomics_fe.DEFAULT_FEATURE_CLASSES
    )
    hab_classes = (
        radiomics_fe.COMPACT_FEATURE_CLASSES
        if compact
        else radiomics_fe.DEFAULT_FEATURE_CLASSES
    )

    # 整灶 radiomics（对照）
    rfeats = radiomics_fe.extract_all_modalities(
        paths, mask_sitk, feature_classes=whole_classes
    )
    row.update(rfeats)

    if mode in ("summary", "both", "per_region_compact"):
        # compact 模式也保留轻量占比/均值，便于消融
        if mode == "per_region_compact":
            pass  # 摘要由 partition 后写入
        else:
            hfeats = habitat.habitat_features_from_stack(
                stack, mask, n_clusters=n_habitat_clusters
            )
            row.update(hfeats)

    do_region = mode in ("per_region", "both", "per_region_compact")
    if do_region:
        part = habitat.partition_habitats(
            stack,
            mask,
            k_min=habitat_k_min,
            k_max=habitat_k_max,
            fixed_k=habitat_fixed_k,
            min_voxels_per_cluster=habitat_min_voxels,
            order_channel=-1,
        )
        if part is None:
            row["habitat_selected_k"] = 0.0
            row["habitat_ch_score"] = 0.0
            row["habitat_partition_ok"] = 0.0
        else:
            row.update(
                habitat.habitat_summary_from_partition(
                    part, stack, max_habitats=max_hab
                )
            )
            row["habitat_partition_ok"] = 1.0
            prefix_tag = "hab"  # hab{k}_rad_* ；compact 与全套同前缀，目录分开即可
            for k in range(max_hab):
                sel = part.labels_zyx == k
                if int(sel.sum()) < habitat_min_voxels:
                    continue
                if sel.shape != tuple(mask_sitk.GetSize()[::-1]):
                    raise ValueError(
                        f"{sid}: habitat mask shape {sel.shape} "
                        f"!= sitk ZYX {tuple(mask_sitk.GetSize()[::-1])}"
                    )
                hab_mask = _numpy_mask_to_sitk(sel, mask_sitk)
                for name, p in paths.items():
                    if not p.is_file():
                        continue
                    img = sitk.ReadImage(str(p))
                    try:
                        sub = radiomics_fe.extract_radiomics_for_image(
                            img,
                            hab_mask,
                            prefix=f"{prefix_tag}{k}_rad_{name}",
                            feature_classes=hab_classes,
                        )
                        row.update(sub)
                    except Exception:
                        continue

    return row
