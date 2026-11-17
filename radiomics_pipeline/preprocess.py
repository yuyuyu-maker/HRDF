"""强度归一化 + 自动前景二值化（Otsu，非病灶分割；后续可换真实 ROI）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from mri_register_to_t2 import _cast_float


def load_sitk(path: Path | str) -> sitk.Image:
    return sitk.ReadImage(str(path))


def sitk_to_numpy_zyx(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(_cast_float(img))


def otsu_foreground_mask_sitk(image: sitk.Image) -> sitk.Image:
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    otsu.SetNumberOfHistogramBins(128)
    f = _cast_float(image)
    return sitk.Cast(otsu.Execute(f) > 0.5, sitk.sitkUInt8)


def get_foreground_mask_sitk(t2_path: Path | str) -> sitk.Image:
    """与 ``build_normalized_stack`` 中相同的 T2 Otsu 前景，用于 PyRadiomics label。"""
    return otsu_foreground_mask_sitk(load_sitk(t2_path))


def minmax_volume(vol: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    v = vol.astype(np.float64)
    if mask is not None:
        m = mask.astype(bool)
        if m.sum() < 10:
            lo, hi = np.percentile(v, [0.5, 99.5])
        else:
            lo, hi = v[m].min(), v[m].max()
    else:
        lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    out = (v - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def zscore_after_clip(vol: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    v = vol.astype(np.float64)
    if mask is not None and mask.sum() > 50:
        vals = v[mask.astype(bool)]
        lo, hi = np.percentile(vals, [0.5, 99.5])
    else:
        lo, hi = np.percentile(v, [0.5, 99.5])
    v = np.clip(v, lo, hi)
    if mask is not None and mask.sum() > 10:
        mu, sd = float(v[mask].mean()), float(v[mask].std())
    else:
        mu, sd = float(v.mean()), float(v.std())
    if sd < 1e-8:
        sd = 1.0
    return ((v - mu) / sd).astype(np.float32)


def build_normalized_stack(
    t2_path: Path,
    dwi_reg_path: Path,
    adc_reg_path: Path | None,
    *,
    use_adc: bool,
    roi_mask_zyx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    返回 ``(stack_zyx_c, mask_zyx, channel_names)``。
    stack 通道顺序：T2（min-max）、DWI（min-max）、ADC（分位裁剪+z-score，若存在）。
    mask：默认 T2 Otsu；若提供 ``roi_mask_zyx`` 则用病灶/前列腺 ROI。
    """
    t2_img = load_sitk(t2_path)
    dwi_img = load_sitk(dwi_reg_path)
    t2 = sitk_to_numpy_zyx(t2_img)
    dwi = sitk_to_numpy_zyx(dwi_img)
    if t2.shape != dwi.shape:
        raise ValueError(f"T2 与 DWI_reg 形状不一致: {t2.shape} vs {dwi.shape}")

    if roi_mask_zyx is not None:
        mask = np.asarray(roi_mask_zyx).astype(bool)
        if mask.shape != t2.shape:
            raise ValueError(f"ROI 与 T2 形状不一致: {mask.shape} vs {t2.shape}")
        if mask.sum() < 10:
            raise ValueError("ROI 内体素过少 (<10)")
    else:
        mask_img = otsu_foreground_mask_sitk(t2_img)
        mask = sitk.GetArrayFromImage(mask_img).astype(bool)

    t2_n = minmax_volume(t2, mask)
    dwi_n = minmax_volume(dwi, mask)
    names = ["T2", "DWI"]
    channels: list[np.ndarray] = [t2_n, dwi_n]

    if use_adc and adc_reg_path is not None:
        adc = sitk_to_numpy_zyx(load_sitk(adc_reg_path))
        if adc.shape != t2.shape:
            raise ValueError(f"ADC_reg 形状不一致: {adc.shape} vs {t2.shape}")
        adc_n = zscore_after_clip(adc, mask)
        channels.append(adc_n)
        names.append("ADC")

    stack = np.stack(channels, axis=0)
    return stack, mask.astype(np.uint8), names
