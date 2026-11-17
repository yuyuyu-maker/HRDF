"""PyRadiomics：在自动前景 mask 上对每个模态提取特征。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from mri_register_to_t2 import _cast_float

# 一阶 + 形状 + 纹理等高阶类（每模态各算一套）
DEFAULT_FEATURE_CLASSES: tuple[str, ...] = (
    "firstorder",
    "shape",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
)

# 对齐文献常用：每生境仅一阶 + 形状（不含纹理）
COMPACT_FEATURE_CLASSES: tuple[str, ...] = (
    "firstorder",
    "shape",
)

_EXTRACTORS: dict[tuple[str, ...], Any] = {}


def _scalar_feature_value(v: Any) -> float | None:
    """
    PyRadiomics 在 float 图像上常返回 0 维 ``numpy.ndarray``，原先用
    ``isinstance(v, (int, float))`` 会全部丢弃，导致 ``rad_*`` 为空。
    """
    if isinstance(v, str):
        return None
    try:
        if isinstance(v, np.ndarray):
            if v.size != 1:
                return None
            f = float(v.reshape(-1)[0])
        else:
            f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _extractor(
    feature_classes: tuple[str, ...] = DEFAULT_FEATURE_CLASSES,
) -> Any:
    key = tuple(feature_classes)
    if key in _EXTRACTORS:
        return _EXTRACTORS[key]

    ex = featureextractor.RadiomicsFeatureExtractor()
    ex.settings["binWidth"] = 25
    ex.settings["normalize"] = False
    ex.settings["verbose"] = False
    daf = getattr(ex, "disableAllFeatures", None)
    if callable(daf):
        try:
            daf()
        except Exception:
            pass
    for cls in feature_classes:
        try:
            ex.enableFeatureClassByName(cls)
        except Exception:
            pass
    _EXTRACTORS[key] = ex
    return ex


def reset_extractor() -> None:
    """测试或更换特征类后调用。"""
    global _EXTRACTORS
    _EXTRACTORS = {}


def extract_radiomics_for_image(
    image_sitk: sitk.Image,
    label_sitk: sitk.Image,
    prefix: str,
    *,
    feature_classes: tuple[str, ...] = DEFAULT_FEATURE_CLASSES,
) -> dict[str, float]:
    ex = _extractor(feature_classes)
    img = _cast_float(image_sitk)
    out = ex.execute(img, label_sitk)
    feats: dict[str, float] = {}
    for k, v in out.items():
        if k.startswith("diagnostics_"):
            continue
        val = _scalar_feature_value(v)
        if val is not None:
            feats[f"{prefix}_{k}"] = val
    return feats


def extract_all_modalities(
    paths: dict[str, Path],
    mask_sitk: sitk.Image,
    *,
    feature_classes: tuple[str, ...] = DEFAULT_FEATURE_CLASSES,
) -> dict[str, float]:
    """``paths`` 键如 ``T2``、``DWI_reg``、``ADC_reg``（存在的键才提）。"""
    allf: dict[str, float] = {}
    for name, p in paths.items():
        if not p.is_file():
            continue
        img = sitk.ReadImage(str(p))
        sub = extract_radiomics_for_image(
            img,
            mask_sitk,
            prefix=f"rad_{name}",
            feature_classes=feature_classes,
        )
        allf.update(sub)
    return allf
