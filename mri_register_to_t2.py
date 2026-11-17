"""
将病例内 DWI、ADC 等 moving 序列配准到 T2（fixed），并重采样到 T2 体素网格。

默认采用 **相似性（7 自由度）+ 多分辨率 Mattes**；**仿射精化默认关闭**（T2–DWI 上易过拟合互信息、目视变差），需要时用 ``--affine-refine`` 开启二阶段。配准前 **直方图匹配**、T2 **前景度量掩膜**、**物理空间优化步长缩放**（`SetOptimizerScalesFromPhysicalShift`）。相似性失败时可回退多档相似性/刚性。

与 README 中「T2 为参考、多序列配准到 T2、ROI 在 T2 上定义」的管线一致。

依赖：SimpleITK。

QC：NCC、ΔNCC、棋盘格；另输出 **并排融合** 便于肉眼看对齐。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "请先安装 SimpleITK：pip install SimpleITK\n"
        "（若已安装仍报错，请检查当前 Python 环境。）"
    ) from e

from mri_case_reader import discover_nifti_stems


DEFAULT_MOVING_MODALITIES = ("DWI", "ADC")


@dataclass
class QCResult:
    modality: str
    ncc_initial: float
    ncc_registered: float
    delta_ncc: float


def _cast_float(img: sitk.Image) -> sitk.Image:
    """
    标量体转为 float32。若为向量体（如多 b 值 DWI、多回波 T2 存成
    ``VectorUInt16``），SimpleITK 的 ``Cast`` 不能直接转 float，先取第 0 分量。
    """
    if img.GetNumberOfComponentsPerPixel() > 1:
        return sitk.VectorIndexSelectionCast(img, 0, sitk.sitkFloat32)
    return sitk.Cast(img, sitk.sitkFloat32)


def _otsu_foreground_mask(fixed_f32: sitk.Image) -> sitk.Image:
    """在 T2 上 Otsu 二值化得到大致前景，用于 NCC 只在组织内统计。"""
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    otsu.SetNumberOfHistogramBins(128)
    return otsu.Execute(fixed_f32)


def _masked_pearson_r(
    fixed_arr: np.ndarray,
    moving_arr: np.ndarray,
    mask_arr: np.ndarray,
) -> float:
    m = mask_arr.astype(bool) & np.isfinite(fixed_arr) & np.isfinite(moving_arr)
    if m.sum() < 50:
        return float("nan")
    a = fixed_arr[m].ravel().astype(np.float64)
    b = moving_arr[m].ravel().astype(np.float64)
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float(np.mean(a * b))


def _sitk_arr(img: sitk.Image) -> np.ndarray:
    """numpy 形状为 **(z, y, x)**（与 SimpleITK ``GetArrayFromImage`` 一致），勿与 nibabel 常见 (x,y,z) 混淆。"""
    return sitk.GetArrayFromImage(img)


def _ncc_on_fixed_grid(
    fixed: sitk.Image,
    moving_resampled: sitk.Image,
    mask: sitk.Image,
) -> float:
    fa = _sitk_arr(_cast_float(fixed))
    ma = _sitk_arr(_cast_float(moving_resampled))
    mk = _sitk_arr(mask) > 0
    return _masked_pearson_r(fa, ma, mk)


def _histogram_match_moving_to_fixed(moving_f: sitk.Image, fixed_f: sitk.Image) -> sitk.Image:
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    return matcher.Execute(moving_f, fixed_f)


def _prepare_moving_for_metric(
    moving: sitk.Image,
    fixed_f: sitk.Image,
    *,
    hist_match: bool,
) -> sitk.Image:
    """用于度量的 moving（float）；直方图匹配只改强度，几何与原始 moving 一致。"""
    moving_f = _cast_float(moving)
    if hist_match:
        moving_f = _histogram_match_moving_to_fixed(moving_f, fixed_f)
    return moving_f


def _centered_transform_init(
    fixed_f: sitk.Image,
    moving_f: sitk.Image,
    *,
    transform: str,
    mode: str,
) -> sitk.Transform:
    """
    transform: ``"similarity"``、``"euler"`` 或 ``"affine"``（12 自由度）。
    mode: ``"geometry"`` 或 ``"moments"``。
    """
    strategy = (
        sitk.CenteredTransformInitializerFilter.MOMENTS
        if mode == "moments"
        else sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    if transform == "similarity":
        t = sitk.Similarity3DTransform()
    elif transform == "euler":
        t = sitk.Euler3DTransform()
    elif transform == "affine":
        t = sitk.AffineTransform(3)
    else:
        raise ValueError(f"未知 transform: {transform}")
    return sitk.CenteredTransformInitializer(fixed_f, moving_f, t, strategy)


def _dilated_foreground_mask_for_metric(fixed_f: sitk.Image) -> sitk.Image | None:
    """T2 Otsu 前景略膨胀，限制 Mattes 采样在解剖附近，减轻 FOV 外空气干扰。"""
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    otsu.SetNumberOfHistogramBins(128)
    raw = sitk.Cast(otsu.Execute(fixed_f) > 0.5, sitk.sitkUInt8)
    dil = sitk.BinaryDilate(raw, (5, 5, 2))
    st = sitk.StatisticsImageFilter()
    st.Execute(dil)
    if st.GetSum() < 500:
        return None
    return dil


def _register_mi_once(
    fixed_f: sitk.Image,
    moving_f: sitk.Image,
    *,
    number_of_iterations: int,
    learning_rate: float,
    shrink_factors: list[int],
    smoothing_sigmas: list[float],
    transform: str,
    init_mode: str,
    sampling_percentage: float,
    fixed_metric_mask: sitk.Image | None,
    moving_initial: sitk.Transform | None = None,
    optimizer: str = "line",
    metric_seed: int = 42,
) -> sitk.Transform:
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    reg.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.RANDOM)
    try:
        reg.SetMetricSamplingPercentage(sampling_percentage, metric_seed)
    except TypeError:
        reg.SetMetricSamplingPercentage(sampling_percentage)
    if fixed_metric_mask is not None:
        reg.SetMetricFixedMask(fixed_metric_mask)
    reg.SetInterpolator(sitk.sitkLinear)
    if optimizer == "regular":
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=learning_rate,
            minStep=0.01,
            numberOfIterations=number_of_iterations,
            relaxationFactor=0.6,
            gradientMagnitudeTolerance=1e-5,
        )
    else:
        reg.SetOptimizerAsGradientDescentLineSearch(
            learningRate=learning_rate,
            numberOfIterations=number_of_iterations,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
    assert len(shrink_factors) == len(smoothing_sigmas)
    reg.SetShrinkFactorsPerLevel(shrink_factors)
    reg.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    reg.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(True)

    if moving_initial is not None:
        reg.SetMovingInitialTransform(moving_initial)

    if moving_initial is not None and transform == "affine":
        init = sitk.AffineTransform(3)
    else:
        init = _centered_transform_init(fixed_f, moving_f, transform=transform, mode=init_mode)
    reg.SetInitialTransform(init, inPlace=False)
    try:
        reg.SetOptimizerScalesFromPhysicalShift()
    except Exception:
        pass
    return reg.Execute(fixed_f, moving_f)


def _compose_fixed_to_moving_sim_then_affine(
    dimension: int,
    sim_tx: sitk.Transform,
    aff_tx: sitk.Transform,
) -> sitk.Transform:
    """
    ITKv4：``p_moving = T_moving_initial(T_optimized(p_fixed))``。
    此处 ``sim_tx`` 为 moving 初值，``aff_tx`` 为优化得到的仿射；先 aff 再 sim。
    Composite 中 **后 Add 的变换先作用于点**。
    """
    c = sitk.CompositeTransform(dimension)
    c.AddTransform(sim_tx)
    c.AddTransform(aff_tx)
    return c


def _register_similarity_only_fallback(
    fixed_f: sitk.Image,
    moving_f: sitk.Image,
    *,
    number_of_iterations: int,
    learning_rate: float,
    metric_mask: sitk.Image | None,
) -> sitk.Transform:
    """单阶段相似性/刚性回退（与旧版逻辑相近）。"""
    attempts: list[tuple[str, list[int], list[float], str, float, float]] = [
        ("similarity", [2, 1], [1.0, 0.0], "geometry", 0.25, 1.0),
        ("similarity", [1], [0.0], "geometry", 0.4, 1.0),
        ("similarity", [1], [0.0], "moments", 0.4, 0.75),
        ("euler", [1], [0.0], "geometry", 0.4, 1.0),
    ]
    n_att = len(attempts)
    for i, (tx, shrink, smooth, init_mode, samp_pct, lr_scale) in enumerate(attempts):
        try:
            return _register_mi_once(
                fixed_f,
                moving_f,
                number_of_iterations=number_of_iterations,
                learning_rate=learning_rate * lr_scale,
                shrink_factors=shrink,
                smoothing_sigmas=smooth,
                transform=tx,
                init_mode=init_mode,
                sampling_percentage=samp_pct,
                fixed_metric_mask=metric_mask,
                moving_initial=None,
                optimizer="line",
            )
        except Exception as e:
            if i == n_att - 1:
                raise RuntimeError(f"相似性/刚性回退仍失败: {e}") from e
    raise RuntimeError("unreachable")


def _register_to_t2_transform(
    fixed: sitk.Image,
    moving: sitk.Image,
    *,
    number_of_iterations: int = 400,
    learning_rate: float = 1.0,
    hist_match: bool = True,
    affine_refine: bool = False,
) -> sitk.Transform:
    """
    主流程：**相似性（多分辨率）**；可选 ``affine_refine=True`` 时再跑 **仿射** 精化。

    默认只做相似性：在 T2–DWI 多模态上，Mattes 下的 12 自由度仿射常把互信息推好却把
    **解剖拉歪/过度平滑**，故默认关闭；若课题确有需要再加 ``--affine-refine`` 试验。

    使用直方图匹配、T2 前景度量掩膜、``SetOptimizerScalesFromPhysicalShift``。
    相似性失败时回退旧版多档相似性/刚性。
    """
    fixed_f = _cast_float(fixed)
    moving_f = _prepare_moving_for_metric(moving, fixed_f, hist_match=hist_match)
    metric_mask = _dilated_foreground_mask_for_metric(fixed_f)

    n_sim = (
        max(180, int(number_of_iterations * 2 / 3))
        if affine_refine
        else max(250, number_of_iterations)
    )
    n_aff = max(120, number_of_iterations - n_sim) if affine_refine else 0

    sim_tx: sitk.Transform | None = None
    try:
        sim_tx = _register_mi_once(
            fixed_f,
            moving_f,
            number_of_iterations=n_sim,
            learning_rate=learning_rate,
            shrink_factors=[2, 1],
            smoothing_sigmas=[1.0, 0.0],
            transform="similarity",
            init_mode="geometry",
            sampling_percentage=0.28,
            fixed_metric_mask=metric_mask,
            moving_initial=None,
            optimizer="line",
        )
    except Exception:
        return _register_similarity_only_fallback(
            fixed_f,
            moving_f,
            number_of_iterations=max(250, number_of_iterations),
            learning_rate=learning_rate,
            metric_mask=metric_mask,
        )

    if not affine_refine:
        return sim_tx

    try:
        aff_tx = _register_mi_once(
            fixed_f,
            moving_f,
            number_of_iterations=n_aff,
            learning_rate=min(learning_rate, 0.6),
            shrink_factors=[1],
            smoothing_sigmas=[0.0],
            transform="affine",
            init_mode="geometry",
            sampling_percentage=0.45,
            fixed_metric_mask=metric_mask,
            moving_initial=sim_tx,
            optimizer="regular",
        )
        return _compose_fixed_to_moving_sim_then_affine(
            fixed.GetDimension(), sim_tx, aff_tx
        )
    except Exception as e:
        import warnings

        warnings.warn(
            f"仿射精化未收敛或失败，已退回仅相似性变换（建议目视 QC）：{e}",
            UserWarning,
            stacklevel=2,
        )
        return sim_tx


def _initial_baseline_transform(
    fixed: sitk.Image,
    moving: sitk.Image,
    hist_match: bool,
) -> sitk.Transform:
    """与默认配准同类的「仅几何初值」相似变换，用于 ΔNCC 基线（不做迭代优化）。"""
    fixed_f = _cast_float(fixed)
    moving_f = _prepare_moving_for_metric(moving, fixed_f, hist_match=hist_match)
    return _centered_transform_init(fixed_f, moving_f, transform="similarity", mode="geometry")


def save_registration_transforms(tfm: sitk.Transform, dest_base: Path) -> list[str]:
    """
    将配准变换写入磁盘。

    Insight Legacy 的 ``.tfm`` **不能**保存单个 ``CompositeTransform``（会报
    ``Composite Transform can only be 1st transform in a file``）。
    若为 Composite，则按子变换顺序拆成 ``{dest_base}_00.tfm``、``_01.tfm`` …；
    单变换仍写 ``{dest_base}.tfm``（``dest_base`` 无扩展名）。

    复现时按序号依次 ``AddTransform`` 到新的 ``CompositeTransform`` 即可（与脚本内顺序一致）。
    """
    dest_base = dest_base.expanduser().resolve()
    parent = dest_base.parent
    stem = dest_base.name
    written: list[Path] = []

    n_sub = getattr(tfm, "GetNumberOfTransforms", None)
    n = n_sub() if callable(n_sub) else 1
    if n > 1:
        for i in range(n):
            sub = tfm.GetNthTransform(i)
            p = parent / f"{stem}_{i:02d}.tfm"
            sitk.WriteTransform(sub, str(p))
            written.append(p)
    else:
        p = parent / f"{stem}.tfm"
        sitk.WriteTransform(tfm, str(p))
        written.append(p)

    return [str(x) for x in written]


def resample_moving_to_fixed(
    moving: sitk.Image,
    fixed: sitk.Image,
    transform_fixed_to_moving: sitk.Transform,
    output_pixel_type: int | None = None,
) -> sitk.Image:
    """
    将 moving 重采样到 fixed 的网格上。
    ``transform_fixed_to_moving``：SimpleITK 配准得到的、把 fixed 物理坐标映射到 moving 物理坐标的变换。
    """
    pid = output_pixel_type if output_pixel_type is not None else moving.GetPixelID()
    return sitk.Resample(
        moving,
        fixed,
        transform_fixed_to_moving,
        sitk.sitkLinear,
        0.0,
        pid,
    )


def validate_roi_matches_fixed(roi: sitk.Image, fixed: sitk.Image, tol: float = 1e-4) -> None:
    if roi.GetSize() != fixed.GetSize():
        raise ValueError(f"ROI 与 T2 尺寸不一致: ROI {roi.GetSize()} vs T2 {fixed.GetSize()}")
    sp_r = np.array(roi.GetSpacing(), dtype=np.float64)
    sp_f = np.array(fixed.GetSpacing(), dtype=np.float64)
    if np.max(np.abs(sp_r - sp_f)) > tol:
        raise ValueError(f"ROI 与 T2 spacing 不一致: {sp_r} vs {sp_f}")
    d_o = np.array(roi.GetDirection(), dtype=np.float64) - np.array(fixed.GetDirection(), dtype=np.float64)
    if np.max(np.abs(d_o)) > tol:
        raise ValueError("ROI 与 T2 方向矩阵 direction 不一致（超过容差）")
    p_o = np.array(roi.GetOrigin(), dtype=np.float64) - np.array(fixed.GetOrigin(), dtype=np.float64)
    if np.max(np.abs(p_o)) > tol:
        raise ValueError(f"ROI 与 T2 原点 origin 不一致（超过容差）: max|Δ|={np.max(np.abs(p_o))}")


def copy_roi_to_output(roi_path: Path, fixed: sitk.Image, out_path: Path) -> None:
    roi = sitk.ReadImage(str(roi_path))
    validate_roi_matches_fixed(roi, fixed)
    sitk.WriteImage(roi, str(out_path))


def save_checkerboard_png(
    fixed: sitk.Image,
    moving_reg: sitk.Image,
    out_png: Path,
    *,
    checker_size_vox: int = 12,
    slice_index: int | None = None,
) -> None:
    """
    取与 SimpleITK 一致的 **轴位一层**（数组轴 0 = z）画棋盘格。
    fixed / moving_reg 须已在同一网格上。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fa = _sitk_arr(_cast_float(fixed))
    ma = _sitk_arr(_cast_float(moving_reg))
    assert fa.shape == ma.shape
    if fa.ndim != 3:
        raise ValueError(f"仅支持 3D，当前 shape={fa.shape}")

    nz, _ny, _nx = fa.shape
    zi = slice_index if slice_index is not None else nz // 2
    zi = int(np.clip(zi, 0, nz - 1))

    a = np.rot90(fa[zi, :, :])
    b = np.rot90(ma[zi, :, :])

    def norm(x: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(x, [1.0, 99.0])
        if hi <= lo:
            hi = lo + 1.0
        y = (x - lo) / (hi - lo)
        return np.clip(y, 0.0, 1.0)

    a = norm(a)
    b = norm(b)
    h, w = a.shape
    cb = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            use_a = ((i // checker_size_vox) + (j // checker_size_vox)) % 2 == 0
            v = a[i, j] if use_a else b[i, j]
            cb[i, j, :] = v

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cb[:, :, 0], cmap="gray", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_title(f"Checkerboard  z={zi}/{nz - 1}  (T2 vs registered, SimpleITK ZYX)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_side_by_side_png(
    fixed: sitk.Image,
    moving_reg: sitk.Image,
    out_png: Path,
    *,
    slice_index: int | None = None,
) -> None:
    """T2 与配准后 moving 同一层并排，便于直观看解剖是否重合。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fa = _sitk_arr(_cast_float(fixed))
    ma = _sitk_arr(_cast_float(moving_reg))
    assert fa.shape == ma.shape
    nz, _, _ = fa.shape
    zi = slice_index if slice_index is not None else nz // 2
    zi = int(np.clip(zi, 0, nz - 1))

    def norm(x: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(x, [1.0, 99.0])
        if hi <= lo:
            hi = lo + 1.0
        y = (x - lo) / (hi - lo)
        return np.clip(y, 0.0, 1.0)

    a = norm(np.rot90(fa[zi, :, :]))
    b = norm(np.rot90(ma[zi, :, :]))
    h, w = a.shape
    gap = max(4, w // 80)
    combo = np.ones((h, w * 2 + gap), dtype=np.float32)
    combo[:, :w] = a
    combo[:, w + gap :] = b

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.imshow(combo, cmap="gray", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_title(f"T2 (L) | registered (R)  z={zi}/{nz - 1}  (SimpleITK ZYX)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_case(
    case_dir: Path,
    out_dir: Path,
    *,
    moving_modalities: tuple[str, ...] = DEFAULT_MOVING_MODALITIES,
    iterations: int = 400,
    hist_match: bool = True,
    roi_path: Path | None = None,
    save_transforms: bool = True,
    checkerboard_slice_index: int | None = None,
    affine_refine: bool = False,
    write_qc: bool = True,
) -> list[QCResult]:
    case_dir = case_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(exist_ok=True)

    stems = discover_nifti_stems(case_dir)
    if "T2" not in stems:
        raise FileNotFoundError(f"病例目录缺少 T2: {case_dir}")

    t2_path = stems["T2"]
    fixed = sitk.ReadImage(str(t2_path))

    stem = t2_path.name.removesuffix(".nii.gz").removesuffix(".nii")
    sid = stem.rsplit("_", 1)[0]

    results: list[QCResult] = []
    report: dict[str, Any] = {"case_dir": str(case_dir), "subject_id": sid, "modalities": {}}

    if roi_path is not None:
        roi_out = out_dir / f"{sid}_roi_T2.nii.gz"
        copy_roi_to_output(Path(roi_path), fixed, roi_out)
        report["roi_copied"] = str(roi_out)

    mask = _otsu_foreground_mask(_cast_float(fixed))

    for mod in moving_modalities:
        if mod not in stems:
            report["modalities"][mod] = {"status": "skipped", "reason": "file not found"}
            continue
        mov_path = stems[mod]
        moving = sitk.ReadImage(str(mov_path))

        tfm_init = _initial_baseline_transform(fixed, moving, hist_match)
        moving_init = resample_moving_to_fixed(moving, fixed, tfm_init)
        ncc_i = _ncc_on_fixed_grid(fixed, moving_init, mask)

        tfm = _register_to_t2_transform(
            fixed,
            moving,
            number_of_iterations=iterations,
            hist_match=hist_match,
            affine_refine=affine_refine,
        )
        moving_reg = resample_moving_to_fixed(moving, fixed, tfm)
        ncc_r = _ncc_on_fixed_grid(fixed, moving_reg, mask)

        out_vol = out_dir / f"{sid}_{mod}_reg2T2.nii.gz"
        sitk.WriteImage(moving_reg, str(out_vol))

        cb_path_str = ""
        sbs_path_str = ""
        if write_qc:
            cb_path = qc_dir / f"checkerboard_{mod}.png"
            save_checkerboard_png(
                fixed, moving_reg, cb_path, slice_index=checkerboard_slice_index
            )
            sbs_path = qc_dir / f"side_by_side_{mod}.png"
            save_side_by_side_png(fixed, moving_reg, sbs_path, slice_index=checkerboard_slice_index)
            cb_path_str = str(cb_path)
            sbs_path_str = str(sbs_path)

        dlt = ncc_r - ncc_i
        results.append(QCResult(mod, ncc_i, ncc_r, dlt))
        mod_report: dict[str, Any] = {
            "status": "ok",
            "affine_refine": affine_refine,
            "ncc_initial_similarity_geometry": ncc_i,
            "ncc_after_registration": ncc_r,
            "delta_ncc": dlt,
            "registered_volume": str(out_vol),
            "checkerboard": cb_path_str,
            "side_by_side": sbs_path_str,
        }
        if save_transforms:
            mod_report["transform_files"] = save_registration_transforms(
                tfm, out_dir / f"{sid}_{mod}_to_T2"
            )
        report["modalities"][mod] = mod_report

    with open(qc_dir / "qc_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return results


def _ncc_delta_hint(delta: float) -> str:
    """多模态下 NCC 仅为粗指标，按 Δ 符号给可读提示。"""
    if delta > 0.01:
        return "Δ>0：在 Otsu 前景内线相关相对「相似性几何初值」有提升（仍须结合并排图）。"
    if delta < -0.01:
        return (
            "Δ<0：线相关在掩膜内下降，并不等价于解剖一定更差，但建议重点看 "
            "side_by_side；若解剖错位明显，可试 --no-hist-match 或在 Slicer 中重做。"
        )
    return "Δ接近 0：粗指标变化小，请以并排/棋盘格目视为准。"


def main() -> None:
    default_case = Path(__file__).resolve().parents[1] / "MRI-DN" / "MRI" / "Internal" / "004"

    parser = argparse.ArgumentParser(description="将 DWI/ADC 等配准到 T2 网格并输出 QC")
    parser.add_argument("case_dir", nargs="?", type=Path, default=default_case)
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="输出目录（将写入 *_reg2T2.nii.gz、qc/ 等）",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=",".join(DEFAULT_MOVING_MODALITIES),
        help="逗号分隔 moving 模态，默认 DWI,ADC",
    )
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument(
        "--no-hist-match",
        action="store_true",
        help="关闭配准前直方图匹配（默认开启，利于 T2–DWI 多模态）",
    )
    parser.add_argument("--roi", type=Path, default=None, help="已在 T2 网格上的 ROI 掩膜 NIfTI，校验后复制到输出目录")
    parser.add_argument("--no-save-transform", action="store_true")
    parser.add_argument(
        "--affine-refine",
        action="store_true",
        help="在相似性之后追加仿射精化（默认关闭；多模态上易变差，仅实验用）",
    )
    parser.add_argument(
        "--checker-slice",
        type=int,
        default=None,
        help="QC 层索引（SimpleITK ZYX 轴 0）；棋盘格与并排图共用，默认中间层",
    )
    parser.add_argument(
        "--no-qc-plots",
        action="store_true",
        help="不生成棋盘格/并排 PNG（仍写 qc_metrics.json），批量时更快",
    )
    args = parser.parse_args()

    mods = tuple(m.strip() for m in args.modalities.split(",") if m.strip())
    try:
        results = run_case(
            args.case_dir,
            args.out_dir,
            moving_modalities=mods,
            iterations=args.iterations,
            hist_match=not args.no_hist_match,
            roi_path=args.roi,
            save_transforms=not args.no_save_transform,
            checkerboard_slice_index=args.checker_slice,
            affine_refine=args.affine_refine,
            write_qc=not args.no_qc_plots,
        )
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    for r in results:
        hint = _ncc_delta_hint(r.delta_ncc)
        print(
            f"{r.modality}: NCC 初值={r.ncc_initial:.4f} 配准后={r.ncc_registered:.4f} "
            f"Δ={r.delta_ncc:+.4f}。{hint}"
        )
    print(f"QC 已写入: {args.out_dir / 'qc'}")


if __name__ == "__main__":
    main()
