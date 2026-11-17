"""
读取 MRI-DN 内部数据目录下单个病例文件夹中的 NIfTI 序列。

约定（以 ``MRI/Internal/004`` 为例）：
  - ``{id}_T2.nii``：T2 加权
  - ``{id}_DWI.nii``：DWI
  - ``{id}_ADC.nii``：ADC（部分病例可能缺失）
  - ``{id}_T2_P.nii.gz``、``{id}_T2_T.nii.gz``：附加衍生/标注体（具体含义以课题说明为准）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np

# 文件名中的模态后缀（不含扩展名）；较长后缀必须排在前面，避免 ``*_T2_T`` 被误判为 ``T2``。
KNOWN_SUFFIXES = ("T2_T", "T2_P", "T2", "DWI", "ADC")


@dataclass
class Volume:
    """单个 NIfTI 体数据与仿射矩阵。"""

    data: np.ndarray
    affine: np.ndarray
    path: Path


@dataclass
class MRICaseVolumes:
    """一个病例目录下解析到的各序列；缺失的键不会出现或为 None。"""

    case_dir: Path
    subject_id: str
    volumes: dict[str, Volume] = field(default_factory=dict)

    def get(self, name: str) -> Volume | None:
        return self.volumes.get(name)


def _infer_subject_id(case_dir: Path, modality_to_path: Mapping[str, Path]) -> str:
    """从目录名或文件名推断病例 ID（如 ``004``、``410``）。"""
    name = case_dir.name
    if re.fullmatch(r"\d+", name):
        return name
    m = re.match(r"^(\d+)", name)
    if m:
        return m.group(1)
    for fp in modality_to_path.values():
        stem = fp.name.removesuffix(".nii.gz").removesuffix(".nii")
        for suf in KNOWN_SUFFIXES:
            if stem.endswith("_" + suf):
                return stem[: -(len(suf) + 1)]
    return name


def discover_nifti_stems(case_dir: str | Path) -> dict[str, Path]:
    """
    扫描病例目录，返回 ``模态后缀 -> 文件路径``。
    只识别 ``{id}_{T2|DWI|ADC|T2_P|T2_T}.nii(.gz)`` 形式。
    """
    case_dir = Path(case_dir).expanduser().resolve()
    if not case_dir.is_dir():
        raise FileNotFoundError(f"病例目录不存在: {case_dir}")

    mapping: dict[str, Path] = {}
    for p in sorted(case_dir.iterdir()):
        if not p.is_file():
            continue
        if not (p.suffix == ".gz" and p.name.endswith(".nii.gz") or p.suffix == ".nii"):
            continue
        stem = p.name.removesuffix(".nii.gz").removesuffix(".nii")
        for suf in KNOWN_SUFFIXES:
            if stem.endswith("_" + suf):
                mapping[suf] = p
                break
    return mapping


def load_volume(path: str | Path) -> Volume:
    """读取单个 NIfTI 文件为 ``Volume``。"""
    path = Path(path)
    img: Any = nib.load(str(path))
    data = np.asarray(img.dataobj)
    affine = np.asarray(img.affine, dtype=np.float64)
    return Volume(data=data, affine=affine, path=path.resolve())


def infer_subject_id(case_dir: str | Path) -> str:
    """仅从目录名与文件名推断病例 ID，不读入体数据。"""
    case_path = Path(case_dir).expanduser().resolve()
    stems = discover_nifti_stems(case_path)
    return _infer_subject_id(case_path, stems)


def load_mri_case(case_dir: str | Path) -> MRICaseVolumes:
    """
    加载病例目录下所有已知模态；缺失的序列不会出现在 ``volumes`` 中。

    若只需路径映射而不加载体素，请使用 ``discover_nifti_stems``。
    """
    stems = discover_nifti_stems(case_dir)
    case_path = Path(case_dir).expanduser().resolve()
    sid = _infer_subject_id(case_path, stems)
    out = MRICaseVolumes(case_dir=case_path, subject_id=sid, volumes={})
    for suffix, fp in stems.items():
        out.volumes[suffix] = load_volume(fp)
    return out


def summarize_case(case: MRICaseVolumes) -> str:
    """人类可读的体积形状与 dtype 摘要。"""
    lines = [f"病例目录: {case.case_dir}", f"推断 ID: {case.subject_id}", "序列:"]
    for name in sorted(case.volumes):
        v = case.volumes[name]
        lines.append(f"  {name}: shape={v.data.shape}, dtype={v.data.dtype}, file={v.path.name}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys

    default_example = Path(__file__).resolve().parents[1] / "MRI-DN" / "MRI" / "Internal" / "004"

    parser = argparse.ArgumentParser(description="读取 MRI-DN 单病例 NIfTI 目录")
    parser.add_argument(
        "case_dir",
        nargs="?",
        type=Path,
        default=default_example,
        help=f"病例文件夹路径（默认: {default_example}）",
    )
    args = parser.parse_args()
    try:
        case = load_mri_case(args.case_dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    print(summarize_case(case))
