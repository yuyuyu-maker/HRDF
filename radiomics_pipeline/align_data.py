"""批量将各病例配准到 T2（与 ``mri_register_to_t2`` 默认一致：相似性、无仿射精化）。"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from mri_case_reader import infer_subject_id

from mri_register_to_t2 import run_case


def case_subject_id(case_dir: Path) -> str:
    return infer_subject_id(case_dir)


def aligned_dwi_path(align_dir: Path, subject_id: str) -> Path:
    return align_dir / f"{subject_id}_DWI_reg2T2.nii.gz"


def needs_registration(case_dir: Path, align_dir: Path) -> bool:
    sid = case_subject_id(case_dir)
    return not aligned_dwi_path(align_dir, sid).is_file()


def ensure_case_registered(
    case_dir: Path,
    align_root: Path,
    *,
    iterations: int = 400,
    hist_match: bool = True,
    affine_refine: bool = False,
    save_transforms: bool = False,
) -> Path:
    """
    若 ``{align_root}/{id}_DWI_reg2T2.nii.gz`` 已存在则跳过；否则调用 ``run_case``。
    返回该病例的对齐输出目录 ``align_root``（多病例时每个病例一个子目录更清晰，
    此处约定 ``align_root`` 即为该病例目录，由调用方传入 ``align_root / id``）。
    """
    case_dir = case_dir.resolve()
    align_root = align_root.resolve()
    align_root.mkdir(parents=True, exist_ok=True)
    sid = case_subject_id(case_dir)
    if aligned_dwi_path(align_root, sid).is_file():
        return align_root
    run_case(
        case_dir,
        align_root,
        iterations=iterations,
        hist_match=hist_match,
        save_transforms=save_transforms,
        affine_refine=affine_refine,
        write_qc=False,
    )
    return align_root
