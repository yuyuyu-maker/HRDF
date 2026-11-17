"""读取/写入病例 ID 列表。"""

from __future__ import annotations

import json
from pathlib import Path


def load_subject_ids(path: Path | str) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"IDs 文件不存在: {path}")
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "selected_ids" in data:
            ids = data["selected_ids"]
        elif isinstance(data, list):
            ids = data
        else:
            raise ValueError(f"无法解析 JSON IDs: {path}")
    else:
        ids = [
            ln.strip()
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    out: list[str] = []
    for x in ids:
        s = str(x).strip()
        out.append(s.zfill(3) if s.isdigit() else s)
    return out


def save_subject_ids(ids: list[str], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
