"""
AxialGate3D 深度分支：配准后的 T2、DWI、ADC 病灶局部体，分层 K 折二分类。

输入按病灶 ROI 的包围盒裁剪，并重采样到 64×64×12。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from mri_case_reader import discover_nifti_stems, infer_subject_id
from radiomics_pipeline.cohort_qc import filter_subject_ids
from radiomics_pipeline.id_list import load_subject_ids
from radiomics_pipeline.preprocess import build_normalized_stack
from radiomics_pipeline.roi_mask import find_lesion_roi_path, lesion_mask_numpy
from radiomics_pipeline.run_pipeline import load_label_csv

# 固定 3 通道：T2、DWI、ADC（无 ADC 时第三通道为 0，避免 batch stack 维度不一致）
N_INPUT_CHANNELS = 3


def _ensure_channels(vol: np.ndarray, n_ch: int = N_INPUT_CHANNELS) -> np.ndarray:
    """统一通道数；部分病例无 ADC 时 stack 仅 2 通道。"""
    c = vol.shape[0]
    if c == n_ch:
        return vol.astype(np.float32, copy=False)
    if c > n_ch:
        return vol[:n_ch].astype(np.float32)
    out = np.zeros((n_ch, *vol.shape[1:]), dtype=np.float32)
    out[:c] = vol
    return out


def _cache_name(sid: str, spatial: tuple[int, int, int], *, mask_tag: str) -> str:
    z, y, x = spatial
    return f"{sid}_{mask_tag}_c{N_INPUT_CHANNELS}_{z}x{y}x{x}.npy"


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset

        return torch, nn, DataLoader, Dataset
    except ImportError as e:
        raise SystemExit(
            "未安装 PyTorch。请执行：pip install torch\n"
            "CPU 版：pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from e


def _list_registered_cases(registered_root: Path) -> list[str]:
    ids = []
    for p in sorted(registered_root.iterdir()):
        if p.is_dir() and p.name.isdigit():
            ids.append(p.name)
    return ids


def _load_volume_tensor(
    case_dir: Path,
    spatial: tuple[int, int, int],
    *,
    roi_mask_root: Path | None = None,
    bbox_pad: int = 4,
    lesion_only: bool = False,
) -> np.ndarray:
    """返回 ``(C, Z, Y, X)`` float32；按 Otsu 或病灶 ROI 的 bbox 裁剪。"""
    stems = discover_nifti_stems(case_dir)
    sid = infer_subject_id(case_dir)
    if "T2" not in stems:
        raise FileNotFoundError(f"无 T2: {case_dir}")
    t2_path = stems["T2"]
    dwi_reg = case_dir / f"{sid}_DWI_reg2T2.nii.gz"
    if not dwi_reg.is_file():
        raise FileNotFoundError(f"无 DWI_reg2T2: {dwi_reg}")
    adc_reg = case_dir / f"{sid}_ADC_reg2T2.nii.gz"
    use_adc = adc_reg.is_file()

    roi_zyx = None
    if roi_mask_root is not None:
        roi_path = find_lesion_roi_path(
            roi_mask_root / sid, sid, lesion_only=lesion_only
        )
        if roi_path is None:
            hint = "T2_T" if lesion_only else "T2_T/T2_P"
            raise FileNotFoundError(f"无 {hint} ROI: {roi_mask_root / sid}")
        roi_zyx = lesion_mask_numpy(roi_path)

    stack, mask, _ = build_normalized_stack(
        t2_path,
        dwi_reg,
        adc_reg if use_adc else None,
        use_adc=use_adc,
        roi_mask_zyx=roi_zyx,
    )
    m = mask.astype(bool)
    if m.sum() < 10:
        vol = stack
    else:
        zz, yy, xx = np.where(m)
        pad = bbox_pad
        z0, z1 = max(0, zz.min() - pad), min(stack.shape[1], zz.max() + pad + 1)
        y0, y1 = max(0, yy.min() - pad), min(stack.shape[2], yy.max() + pad + 1)
        x0, x1 = max(0, xx.min() - pad), min(stack.shape[3], xx.max() + pad + 1)
        vol = stack[:, z0:z1, y0:y1, x0:x1]

    tgt = np.array(spatial, dtype=np.float64)
    src = np.array(vol.shape[1:], dtype=np.float64)
    factors = tgt / np.maximum(src, 1.0)
    out = np.zeros((vol.shape[0], *spatial), dtype=np.float32)
    for c in range(vol.shape[0]):
        out[c] = zoom(vol[c], factors, order=1).astype(np.float32)
    return _ensure_channels(out)


class _CaseDataset:
    def __init__(self, tensors: list[np.ndarray], labels: list[int]):
        self.tensors = tensors
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        torch, _, _, _ = _require_torch()
        x = torch.from_numpy(self.tensors[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def _build_axialgate3d(in_ch: int, n_classes: int):
    """
    AxialGate3D：强化各向异性 3D MRI 的跨切片/跨轴建模。

    针对输入为体数据（默认 64×64×12）且层厚不均的问题：
    1. 分解卷积 stem：先 (3,3,1) 面内，再 (1,1,3) 轴间，避免各向同性大核糊掉薄层
    2. 三轴向注意力（D/H/W）：显式建模切片间与面内长程依赖
    3. 轻量 3D 相对位置偏置（可学习轴坐标嵌入）
    4. 保留 SE + 空间注意力池化，便于可视化与组学晚期融合

    ``forward_with_attention`` 返回分类 logits、空间注意力图和通道权重。
    """
    torch, nn, _, _ = _require_torch()
    F = __import__("torch.nn.functional", fromlist=["functional"])

    class _SE3d(nn.Module):
        def __init__(self, ch: int, reduction: int = 8):
            super().__init__()
            mid = max(ch // reduction, 4)
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(ch, mid, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(mid, ch, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x):
            w = self.fc(x).view(x.size(0), x.size(1), 1, 1, 1)
            return x * w, w.squeeze(-1).squeeze(-1).squeeze(-1)

    class _AnisoConv(nn.Module):
        """面内 (3,3,1) + 轴间 (1,1,3) 分解，强化薄层方向信息。"""

        def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
            super().__init__()
            self.pw = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm3d(out_ch)
            self.plane = nn.Conv3d(
                out_ch,
                out_ch,
                kernel_size=(3, 3, 1),
                stride=(stride, stride, 1),
                padding=(1, 1, 0),
                groups=out_ch,
                bias=False,
            )
            self.depth = nn.Conv3d(
                out_ch,
                out_ch,
                kernel_size=(1, 1, 3),
                stride=(1, 1, stride),
                padding=(0, 0, 1),
                groups=out_ch,
                bias=False,
            )
            self.bn1 = nn.BatchNorm3d(out_ch)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.act(self.bn0(self.pw(x)))
            x = self.plane(x)
            x = self.depth(x)
            return self.act(self.bn1(x))

    class _AxisAttn(nn.Module):
        """沿单一空间轴的轻量自注意力（轴向注意力）。"""

        def __init__(self, ch: int, axis: int, max_len: int = 64):
            super().__init__()
            assert axis in (2, 3, 4)  # D,H,W in B,C,D,H,W
            self.axis = axis
            self.qkv = nn.Conv1d(ch, ch * 3, kernel_size=1, bias=False)
            self.proj = nn.Conv1d(ch, ch, kernel_size=1, bias=False)
            self.pos = nn.Parameter(torch.zeros(1, ch, max_len))
            nn.init.trunc_normal_(self.pos, std=0.02)
            self.scale = (ch) ** -0.5
            self.gamma = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            # move target axis to last, pool other spatial dims
            if self.axis == 2:
                # B,C,D,H,W → mean H,W → B,C,D
                pooled = x.mean(dim=(3, 4))
            elif self.axis == 3:
                pooled = x.mean(dim=(2, 4))
            else:
                pooled = x.mean(dim=(2, 3))
            b, c, l = pooled.shape
            pos = self.pos[:, :, :l]
            qkv = self.qkv(pooled + pos)
            q, k, v = qkv.chunk(3, dim=1)
            attn = torch.softmax(torch.einsum("bcl,bcm->blm", q, k) * self.scale, dim=-1)
            out = torch.einsum("blm,bcm->bcl", attn, v)
            out = self.proj(out)
            # broadcast back
            if self.axis == 2:
                out = out.unsqueeze(-1).unsqueeze(-1)
            elif self.axis == 3:
                out = out.unsqueeze(2).unsqueeze(-1)
            else:
                out = out.unsqueeze(2).unsqueeze(3)
            return x + self.gamma * out

    class _TriAxialAttn(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            self.d = _AxisAttn(ch, axis=2)
            self.h = _AxisAttn(ch, axis=3)
            self.w = _AxisAttn(ch, axis=4)

        def forward(self, x):
            return self.w(self.h(self.d(x)))

    class _AnisoSEBlock(nn.Module):
        def __init__(self, in_planes: int, planes: int, stride: int = 1):
            super().__init__()
            self.conv = _AnisoConv(in_planes, planes, stride=stride)
            self.se = _SE3d(planes)
            self.axial = _TriAxialAttn(planes)
            self.relu = nn.ReLU(inplace=True)
            downsample = None
            if stride != 1 or in_planes != planes:
                downsample = nn.Sequential(
                    nn.Conv3d(in_planes, planes, 1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes),
                )
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.conv(x)
            out, _ = self.se(out)
            out = self.axial(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.relu(out + identity)

    class _SpatialAttnPool3d(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            mid = max(ch // 4, 8)
            self.proj = nn.Sequential(
                nn.Conv3d(ch, mid, kernel_size=1, bias=False),
                nn.BatchNorm3d(mid),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid, 1, kernel_size=1, bias=True),
            )

        def forward(self, x):
            logits = self.proj(x)
            b, _, d, h, w = logits.shape
            attn = F.softmax(logits.view(b, 1, -1), dim=-1).view(b, 1, d, h, w)
            pooled = (x * attn).sum(dim=(2, 3, 4))
            return pooled, attn

    class _AxialGate3D(nn.Module):
        def __init__(self):
            super().__init__()
            # 分解 stem：先降采样面内，再混薄层轴
            self.stem = nn.Sequential(
                nn.Conv3d(
                    in_ch,
                    32,
                    kernel_size=(7, 7, 3),
                    stride=(2, 2, 1),
                    padding=(3, 3, 1),
                    bias=False,
                ),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    32,
                    32,
                    kernel_size=(1, 1, 3),
                    stride=(1, 1, 1),
                    padding=(0, 0, 1),
                    bias=False,
                ),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0)),
            )
            self.layer1 = nn.Sequential(
                _AnisoSEBlock(32, 48),
                _AnisoSEBlock(48, 48),
            )
            self.layer2 = nn.Sequential(
                _AnisoSEBlock(48, 80, stride=2),
                _AnisoSEBlock(80, 80),
            )
            self.layer3 = nn.Sequential(
                _AnisoSEBlock(80, 96),
            )
            self.attn_pool = _SpatialAttnPool3d(96)
            self.channel_se = _SE3d(96, reduction=8)
            self.dropout = nn.Dropout(0.45)
            self.fc = nn.Linear(96, n_classes)

        def forward_with_attention(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x, ch_w = self.channel_se(x)
            pooled, attn = self.attn_pool(x)
            logits = self.fc(self.dropout(pooled))
            return logits, attn, ch_w

        def forward(self, x):
            logits, _, _ = self.forward_with_attention(x)
            return logits

    return _AxialGate3D()


def build_3d_model(name: str, in_ch: int, n_classes: int):
    if name != "axialgate":
        raise ValueError("只保留 AxialGate3D，请使用 axialgate")
    return _build_axialgate3d(in_ch, n_classes)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _save_checkpoint(path: Path, model, *, meta: dict) -> None:
    torch, _, _, _ = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "in_channels": N_INPUT_CHANNELS,
        "n_classes": 2,
        "model_name": meta.get("model_name", "axialgate"),
        **meta,
    }
    torch.save(payload, path)
    _log(f"  已保存权重: {path}")


def _train_eval_fold(
    train_x: list[np.ndarray],
    train_y: list[int],
    val_x: list[np.ndarray],
    val_y: list[int],
    *,
    fold: int,
    n_splits: int,
    device,
    epochs: int,
    batch_size: int,
    lr: float,
    checkpoint_path: Path | None = None,
    checkpoint_meta: dict | None = None,
    model_name: str = "axialgate",
    early_stop_patience: int = 0,
    early_stop_min_epochs: int = 20,
) -> dict[str, float]:
    torch, nn, DataLoader, Dataset = _require_torch()

    class DS(Dataset):
        def __init__(self, xs, ys):
            self.xs, self.ys = xs, ys

        def __len__(self):
            return len(self.ys)

        def __getitem__(self, i):
            return torch.from_numpy(self.xs[i]), torch.tensor(self.ys[i], dtype=torch.long)

    in_ch = N_INPUT_CHANNELS
    if any(x.shape[0] != in_ch for x in train_x):
        raise ValueError(
            f"训练样本通道不一致，期望 {in_ch}，"
            f"实际: {sorted({x.shape[0] for x in train_x})}"
        )
    model = build_3d_model(model_name, in_ch, 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    counts = np.bincount(np.array(train_y, dtype=np.int64), minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (2.0 * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32, device=device)
    )

    train_loader = DataLoader(
        DS(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # 早停用训练折内 holdout（不碰真正验证折，避免乐观偏差）
    fit_x, fit_y = train_x, train_y
    mon_x, mon_y = [], []
    if early_stop_patience > 0 and len(train_y) >= 20 and len(set(train_y)) >= 2:
        sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold + 17)
        idx = np.arange(len(train_y))
        fit_idx, mon_idx = next(sk.split(idx, train_y))
        fit_x = [train_x[i] for i in fit_idx]
        fit_y = [train_y[i] for i in fit_idx]
        mon_x = [train_x[i] for i in mon_idx]
        mon_y = [train_y[i] for i in mon_idx]
        train_loader = DataLoader(
            DS(fit_x, fit_y),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        counts = np.bincount(np.array(fit_y, dtype=np.int64), minlength=2).astype(
            np.float64
        )
        counts = np.maximum(counts, 1.0)
        w = counts.sum() / (2.0 * counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32, device=device)
        )

    n_batches = max(1, len(train_loader))
    best_state = None
    best_auc = -1.0
    best_ep = 0
    stale = 0

    def _monitor_auc() -> float:
        model.eval()
        probs_e: list[float] = []
        with torch.no_grad():
            for xb, _yb in zip(mon_x, mon_y):
                t = torch.from_numpy(xb).unsqueeze(0).to(device)
                p = torch.softmax(model(t), dim=1)[0, 1].item()
                probs_e.append(p)
        if len(set(mon_y)) < 2:
            return float("nan")
        return float(roc_auc_score(mon_y, probs_e))

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for bi, (xb, yb) in enumerate(train_loader, start=1):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            running_loss += float(loss.item()) * len(yb)
            n_seen += len(yb)
            if bi == 1 or bi == n_batches or bi % max(1, n_batches // 5) == 0:
                _log(
                    f"  [Fold {fold}/{n_splits}] Epoch {ep}/{epochs}  "
                    f"batch {bi}/{n_batches}  loss={loss.item():.4f}"
                )
        mean_loss = running_loss / max(n_seen, 1)
        _log(
            f"  [Fold {fold}/{n_splits}] Epoch {ep}/{epochs} 完成  "
            f"mean_loss={mean_loss:.4f}"
        )

        if early_stop_patience > 0 and mon_x:
            mauc = _monitor_auc()
            _log(
                f"  [Fold {fold}/{n_splits}] Epoch {ep}/{epochs}  "
                f"monitor_auc={mauc:.4f}"
            )
            # 前 min_epochs 只记录最佳，不累计 patience（避免噪声尖峰导致过早停）
            if not np.isnan(mauc) and mauc > best_auc + 1e-4:
                best_auc = mauc
                best_ep = ep
                stale = 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            elif ep >= early_stop_min_epochs:
                stale += 1
                if stale >= early_stop_patience:
                    _log(
                        f"  [Fold {fold}/{n_splits}] early stop @ ep={ep}  "
                        f"best_ep={best_ep} monitor_auc={best_auc:.4f}"
                    )
                    break

    if best_state is not None and best_ep > 0:
        retrain_ep = max(int(best_ep), int(early_stop_min_epochs))
        # 用全训练折按最佳轮数重训，充分利用小样本数据
        _log(
            f"  [Fold {fold}/{n_splits}] 全训练折重训 ep={retrain_ep} "
            f"(best_ep={best_ep}, monitor_auc={best_auc:.4f})"
        )
        best_ep = retrain_ep
        model = build_3d_model(model_name, in_ch, 2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        counts = np.bincount(np.array(train_y, dtype=np.int64), minlength=2).astype(
            np.float64
        )
        counts = np.maximum(counts, 1.0)
        w = counts.sum() / (2.0 * counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32, device=device)
        )
        full_loader = DataLoader(
            DS(train_x, train_y),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        for ep in range(1, best_ep + 1):
            model.train()
            for xb, yb in full_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                opt.step()
            if ep == 1 or ep == best_ep or ep % max(1, best_ep // 5) == 0:
                _log(f"  [Fold {fold}/{n_splits}] retrain ep {ep}/{best_ep}")

    _log(f"  [Fold {fold}/{n_splits}] 验证集推理 ({len(val_x)} 例)...")
    model.eval()
    probs: list[float] = []
    preds: list[int] = []
    with torch.no_grad():
        for xb, yb in zip(val_x, val_y):
            t = torch.from_numpy(xb).unsqueeze(0).to(device)
            logit = model(t)
            p = torch.softmax(logit, dim=1)[0, 1].item()
            probs.append(p)
            preds.append(int(p >= 0.5))

    if len(set(val_y)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(val_y, probs))

    if checkpoint_path is not None:
        meta = dict(checkpoint_meta or {})
        meta.update(
            {
                "fold": fold,
                "n_splits": n_splits,
                "epochs": epochs,
                "early_stop_patience": early_stop_patience,
                "best_epoch": best_ep if best_state is not None else epochs,
                "model_name": model_name,
                "roc_auc": auc,
                "accuracy": float(accuracy_score(val_y, preds)),
                "f1": float(f1_score(val_y, preds, zero_division=0)),
            }
        )
        _save_checkpoint(checkpoint_path, model, meta=meta)

    return {
        "roc_auc": auc,
        "accuracy": float(accuracy_score(val_y, preds)),
        "f1": float(f1_score(val_y, preds, zero_division=0)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_reg = root.parent / "MRI-DN" / "MRI" / "Internal_registered"
    default_labels = root.parent / "MRI-DN" / "MRI" / "Internal" / "PTEN label.csv"

    ap = argparse.ArgumentParser(description="AxialGate3D 深度分支，分层 K 折二分类")
    ap.add_argument("--registered-root", type=Path, default=default_reg)
    ap.add_argument("--labels-csv", type=Path, default=default_labels)
    ap.add_argument("--output-dir", type=Path, default=root / "deep_out_registered")
    ap.add_argument("--k-fold", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--spatial",
        type=int,
        nargs=3,
        default=(64, 64, 12),
        metavar=("Z", "Y", "X"),
        help="重采样体素尺寸",
    )
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--cache-npy", type=Path, default=None, help="预处理体缓存目录（可选）")
    ap.add_argument(
        "--roi-mask-root",
        type=Path,
        default=None,
        help="Internal 原始目录（含 *_T2_T / *_T2_P）；启用后按病灶 ROI bbox 裁剪",
    )
    ap.add_argument(
        "--lesion-only",
        action="store_true",
        help="仅 *_T2_T 纯病灶掩膜（不用 T2_P）",
    )
    ap.add_argument("--min-roi-voxels", type=int, default=0)
    ap.add_argument("--exclude-ids", type=str, default="")
    ap.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="仅训练列表中的 subject_id",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="训练设备；默认 cuda，无 GPU 时需显式 --device cpu",
    )
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="每折权重保存目录（默认 <output-dir>/checkpoints）",
    )
    ap.add_argument(
        "--no-save-checkpoints",
        action="store_true",
        help="不保存 .pt 权重（不推荐）",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="axialgate",
        choices=("axialgate",),
        help="深度分支只保留 AxialGate3D",
    )
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=15,
        help="训练折内 holdout 的早停耐心轮数",
    )
    ap.add_argument(
        "--early-stop-min-epochs",
        type=int,
        default=25,
        help="早停生效前的最少训练轮数；重训轮数也不低于此值",
    )
    args = ap.parse_args()

    torch, _, _, _ = _require_torch()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "未检测到 CUDA。请检查 nvidia-smi / 驱动，或改用 --device cpu"
            )
        device = torch.device("cuda")
        _log(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        _log("警告: 使用 CPU 训练，速度很慢，建议在有 GPU 的机器上跑")
    roi_root = args.roi_mask_root.expanduser().resolve() if args.roi_mask_root else None
    if args.lesion_only and not roi_root:
        raise SystemExit("--lesion-only 需同时指定 --roi-mask-root")
    if roi_root:
        mask_tag = "lesion_t2t" if args.lesion_only else "lesion"
    else:
        mask_tag = "otsu"
    exclude = {x.strip().zfill(3) for x in args.exclude_ids.split(",") if x.strip()}

    labels = load_label_csv(args.labels_csv)
    label_map = dict(zip(labels["subject_id"], labels["label"].astype(int)))

    case_ids = _list_registered_cases(args.registered_root)
    if args.max_cases:
        case_ids = case_ids[: args.max_cases]

    tensors: list[np.ndarray] = []
    ys: list[int] = []
    sids: list[str] = []
    skipped: list[dict] = []

    spatial = tuple(args.spatial)
    if args.cache_npy:
        args.cache_npy.mkdir(parents=True, exist_ok=True)

    load_ids = [sid for sid in case_ids if sid in label_map]
    if roi_root and (args.min_roi_voxels > 0 or exclude or args.lesion_only):
        load_ids, qc_drop = filter_subject_ids(
            load_ids,
            registered_root=args.registered_root,
            roi_mask_root=roi_root,
            min_roi_voxels=args.min_roi_voxels,
            exclude_ids=exclude or None,
            lesion_only=args.lesion_only,
        )
        if qc_drop:
            qc_path = args.output_dir / "qc_excluded.json"
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with open(qc_path, "w", encoding="utf-8") as f:
                json.dump({"excluded": qc_drop, "n_kept": len(load_ids)}, f, indent=2)
            _log(f"质控剔除 {len(qc_drop)} 例 -> {qc_path}")
    if args.ids_file is not None:
        allow = set(load_subject_ids(args.ids_file))
        load_ids = [s for s in load_ids if s in allow]
        _log(f"--ids-file: 仅训练 {len(load_ids)} 例")
    n_load = len(load_ids)
    _log(
        f"设备: {device}  裁剪: {mask_tag}  空间尺寸: {spatial}  "
        f"待加载: {n_load} 例  epochs={args.epochs}  batch={args.batch_size}"
    )
    if roi_root:
        mode = "纯病灶 T2_T" if args.lesion_only else "T2_T 优先"
        _log(f"ROI: {roi_root}  ({mode})")

    for i, sid in enumerate(load_ids, start=1):
        case_dir = args.registered_root / sid
        cache_path = (
            args.cache_npy / _cache_name(sid, spatial, mask_tag=mask_tag)
            if args.cache_npy
            else None
        )
        try:
            if cache_path and cache_path.is_file():
                vol = _ensure_channels(np.load(cache_path))
                _log(f"[加载 {i}/{n_load}] {sid}  cache")
            else:
                _log(f"[加载 {i}/{n_load}] {sid}  预处理中...")
                vol = _load_volume_tensor(
                    case_dir,
                    spatial,
                    roi_mask_root=roi_root,
                    lesion_only=args.lesion_only,
                )
                if cache_path:
                    np.save(cache_path, vol)
                    _log(f"[加载 {i}/{n_load}] {sid}  已缓存")
            tensors.append(vol)
            ys.append(int(label_map[sid]))
            sids.append(sid)
        except Exception as e:
            skipped.append({"id": sid, "error": str(e)})
            _log(f"[加载 {i}/{n_load}] [skip] {sid}: {e}")

    if len(tensors) < 10:
        raise SystemExit(f"有效样本过少: {len(tensors)}")
    n_pos = int(np.sum(y_arr := np.array(ys, dtype=np.int64)))
    _log(f"加载完成: {len(tensors)} 例  label=1: {n_pos}  label=0: {len(ys) - n_pos}  跳过: {len(skipped)}")

    X_idx = np.arange(len(tensors))
    min_class = int(np.bincount(y_arr).min())
    n_splits = min(args.k_fold, max(2, min_class))

    ckpt_dir = (
        None
        if args.no_save_checkpoints
        else (args.checkpoint_dir or args.output_dir / "checkpoints")
    )
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        _log(f"权重目录: {ckpt_dir.resolve()}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: list[dict] = []
    for fold, (tr, va) in enumerate(cv.split(X_idx, y_arr), start=1):
        _log(f"\n======== Fold {fold}/{n_splits}  train={len(tr)}  val={len(va)} ========")
        tr_x = [tensors[i] for i in tr]
        tr_y = [ys[i] for i in tr]
        va_x = [tensors[i] for i in va]
        va_y = [ys[i] for i in va]
        tr_sids = [sids[i] for i in tr]
        va_sids = [sids[i] for i in va]
        ckpt_path = (
            ckpt_dir / f"fold_{fold:02d}.pt" if ckpt_dir is not None else None
        )
        m = _train_eval_fold(
            tr_x,
            tr_y,
            va_x,
            va_y,
            fold=fold,
            n_splits=n_splits,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_path=ckpt_path,
            checkpoint_meta={
                "mask_mode": mask_tag,
                "spatial": list(spatial),
                "train_subject_ids": tr_sids,
                "val_subject_ids": va_sids,
                "device": str(device),
            },
            model_name=args.model,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_epochs=args.early_stop_min_epochs,
        )
        m["fold"] = fold - 1
        if ckpt_path is not None:
            m["checkpoint"] = str(ckpt_path)
        fold_metrics.append(m)
        _log(
            f"  [Fold {fold}/{n_splits}] 验证  "
            f"AUC={m['roc_auc']:.3f}  Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}"
        )

    summary = {
        "model": args.model,
        "mask_mode": mask_tag,
        "roi_mask_root": str(roi_root) if roi_root else None,
        "device": str(device),
        "spatial": list(spatial),
        "n_samples": len(tensors),
        "n_splits": n_splits,
        "epochs": args.epochs,
        "skipped": skipped,
        "roc_auc_mean": float(np.nanmean([m["roc_auc"] for m in fold_metrics])),
        "roc_auc_std": float(np.nanstd([m["roc_auc"] for m in fold_metrics])),
        "accuracy_mean": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "f1_mean": float(np.mean([m["f1"] for m in fold_metrics])),
        "fold_metrics": fold_metrics,
        "checkpoints_dir": str(ckpt_dir) if ckpt_dir is not None else None,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "deep_cv_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _log(
        f"\nAxialGate3D 样本数={summary['n_samples']}  "
        f"ROC-AUC={summary['roc_auc_mean']:.3f}±{summary['roc_auc_std']:.3f}  "
        f"Acc={summary['accuracy_mean']:.3f}  F1={summary['f1_mean']:.3f}"
    )
    _log(f"结果: {out_path}")


if __name__ == "__main__":
    main()
