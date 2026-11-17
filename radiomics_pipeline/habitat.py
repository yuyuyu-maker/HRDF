"""生境：强度 K-means 划区；支持 CH 选 K，并对子区提全套组学。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score


@dataclass
class HabitatPartition:
    """ROI 内生境划分结果。"""

    labels_zyx: np.ndarray  # int, -1=背景, 0..K-1=生境（已按排序键重排）
    selected_k: int
    ch_scores: dict[int, float]  # 各候选 K 的 CH 分数
    props: dict[int, float]  # 各生境体积占比
    order_key_means: dict[int, float]  # 排序键（默认 ADC）类内均值


def _empty_habitat_feature_dict(n_clusters: int, n_ch: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in range(n_clusters):
        out[f"habitat_prop_cluster{k}"] = 0.0
        for ch in range(n_ch):
            out[f"habitat_mean_ch{ch}_cluster{k}"] = 0.0
    out["habitat_inertia"] = 0.0
    out["habitat_selected_k"] = float(n_clusters)
    out["habitat_ch_score"] = 0.0
    return out


def _fit_predict_full(
    stack_czyx: np.ndarray,
    mask_zyx: np.ndarray,
    *,
    n_clusters: int,
    max_voxels: int,
    random_state: int,
) -> tuple[np.ndarray, KMeans, np.ndarray]:
    """在 mask 内拟合 KMeans，返回全分辨率标签（背景=-1）、模型、采样特征。"""
    m = mask_zyx.astype(bool)
    idx = np.argwhere(m)
    rng = np.random.default_rng(random_state)
    if idx.shape[0] > max_voxels:
        pick = rng.choice(idx.shape[0], size=max_voxels, replace=False)
        idx_fit = idx[pick]
    else:
        idx_fit = idx

    feats_fit = stack_czyx[:, idx_fit[:, 0], idx_fit[:, 1], idx_fit[:, 2]].T
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    km.fit(feats_fit)

    feats_all = stack_czyx[:, idx[:, 0], idx[:, 1], idx[:, 2]].T
    lab_all = km.predict(feats_all)
    labels = np.full(mask_zyx.shape, -1, dtype=np.int16)
    labels[idx[:, 0], idx[:, 1], idx[:, 2]] = lab_all.astype(np.int16)
    return labels, km, feats_fit


def _reorder_by_channel_mean(
    labels: np.ndarray,
    stack_czyx: np.ndarray,
    *,
    order_channel: int,
    n_clusters: int,
) -> tuple[np.ndarray, dict[int, float]]:
    """
    按指定通道类内均值升序重排聚类 ID，使跨病例 hab0/hab1/... 语义大致对齐
   （例如 ADC 低 → 可能更致密/受限扩散）。
    """
    ch = int(np.clip(order_channel, 0, stack_czyx.shape[0] - 1))
    means: dict[int, float] = {}
    for k in range(n_clusters):
        sel = labels == k
        if not np.any(sel):
            means[k] = float("inf")
        else:
            means[k] = float(stack_czyx[ch][sel].mean())
    order = sorted(means.keys(), key=lambda k: means[k])
    remap = {old: new for new, old in enumerate(order)}
    out = labels.copy()
    for old, new in remap.items():
        out[labels == old] = new
    ordered_means = {new: means[old] for old, new in remap.items()}
    return out, ordered_means


def partition_habitats(
    stack_czyx: np.ndarray,
    mask_zyx: np.ndarray,
    *,
    k_min: int = 3,
    k_max: int = 4,
    fixed_k: int | None = None,
    max_voxels: int = 12000,
    min_voxels_per_cluster: int = 30,
    order_channel: int = -1,
    random_state: int = 42,
) -> HabitatPartition | None:
    """
    强度 K-means 划分生境。

    K 选取：
      - ``fixed_k`` 非空：强制该 K
      - 否则在 ``[k_min, k_max]`` 上取 Calinski-Harabasz 最大者
      - 若某 K 存在过小簇（< min_voxels_per_cluster）则跳过
      - 全部失败则回退到可行的最大 K，再不行返回 None
    """
    m = mask_zyx.astype(bool)
    n_vox = int(m.sum())
    if n_vox < max(k_min, 2) * min_voxels_per_cluster:
        return None

    if order_channel < 0:
        order_channel = stack_czyx.shape[0] - 1  # 默认最后一通道（ADC）

    if fixed_k is not None:
        candidates = [int(fixed_k)]
    else:
        candidates = list(range(int(k_min), int(k_max) + 1))

    ch_scores: dict[int, float] = {}
    best_k: int | None = None
    best_score = -1.0
    best_labels: np.ndarray | None = None

    idx = np.argwhere(m)
    rng = np.random.default_rng(random_state)
    if idx.shape[0] > max_voxels:
        pick = rng.choice(idx.shape[0], size=max_voxels, replace=False)
        idx_fit = idx[pick]
    else:
        idx_fit = idx
    feats_fit = stack_czyx[:, idx_fit[:, 0], idx_fit[:, 1], idx_fit[:, 2]].T

    for k in candidates:
        if n_vox < k * min_voxels_per_cluster:
            continue
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            lab_fit = km.fit_predict(feats_fit)
        except Exception:
            continue
        counts = np.bincount(lab_fit, minlength=k)
        if counts.min() < max(5, min_voxels_per_cluster // 3):
            # 采样集上簇过小，不稳定
            ch_scores[k] = float("nan")
            continue
        try:
            score = float(calinski_harabasz_score(feats_fit, lab_fit))
        except Exception:
            ch_scores[k] = float("nan")
            continue
        ch_scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        # 回退：选不超过体素数约束的最大候选 K
        for k in sorted(candidates, reverse=True):
            if n_vox >= k * min_voxels_per_cluster:
                best_k = k
                break
        if best_k is None:
            return None

    labels, _km, _ = _fit_predict_full(
        stack_czyx,
        mask_zyx,
        n_clusters=best_k,
        max_voxels=max_voxels,
        random_state=random_state,
    )
    # 全分辨率再检查簇大小；过小则合并到最近中心（简单：标为邻域最大簇）
    for k in range(best_k):
        if int((labels == k).sum()) < min_voxels_per_cluster:
            # 将该簇并入体积最大的其他簇
            sizes = [(j, int((labels == j).sum())) for j in range(best_k) if j != k]
            if not sizes:
                continue
            target = max(sizes, key=lambda t: t[1])[0]
            labels[labels == k] = target

    # 压缩空簇后可能实际簇数减少 → 重新映射到 0..K'-1，但特征列仍按 max_k 对齐
    present = sorted(int(x) for x in np.unique(labels) if x >= 0)
    if len(present) < 2:
        return None
    remap = {old: new for new, old in enumerate(present)}
    labels2 = np.full_like(labels, -1)
    for old, new in remap.items():
        labels2[labels == old] = new
    labels = labels2
    best_k = len(present)

    labels, order_means = _reorder_by_channel_mean(
        labels, stack_czyx, order_channel=order_channel, n_clusters=best_k
    )

    props = {}
    for k in range(best_k):
        props[k] = float((labels == k).sum() / max(n_vox, 1))

    return HabitatPartition(
        labels_zyx=labels,
        selected_k=best_k,
        ch_scores=ch_scores,
        props=props,
        order_key_means=order_means,
    )


def habitat_features_from_stack(
    stack_czyx: np.ndarray,
    mask_zyx: np.ndarray,
    *,
    n_clusters: int = 4,
    max_voxels: int = 12000,
    random_state: int = 42,
) -> dict[str, float]:
    """
    兼容旧接口：固定 K，仅输出占比 + 类内均值（轻量摘要）。
    """
    c = stack_czyx.shape[0]
    part = partition_habitats(
        stack_czyx,
        mask_zyx,
        k_min=n_clusters,
        k_max=n_clusters,
        fixed_k=n_clusters,
        max_voxels=max_voxels,
        random_state=random_state,
    )
    if part is None:
        return _empty_habitat_feature_dict(n_clusters, c)

    out: dict[str, float] = {}
    for k in range(n_clusters):
        sel = part.labels_zyx == k
        out[f"habitat_prop_cluster{k}"] = float(sel.sum() / max(mask_zyx.astype(bool).sum(), 1))
        for ch in range(c):
            if np.any(sel):
                out[f"habitat_mean_ch{ch}_cluster{k}"] = float(stack_czyx[ch][sel].mean())
            else:
                out[f"habitat_mean_ch{ch}_cluster{k}"] = 0.0
    out["habitat_inertia"] = 0.0
    out["habitat_selected_k"] = float(part.selected_k)
    sc = part.ch_scores.get(part.selected_k, 0.0)
    out["habitat_ch_score"] = float(sc) if sc == sc else 0.0  # NaN-safe
    return out


def habitat_summary_from_partition(
    part: HabitatPartition,
    stack_czyx: np.ndarray,
    *,
    max_habitats: int,
) -> dict[str, float]:
    """固定长度摘要特征（按 max_habitats 零填充）。"""
    c = stack_czyx.shape[0]
    n = int((part.labels_zyx >= 0).sum())
    sc = part.ch_scores.get(part.selected_k, 0.0)
    out: dict[str, float] = {
        "habitat_selected_k": float(part.selected_k),
        "habitat_ch_score": float(sc) if sc == sc else 0.0,
    }
    for k_cand, score in part.ch_scores.items():
        out[f"habitat_ch_score_k{k_cand}"] = float(score) if score == score else 0.0

    for k in range(max_habitats):
        sel = part.labels_zyx == k
        out[f"habitat_prop_cluster{k}"] = float(sel.sum() / max(n, 1)) if n else 0.0
        for ch in range(c):
            out[f"habitat_mean_ch{ch}_cluster{k}"] = (
                float(stack_czyx[ch][sel].mean()) if np.any(sel) else 0.0
            )
        out[f"habitat_order_key_mean_{k}"] = float(part.order_key_means.get(k, 0.0))
    return out
