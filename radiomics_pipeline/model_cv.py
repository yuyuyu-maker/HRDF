"""特征筛选 + 降维 + 逻辑回归，分层 K 折交叉验证。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OOFRecord = dict[str, Any]


def _nan_safe_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """正类概率的切分点：最大化 Youden 指数 J = TPR − FPR。无法估计时回退 0.5。"""
    y_arr = np.asarray(y_true).astype(int).ravel()
    p_arr = np.asarray(prob, dtype=float).ravel()
    ok = np.isfinite(p_arr)
    y_arr, p_arr = y_arr[ok], p_arr[ok]
    if y_arr.size == 0 or len(np.unique(y_arr)) < 2:
        return 0.5
    _fpr, tpr, thresholds = roc_curve(y_arr, p_arr)
    j = tpr - _fpr
    finite = np.isfinite(thresholds)
    if not np.any(finite):
        return 0.5
    j = np.where(finite, j, -np.inf)
    thr = float(thresholds[int(np.argmax(j))])
    return thr if np.isfinite(thr) else 0.5


def _build_pipeline(
    n_features: int,
    *,
    random_state: int,
    select_k: int,
    pca_variance: float,
) -> Pipeline:
    k = max(5, min(select_k, max(1, n_features - 1)))
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kb", SelectKBest(mutual_info_classif, k=k)),
            ("pca", PCA(n_components=pca_variance, svd_solver="full")),
            (
                "clf",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )


def collect_repeated_oof(
    X: pd.DataFrame,
    y: pd.Series,
    subject_ids: pd.Index | list[str],
    *,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
    select_k: int = 80,
    pca_variance: float = 0.85,
) -> pd.DataFrame:
    """
    重复分层 K 折，汇总每例在验证折上的 OOF 预测与困难度。

    返回列：subject_id, y_true, n_val, n_wrong, error_rate, mean_prob,
    mean_prob_when_wrong, difficulty_score
    """
    X = _nan_safe_matrix(X)
    y = y.astype(int)
    ids = list(subject_ids)
    if len(ids) != len(y):
        raise ValueError(f"subject_ids 长度 {len(ids)} 与样本数 {len(y)} 不一致")

    records: list[OOFRecord] = []
    for rep in range(n_repeats):
        rs = random_state + rep
        pipe = _build_pipeline(
            X.shape[1],
            random_state=rs,
            select_k=select_k,
            pca_variance=pca_variance,
        )
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=rs
        )
        for fold, (train_idx, val_idx) in enumerate(cv.split(X.values, y.values)):
            pipe.fit(X.values[train_idx], y.values[train_idx])
            thr = youden_threshold(
                y.values[train_idx],
                pipe.predict_proba(X.values[train_idx])[:, 1],
            )
            proba = pipe.predict_proba(X.values[val_idx])[:, 1]
            pred = (proba >= thr).astype(int)
            for j, idx in enumerate(val_idx):
                yt = int(y.values[idx])
                pp = float(proba[j])
                pr = int(pred[j])
                records.append(
                    {
                        "subject_id": ids[idx],
                        "repeat": rep,
                        "fold": fold,
                        "y_true": yt,
                        "oof_prob": pp,
                        "oof_pred": pr,
                        "wrong": int(pr != yt),
                        "prob_label": pp if yt == 1 else 1.0 - pp,
                    }
                )

    detail = pd.DataFrame(records)
    summary = (
        detail.groupby("subject_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            n_val=("wrong", "count"),
            n_wrong=("wrong", "sum"),
            mean_prob=("oof_prob", "mean"),
            mean_prob_label=("prob_label", "mean"),
        )
        .astype({"n_wrong": int, "n_val": int})
    )
    summary["error_rate"] = summary["n_wrong"] / summary["n_val"].clip(lower=1)

    wrong_only = (
        detail[detail["wrong"] == 1]
        .groupby("subject_id")["prob_label"]
        .mean()
        .rename("mean_prob_when_wrong")
    )
    summary = summary.merge(wrong_only, on="subject_id", how="left")
    summary["mean_prob_when_wrong"] = summary["mean_prob_when_wrong"].fillna(
        summary["mean_prob_label"]
    )
    # 错误率越高、错得越自信 → 越难
    summary["difficulty_score"] = summary["error_rate"] * summary[
        "mean_prob_when_wrong"
    ].fillna(0.5)
    summary = summary.sort_values(
        ["error_rate", "difficulty_score", "n_wrong"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return summary


def run_stratified_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    select_k: int = 80,
    pca_variance: float = 0.85,
) -> dict[str, Any]:
    """
    Pipeline: StandardScaler → SelectKBest(mutual_info) → PCA → LogisticRegression。

    准确率与 F1 使用各折训练集概率上的 Youden 切分点，不使用 0.5。
    """
    X = _nan_safe_matrix(X)
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("标签只有一个类别，无法做分层 K 折与 ROC-AUC。")
    if len(y) < n_splits:
        raise ValueError(f"样本数 {len(y)} 小于折数 {n_splits}，请减小 --k-fold。")
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    aucs: list[float] = []
    accs: list[float] = []
    f1s: list[float] = []
    thrs: list[float] = []
    for train_idx, val_idx in cv.split(X.values, y.values):
        pipe = _build_pipeline(
            X.shape[1],
            random_state=random_state,
            select_k=select_k,
            pca_variance=pca_variance,
        )
        y_train = y.values[train_idx]
        y_val = y.values[val_idx]
        pipe.fit(X.values[train_idx], y_train)
        thr = youden_threshold(y_train, pipe.predict_proba(X.values[train_idx])[:, 1])
        proba = pipe.predict_proba(X.values[val_idx])[:, 1]
        pred = (proba >= thr).astype(int)
        if len(np.unique(y_val)) >= 2:
            aucs.append(float(roc_auc_score(y_val, proba)))
        accs.append(float(accuracy_score(y_val, pred)))
        f1s.append(float(f1_score(y_val, pred, zero_division=0)))
        thrs.append(thr)
    if not aucs:
        raise ValueError("各折验证集都只有一个类别，无法计算 ROC-AUC。")
    return {
        "n_samples": int(len(y)),
        "n_splits": n_splits,
        "decision_threshold": "youden",
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "accuracy_mean": float(np.mean(accs)),
        "f1_mean": float(np.mean(f1s)),
        "youden_threshold_mean": float(np.mean(thrs)),
        "fold_roc_auc": aucs,
        "fold_youden_threshold": thrs,
    }


def collect_oof_probs(
    X: pd.DataFrame,
    y: pd.Series,
    subject_ids: pd.Index | list[str],
    *,
    n_splits: int = 5,
    random_state: int = 42,
    select_k: int = 80,
    pca_variance: float = 0.85,
    val_ids_by_fold: dict[int, list[str]] | None = None,
) -> pd.DataFrame:
    """
    单次分层 K 折 OOF 概率。

    若提供 ``val_ids_by_fold``（fold 索引 → 验证集 subject_id 列表），
    则与外部模型（如 CNN checkpoint）使用相同折划分；否则按 random_state 自动划分。

    返回列：subject_id, y_true, oof_prob, fold
    """
    X = _nan_safe_matrix(X)
    y = y.astype(int)
    ids = [str(s).zfill(3) if str(s).isdigit() else str(s) for s in subject_ids]
    if len(ids) != len(y):
        raise ValueError(f"subject_ids 长度 {len(ids)} 与样本数 {len(y)} 不一致")

    sid_to_idx = {sid: i for i, sid in enumerate(ids)}
    records: list[OOFRecord] = []

    if val_ids_by_fold is not None:
        fold_items = sorted(val_ids_by_fold.items(), key=lambda x: x[0])
        for fold, val_sids in fold_items:
            val_set = {str(s).zfill(3) if str(s).isdigit() else str(s) for s in val_sids}
            val_idx = [sid_to_idx[s] for s in val_set if s in sid_to_idx]
            if not val_idx:
                continue
            train_idx = [i for i, sid in enumerate(ids) if sid not in val_set]
            if len(train_idx) < 2 or len(set(y.values[train_idx])) < 2:
                continue
            pipe = _build_pipeline(
                X.shape[1],
                random_state=random_state,
                select_k=select_k,
                pca_variance=pca_variance,
            )
            pipe.fit(X.values[train_idx], y.values[train_idx])
            proba = pipe.predict_proba(X.values[val_idx])[:, 1]
            for j, idx in enumerate(val_idx):
                records.append(
                    {
                        "subject_id": ids[idx],
                        "y_true": int(y.values[idx]),
                        "oof_prob": float(proba[j]),
                        "fold": int(fold),
                    }
                )
    else:
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        for fold, (train_idx, val_idx) in enumerate(cv.split(X.values, y.values)):
            pipe = _build_pipeline(
                X.shape[1],
                random_state=random_state,
                select_k=select_k,
                pca_variance=pca_variance,
            )
            pipe.fit(X.values[train_idx], y.values[train_idx])
            proba = pipe.predict_proba(X.values[val_idx])[:, 1]
            for j, idx in enumerate(val_idx):
                records.append(
                    {
                        "subject_id": ids[idx],
                        "y_true": int(y.values[idx]),
                        "oof_prob": float(proba[j]),
                        "fold": fold,
                    }
                )

    out = pd.DataFrame(records)
    if out.empty:
        return out
    return out.sort_values("subject_id").reset_index(drop=True)


def fit_full_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    select_k: int = 80,
    pca_variance: float = 0.85,
) -> Pipeline:
    X = _nan_safe_matrix(X)
    y = y.astype(int)
    pipe = _build_pipeline(
        X.shape[1],
        random_state=random_state,
        select_k=select_k,
        pca_variance=pca_variance,
    )
    pipe.fit(X.values, y.values)
    return pipe
