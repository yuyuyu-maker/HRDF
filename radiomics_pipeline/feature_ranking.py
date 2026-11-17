"""特征重要性：互信息排序并写出 CSV。"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def rank_features_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    X = X.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    mi = mutual_info_classif(X.values, y.astype(int).values, random_state=random_state)
    out = pd.DataFrame({"feature": X.columns, "mutual_info": mi})
    out = out.sort_values("mutual_info", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out
