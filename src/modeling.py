"""
建模模組：K-Means 音訊風格族群分群、（之後會加）成功預測模型。

分群邏輯對應 original_analysis.py L375-482（剔除離群值後的第二輪分群，
這是原程式最終採用的版本，PCA 3D/2D 視覺化不搬過來，只保留分群本身）。

重要設計原則（對應 MODEL_AUDIT.md 風險三）：
- 群集的數字編號（0,1,2,3）不能保證跨資料集/跨版本語意一致。
- 這個模組只負責「算出 centroid 數值」，不負責幫每個編號取名字。
- 幫群集取中文名稱（例如「高能量舞曲型」）是一個需要人確認的判斷，
  由 assign_cluster_names() 接收一個「已驗證」的 mapping，而不是憑空猜。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 分群使用的 9 個音訊特徵，逐字對照 original_analysis.py L275-277
CLUSTER_FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "liveness",
    "duration_ms",
]

N_CLUSTERS = 4
RANDOM_STATE = 42
INSTRUMENTALNESS_OUTLIER_THRESHOLD = 0.1  # 對應原程式：極高器樂比例的專輯導入曲，視為離群值排除


@dataclass
class ClusterBundle:
    """分群結果的完整封裝，包含驗證用的 centroid profile。"""

    model: KMeans
    scaler: StandardScaler
    feature_columns: list[str]
    hot_stats_clustered: pd.DataFrame       # 已排除離群值、含 cluster_label 欄位
    excluded_songs: pd.DataFrame            # 因 instrumentalness 過高被排除的歌曲（通常很少）
    cluster_stats: pd.DataFrame             # 每個 cluster_label 的特徵平均值（未標準化，人看得懂的單位）
    cluster_sizes: pd.Series                # 每個 cluster_label 的歌曲數
    cluster_names: dict[int, str] | None = field(default=None)  # 需另外呼叫 assign_cluster_names() 才會填入
    pca_explained_variance: tuple[float, float] | None = field(default=None)


def fit_audio_clusters(hot_stats: pd.DataFrame) -> ClusterBundle:
    """對歌曲層級資料做 K-Means 分群，回傳結果與可供人工驗證的 centroid profile。

    注意：這個函式**不會**幫每個群取名字。取名字必須看過 cluster_stats 的實際數值後，
    再呼叫 assign_cluster_names() 明確指定，避免對應到原程式碼裡的舊 mapping。
    """
    excluded_mask = hot_stats["instrumentalness"] >= INSTRUMENTALNESS_OUTLIER_THRESHOLD
    excluded_songs = hot_stats[excluded_mask].copy()
    clustered_base = hot_stats[~excluded_mask].copy()

    X = clustered_base[CLUSTER_FEATURE_COLUMNS]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    labels = model.fit_predict(X_scaled)
    clustered_base["cluster_label"] = labels

    cluster_stats = (
        clustered_base.groupby("cluster_label")[CLUSTER_FEATURE_COLUMNS].mean().round(3)
    )
    cluster_sizes = clustered_base["cluster_label"].value_counts().sort_index()

    return ClusterBundle(
        model=model,
        scaler=scaler,
        feature_columns=CLUSTER_FEATURE_COLUMNS,
        hot_stats_clustered=clustered_base,
        excluded_songs=excluded_songs,
        cluster_stats=cluster_stats,
        cluster_sizes=cluster_sizes,
    )


def assign_cluster_names(bundle: ClusterBundle, name_map: dict[int, str]) -> ClusterBundle:
    """把「已經人工看過 cluster_stats、確認過的」名稱對應表，套用到 ClusterBundle 上。

    Parameters
    ----------
    name_map : dict[int, str]
        必須明確列出每個 cluster_label（0 ~ N_CLUSTERS-1）對應的中文名稱。
        不接受自動猜測——名稱一律要有人看過 profile 之後才能決定。
    """
    missing = set(bundle.cluster_stats.index) - set(name_map.keys())
    if missing:
        raise ValueError(f"name_map 缺少這些 cluster_label 的名稱：{sorted(missing)}")

    bundle.cluster_names = dict(name_map)
    bundle.hot_stats_clustered["cluster_name"] = bundle.hot_stats_clustered["cluster_label"].map(
        name_map
    )
    return bundle


# 已於 2026-07-16 用實際 centroid 數值 + 雷達圖確認過對應關係，不是憑空沿用原程式數字。
# 若之後資料更新導致重新訓練，必須重新跑 fit_audio_clusters() 看 cluster_stats，
# 確認語意還是這樣才能繼續用這份 mapping；不可以未經確認就直接套用。
CONFIRMED_CLUSTER_NAMES = {
    0: "高能量舞曲型",
    1: "抒情原音型",
    2: "輕快節奏型",
    3: "慢節奏情感型",
}


def fit_and_name_audio_clusters(hot_stats: pd.DataFrame) -> ClusterBundle:
    """方便呼叫端使用的組合函式：分群 + 套用已確認的命名。"""
    bundle = fit_audio_clusters(hot_stats)
    return assign_cluster_names(bundle, CONFIRMED_CLUSTER_NAMES)


def compute_pca_projection(bundle: ClusterBundle) -> ClusterBundle:
    """在分群用的同一組標準化特徵上做 2D PCA，加入 PC1/PC2 欄位供散點圖使用。

    對應 original_analysis.py L451-459（僅取 2D 版本，3D 版本依先前討論不搬進網頁）。
    """
    X = bundle.hot_stats_clustered[bundle.feature_columns]
    X_scaled = bundle.scaler.transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    bundle.hot_stats_clustered = bundle.hot_stats_clustered.copy()
    bundle.hot_stats_clustered["PC1"] = X_pca[:, 0]
    bundle.hot_stats_clustered["PC2"] = X_pca[:, 1]
    bundle.pca_explained_variance = tuple(pca.explained_variance_ratio_.round(3))

    return bundle


# ---------------------------------------------------------------------------
# 成功預測模型（Logistic Regression）— 對應 original_analysis.py L873-950
# ---------------------------------------------------------------------------

# 模型輸入欄位順序，固定不可更動（對照 MODEL_AUDIT.md「模型輸入欄位（原始版本）」）
SIMULATOR_FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "duration_ms",
]

MODEL_TEST_SIZE = 0.3
MODEL_RANDOM_STATE = 17  # 逐字比照原程式，不是我方便選的數字；且明確不使用原程式中被註解掉的「試多個seed挑最高分」做法


@dataclass
class ModelBundle:
    """Logistic Regression 模型的完整封裝，含評估結果與模擬器需要的參考分布。"""

    model: LogisticRegression
    scaler: StandardScaler
    feature_columns: list[str]
    accuracy: float
    auc: float
    coef_df: pd.DataFrame            # feature / coef，coef 越大代表越正向影響成功機率
    n_train: int
    n_test: int
    reference_scores: "pd.Series"    # 全部（訓練用的525首）歌曲的模型預測分數，供模擬器算 percentile 用


def train_success_model(
    hot_stats_filtered: pd.DataFrame,
    test_size: float = MODEL_TEST_SIZE,
    random_state: int = MODEL_RANDOM_STATE,
) -> ModelBundle:
    """訓練 Logistic Regression 成功預測模型，逐字比照原程式的切分與模型設定。

    Parameters
    ----------
    hot_stats_filtered : pd.DataFrame
        需為已剔除 instrumentalness>=0.1 離群值的資料（跟分群用同一個篩選後子集），
        且需含 success_class 欄位（classify_success_types 的輸出）。
    """
    X = hot_stats_filtered[SIMULATOR_FEATURE_COLUMNS].copy()
    y = hot_stats_filtered["success_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    coef_df = pd.DataFrame(
        {"feature": SIMULATOR_FEATURE_COLUMNS, "coef": model.coef_[0]}
    ).sort_values(by="coef", ascending=True).reset_index(drop=True)

    # 模擬器要顯示「相對於訓練資料歌曲的 percentile」，這裡取「全部參與訓練/測試的525首歌」
    # （不是只有訓練集的367首）在同一個 scaler+model 下算出的預測分數，作為比較基準。
    # 這是原程式沒有的新功能，用全部建模資料當基準比只用訓練集更穩定，這裡先說明這個假設。
    X_all_scaled = scaler.transform(X)
    reference_scores = pd.Series(model.predict_proba(X_all_scaled)[:, 1], index=X.index)

    return ModelBundle(
        model=model,
        scaler=scaler,
        feature_columns=SIMULATOR_FEATURE_COLUMNS,
        accuracy=accuracy,
        auc=auc,
        coef_df=coef_df,
        n_train=len(X_train),
        n_test=len(X_test),
        reference_scores=reference_scores,
    )


def score_audio_features(input_features: dict, bundle: ModelBundle) -> dict:
    """把使用者輸入的音訊特徵丟進模型，回傳探索性分數、percentile 與特徵貢獻。

    Parameters
    ----------
    input_features : dict
        必須包含 SIMULATOR_FEATURE_COLUMNS 這 7 個 key，duration_ms 需已經是毫秒。

    Returns
    -------
    dict
        {
            "score": float,              # 0-1，模型預測的成功機率（探索性，不是保證）
            "percentile": float,         # 0-100，相對於訓練資料歌曲的百分位
            "contributions": pd.DataFrame,  # feature / contribution / direction，依影響力排序
        }
    """
    X_input = pd.DataFrame([input_features])[bundle.feature_columns]
    X_scaled = bundle.scaler.transform(X_input)

    score = float(bundle.model.predict_proba(X_scaled)[0, 1])
    percentile = float((bundle.reference_scores < score).mean() * 100)

    contributions = X_scaled[0] * bundle.model.coef_[0]
    contrib_df = pd.DataFrame(
        {
            "feature": bundle.feature_columns,
            "contribution": contributions,
            "direction": ["正向" if c > 0 else "負向" for c in contributions],
        }
    )
    contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
    contrib_df = contrib_df.sort_values("abs_contribution", ascending=False).drop(
        columns="abs_contribution"
    ).reset_index(drop=True)

    return {"score": score, "percentile": percentile, "contributions": contrib_df}
