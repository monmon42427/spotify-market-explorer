"""
測試 modeling.py 的分群、模型訓練與評分邏輯。

執行方式：pytest tests/test_modeling.py -v
"""

import numpy as np
import pandas as pd
import pytest

from modeling import (
    INSTRUMENTALNESS_OUTLIER_THRESHOLD,
    SIMULATOR_FEATURE_COLUMNS,
    assign_cluster_names,
    fit_audio_clusters,
    score_audio_features,
    train_success_model,
)


def _synthetic_hot_stats(n=60, seed=42):
    """合成一份跟真實 hot_stats 欄位相容的資料，用於測試分群與模型訓練邏輯。"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "title": [f"song{i}" for i in range(n)],
            "artist": [f"artist{i}" for i in range(n)],
            "danceability": rng.uniform(0, 1, n),
            "energy": rng.uniform(0, 1, n),
            "valence": rng.uniform(0, 1, n),
            "tempo": rng.uniform(60, 180, n),
            "acousticness": rng.uniform(0, 1, n),
            "speechiness": rng.uniform(0, 1, n),
            "instrumentalness": rng.uniform(0, 0.05, n),  # 大多數是低器樂比例
            "liveness": rng.uniform(0, 1, n),
            "duration_ms": rng.uniform(120000, 300000, n),
            "success_class": rng.integers(0, 2, n),
        }
    )
    # 混入幾首高器樂比例的離群值，測試篩選邏輯是否確實排除
    df.loc[0:2, "instrumentalness"] = 0.95
    return df


# ---------------------------------------------------------------------------
# fit_audio_clusters — 離群值排除
# ---------------------------------------------------------------------------

def test_fit_audio_clusters_excludes_high_instrumentalness_outliers():
    hot_stats = _synthetic_hot_stats()
    bundle = fit_audio_clusters(hot_stats)

    assert len(bundle.excluded_songs) == 3  # 上面刻意混入的 3 首離群值
    assert (bundle.excluded_songs["instrumentalness"] >= INSTRUMENTALNESS_OUTLIER_THRESHOLD).all()
    assert (bundle.hot_stats_clustered["instrumentalness"] < INSTRUMENTALNESS_OUTLIER_THRESHOLD).all()


def test_assign_cluster_names_raises_on_incomplete_mapping():
    hot_stats = _synthetic_hot_stats()
    bundle = fit_audio_clusters(hot_stats)
    with pytest.raises(ValueError):
        assign_cluster_names(bundle, {0: "只給一個名字"})


# ---------------------------------------------------------------------------
# train_success_model / score_audio_features
# ---------------------------------------------------------------------------

def test_train_success_model_is_deterministic():
    hot_stats = _synthetic_hot_stats()
    bundle = fit_audio_clusters(hot_stats)

    model_a = train_success_model(bundle.hot_stats_clustered)
    model_b = train_success_model(bundle.hot_stats_clustered)

    assert model_a.n_train == model_b.n_train
    np.testing.assert_array_almost_equal(model_a.model.coef_, model_b.model.coef_)


def test_score_audio_features_returns_valid_probability_range():
    hot_stats = _synthetic_hot_stats()
    bundle = fit_audio_clusters(hot_stats)
    model_bundle = train_success_model(bundle.hot_stats_clustered)

    test_input = {col: 0.5 for col in SIMULATOR_FEATURE_COLUMNS}
    test_input["duration_ms"] = 200000

    result = score_audio_features(test_input, model_bundle)

    assert 0.0 <= result["score"] <= 1.0
    assert 0.0 <= result["percentile"] <= 100.0
    assert set(result["contributions"]["feature"]) == set(SIMULATOR_FEATURE_COLUMNS)


def test_score_audio_features_extreme_inputs_do_not_crash():
    """邊界測試：全部特徵推到 0 或 1 的極端值，模型仍要能正常算出結果，不能crash。"""
    hot_stats = _synthetic_hot_stats()
    bundle = fit_audio_clusters(hot_stats)
    model_bundle = train_success_model(bundle.hot_stats_clustered)

    extreme_input = {
        "danceability": 1.0,
        "energy": 0.0,
        "valence": 1.0,
        "acousticness": 0.0,
        "speechiness": 1.0,
        "instrumentalness": 0.05,
        "duration_ms": 600000,
    }
    result = score_audio_features(extreme_input, model_bundle)
    assert 0.0 <= result["score"] <= 1.0
