"""
測試 data_pipeline.py 的欄位驗證與成功型態分類邏輯。

執行方式：pytest tests/test_data_pipeline.py -v
"""

import pandas as pd
import pytest

from data_pipeline import (
    build_monthly_stats,
    classify_success_types,
    validate_chart_schema,
)


# ---------------------------------------------------------------------------
# validate_chart_schema
# ---------------------------------------------------------------------------

def _minimal_valid_df(n=5):
    return pd.DataFrame(
        {
            "title": [f"song{i}" for i in range(n)],
            "artist": [f"artist{i}" for i in range(n)],
            "position": [1, 5, 10, 20, 50][:n],
            "date": pd.to_datetime(["2024-01-01"] * n),
            "duration_ms": [200000] * n,
            "danceability": [0.5] * n,
            "energy": [0.5] * n,
            "valence": [0.5] * n,
            "acousticness": [0.1] * n,
            "speechiness": [0.05] * n,
            "instrumentalness": [0.0] * n,
        }
    )


def test_validate_chart_schema_valid_data_passes():
    report = validate_chart_schema(_minimal_valid_df())
    assert report["is_valid"] is True
    assert report["issues"] == []


def test_validate_chart_schema_missing_required_column_fails():
    df = _minimal_valid_df().drop(columns=["danceability"])
    report = validate_chart_schema(df)
    assert report["is_valid"] is False
    assert any("danceability" in issue for issue in report["issues"])


def test_validate_chart_schema_empty_dataframe_fails():
    df = _minimal_valid_df(n=0)
    report = validate_chart_schema(df)
    assert report["is_valid"] is False


def test_validate_chart_schema_out_of_range_feature_warns_not_fails():
    df = _minimal_valid_df()
    df.loc[0, "danceability"] = 1.5  # 超出 0-1 合理範圍
    report = validate_chart_schema(df)
    # 範圍異常是警告，不應該擋住整個流程
    assert report["is_valid"] is True
    assert any("danceability" in w for w in report["warnings"])


# ---------------------------------------------------------------------------
# classify_success_types
# ---------------------------------------------------------------------------

def _synthetic_hot_stats():
    """8首合成歌曲，涵蓋高/低 best_rank 與 total_days 的各種組合。"""
    return pd.DataFrame(
        {
            "title": [f"song{i}" for i in range(8)],
            "artist": [f"artist{i}" for i in range(8)],
            "best_rank": [1, 2, 3, 4, 40, 41, 42, 43],
            "total_days": [100, 100, 5, 5, 100, 100, 5, 5],
        }
    )


def test_classify_success_types_others_is_not_a_success_type():
    """回歸測試：對應我們實際發現過的 bug——Others 不該被當成成功型態之一去比較佔比。"""
    hot_stats = _synthetic_hot_stats()
    classified, _ = classify_success_types(hot_stats)
    assert "Others" in classified["success_type"].unique()
    # Others 對應的 success_class 必須是 0（不成功），不能被誤判成功
    others_rows = classified[classified["success_type"] == "Others"]
    assert (others_rows["success_class"] == 0).all()


def test_classify_success_types_thresholds_are_fixed_across_filtered_subsets():
    """回歸測試：門檻只在全部資料上算一次，套用到篩選後子集時不能重新計算。"""
    hot_stats = _synthetic_hot_stats()
    full_classified, thresholds = classify_success_types(hot_stats)

    subset = hot_stats.iloc[:4]
    subset_classified, reused_thresholds = classify_success_types(subset, thresholds=thresholds)

    assert reused_thresholds == thresholds
    for title in subset["title"]:
        full_label = full_classified.loc[full_classified["title"] == title, "success_type"].iloc[0]
        subset_label = subset_classified.loc[subset_classified["title"] == title, "success_type"].iloc[0]
        assert full_label == subset_label


def test_classify_success_types_without_explicit_thresholds_computes_fresh():
    hot_stats = _synthetic_hot_stats()
    _, thresholds = classify_success_types(hot_stats)
    assert set(thresholds.keys()) >= {"thresh_peak", "thresh_short", "thresh_long"}


# ---------------------------------------------------------------------------
# build_monthly_stats — 選配天氣資料缺少時的 fallback
# ---------------------------------------------------------------------------

def _synthetic_daily_chart_df():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "danceability": [0.5] * 40,
            "energy": [0.5] * 40,
            "valence": [0.5] * 40,
            "acousticness": [0.1] * 40,
            "liveness": [0.1] * 40,
            "tempo": [120.0] * 40,
        }
    )


def test_build_monthly_stats_without_weather_does_not_crash():
    """驗收標準：沒有 weather_all.csv 時，核心功能（月份統計）仍要能正常運作。"""
    result = build_monthly_stats(_synthetic_daily_chart_df(), weather_df=None)
    assert result["correlation"] is None
    assert result["weather_missing_dates"] is None
    assert len(result["monthly_stats"]) > 0


def test_build_monthly_stats_with_weather_computes_correlation():
    daily = _synthetic_daily_chart_df()
    weather = pd.DataFrame(
        {
            "date": daily["date"],
            "temp": [25.0] * 40,
            "precip": [0.0] * 40,
            "sunshine": [8.0] * 40,
        }
    )
    result = build_monthly_stats(daily, weather_df=weather)
    assert result["correlation"] is not None
    assert result["weather_missing_dates"] == 0
