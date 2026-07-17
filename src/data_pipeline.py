"""
資料處理模組。

負責：讀取原始 Spotify 榜單 CSV、驗證欄位、篩選台灣資料。
不負責：任何分析判斷（成功型態、分群、模型）。那些邏輯在 modeling.py。
"""

from __future__ import annotations

import pandas as pd

# 原始 CSV 需要用到的欄位（對照 data/README.md 的必要欄位清單）
REQUIRED_RAW_COLUMNS = [
    "name",
    "artists",
    "country",
    "daily_rank",
    "snapshot_date",
    "album_name",
    "album_release_date",
    "is_explicit",
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "popularity",
]

# 讀完之後統一改名，對齊原始分析程式（original_analysis.py）的欄位命名習慣
RENAME_MAP = {
    "name": "title",
    "artists": "artist",
    "daily_rank": "position",
    "snapshot_date": "date",
}


def load_chart_data(path: str, country: str = "TW") -> pd.DataFrame:
    """讀取 Spotify 每日榜單 CSV，只挑必要欄位、篩出指定國家，並轉換日期型別。

    只做讀取與最基本的型別轉換，不做任何成功型態判斷或聚合。

    Parameters
    ----------
    path : str
        CSV 檔案路徑（例如 data/TopSpotifySongsin73Countries.csv）。
    country : str
        要篩選的國家代碼，預設 'TW'。

    Returns
    -------
    pd.DataFrame
        已篩選、改名、日期轉型的原始（歌曲 x 每日）資料，尚未聚合成歌曲層級。
    """
    df = pd.read_csv(path, usecols=REQUIRED_RAW_COLUMNS)
    df = df.rename(columns=RENAME_MAP)
    df = df[df["country"] == country].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["album_release_date"] = pd.to_datetime(
        df["album_release_date"], errors="coerce"
    )

    return df


def validate_chart_schema(df: pd.DataFrame) -> dict:
    """檢查資料是否符合後續分析所需的基本假設，回傳一份檢查報告（不會中斷程式）。

    檢查項目對應 data/README.md「資料驗證要求」：
    - 必要欄位是否存在
    - 日期轉換後有多少筆變成缺失值（errors="coerce" 的副作用）
    - 音訊特徵是否落在合理範圍
    - duration_ms 是否為正數
    - daily_rank（position）是否為合理正整數

    Returns
    -------
    dict
        {
            "is_valid": bool,          # 是否可以繼續往下跑
            "row_count": int,
            "issues": list[str],       # 會擋住流程的嚴重問題
            "warnings": list[str],     # 不會擋流程，但要顯示給使用者看
        }
    """
    issues: list[str] = []
    warnings: list[str] = []

    expected_cols = {
        "title",
        "artist",
        "position",
        "date",
        "duration_ms",
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "speechiness",
        "instrumentalness",
    }
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        issues.append(f"缺少必要欄位：{sorted(missing_cols)}")

    if len(df) == 0:
        issues.append("篩選後資料筆數為 0（可能是 country 篩選條件錯誤或來源檔案為空）")

    if "date" in df.columns:
        n_bad_date = df["date"].isna().sum()
        if n_bad_date > 0:
            warnings.append(f"{n_bad_date} 筆 snapshot_date 轉換失敗（已變成缺失值）")

    if "duration_ms" in df.columns:
        n_bad_duration = (df["duration_ms"] <= 0).sum()
        if n_bad_duration > 0:
            warnings.append(f"{n_bad_duration} 筆 duration_ms <= 0，屬於不合理數值")

    if "position" in df.columns:
        n_bad_rank = (~df["position"].between(1, 200)).sum()
        if n_bad_rank > 0:
            warnings.append(f"{n_bad_rank} 筆 daily_rank 不在合理範圍 (1-200)")

    zero_one_features = [
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "speechiness",
        "instrumentalness",
        "liveness",
    ]
    for col in zero_one_features:
        if col in df.columns:
            n_out_of_range = (~df[col].between(0, 1)).sum()
            if n_out_of_range > 0:
                warnings.append(f"{n_out_of_range} 筆 {col} 不在 0-1 範圍內")

    return {
        "is_valid": len(issues) == 0,
        "row_count": len(df),
        "issues": issues,
        "warnings": warnings,
    }


def build_song_level_stats(df: pd.DataFrame) -> pd.DataFrame:
    """把每日榜單資料（load_chart_data 的輸出）聚合成歌曲層級的 hot_stats。

    對應 original_analysis.py L15-53。邏輯完全比照原始版本：
    - rank_score = 51 - position（假設榜單為 Top 50，daily_rank 落在 1-50）
    - 音訊特徵取每首歌上榜期間的平均值
    - best_rank 取最高名次（數字最小）、total_days 取不重複上榜天數
    - 專輯資訊（album_name / album_release_date / is_explicit）取該歌曲最新一筆

    Returns
    -------
    pd.DataFrame
        一列一首歌（以 title + artist 為單位），依 score_sum 由高到低排序。
    """
    df = df.copy()
    df["rank_score"] = 51 - df["position"]

    hot_stats_core = (
        df.groupby(["title", "artist"])
        .agg(
            duration_ms=("duration_ms", "mean"),
            danceability=("danceability", "mean"),
            energy=("energy", "mean"),
            key=("key", "mean"),
            loudness=("loudness", "mean"),
            mode=("mode", "mean"),
            speechiness=("speechiness", "mean"),
            acousticness=("acousticness", "mean"),
            instrumentalness=("instrumentalness", "mean"),
            liveness=("liveness", "mean"),
            valence=("valence", "mean"),
            tempo=("tempo", "mean"),
            time_signature=("time_signature", "max"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            popularity=("popularity", "mean"),
            score_sum=("rank_score", "sum"),
            score_avg=("rank_score", "mean"),
            total_days=("date", "nunique"),
            best_rank=("position", "min"),
        )
        .reset_index()
    )

    latest = (
        df.sort_values(by="album_release_date", ascending=False)
        .groupby(["title", "artist"], as_index=False)
        .head(1)[["title", "artist", "album_name", "album_release_date", "is_explicit"]]
    )

    hot_stats = hot_stats_core.merge(latest, on=["title", "artist"], how="left")
    hot_stats = hot_stats.sort_values(by="score_sum", ascending=False).reset_index(drop=True)

    return hot_stats


def classify_success_types(
    hot_stats: pd.DataFrame,
    thresholds: dict | None = None,
    q_peak: float = 0.30,
    q_short: float = 0.50,
    q_long: float = 0.70,
) -> tuple[pd.DataFrame, dict]:
    """依 best_rank / total_days 分位數門檻，將歌曲分類為五種成功型態。

    對應 original_analysis.py L170-228。分類邏輯逐字保留，包含 Midrunner 條件
    中的 `peak < 28` 這個原始寫死的數字（不是由 thresholds 動態算出，是原分析
    的既有設計，這裡沿用、不更動）。

    門檻政策（已與專案作者確認 2026-07-16）：門檻只在「全部歷史資料」上計算一次，
    之後即使 UI 依日期區間篩選歌曲，也只影響「顯示哪些歌」，不會重新計算門檻、
    不會改變任何一首歌的 success_type。

    Parameters
    ----------
    hot_stats : pd.DataFrame
        build_song_level_stats() 的輸出，需包含 best_rank、total_days 欄位。
    thresholds : dict | None
        若為 None，會用傳入的 hot_stats（通常是全部資料）現場計算門檻並回傳。
        若要分類「篩選後的子集」，務必把第一次呼叫（全部資料）回傳的 thresholds
        傳進來，確保門檻不會因篩選而改變。

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (加了 success_type / success_class 欄位的 DataFrame, 本次使用的 thresholds)
    """
    if thresholds is None:
        thresholds = {
            "q_peak": q_peak,
            "q_short": q_short,
            "q_long": q_long,
            "thresh_peak": hot_stats["best_rank"].quantile(q_peak),
            "thresh_short": hot_stats["total_days"].quantile(q_short),
            "thresh_long": hot_stats["total_days"].quantile(q_long),
        }

    thresh_peak = thresholds["thresh_peak"]
    thresh_short = thresholds["thresh_short"]
    thresh_long = thresholds["thresh_long"]

    def _classify(row):
        peak, days = row["best_rank"], row["total_days"]
        if peak <= thresh_peak and days >= thresh_long:
            return "Evergreen"
        if peak <= thresh_peak and days <= thresh_short:
            return "Viral"
        if peak > thresh_peak and days >= thresh_long:
            return "Climber"
        if thresh_short < days < thresh_long and peak < 28:
            return "Midrunner"
        return "Others"

    out = hot_stats.copy()
    out["success_type"] = out.apply(_classify, axis=1)
    out["success_class"] = out["success_type"].map(lambda x: 0 if x == "Others" else 1)

    return out, thresholds


# ---------------------------------------------------------------------------
# 月份季節性與天氣分析 — 對應 original_analysis.py L700-825
# ---------------------------------------------------------------------------

# 注意：原程式這段用的變數叫 df_2024，但實際上是「全部日期範圍」的每日榜單資料，
# 不是只篩 2024 年——這是原程式的命名誤導，這裡沿用邏輯但不沿用誤導性的變數名稱。

MONTHLY_FEATURE_COLUMNS = ["danceability", "energy", "valence", "acousticness", "liveness", "tempo"]
WEATHER_COLUMNS = ["temp", "precip", "sunshine"]


def build_monthly_stats(daily_chart_df: pd.DataFrame, weather_df: pd.DataFrame | None = None) -> dict:
    """把每日榜單資料聚合成「日期層級」與「月份層級」的音訊特徵統計，選配天氣合併。

    對應 original_analysis.py L707-751。

    Parameters
    ----------
    daily_chart_df : pd.DataFrame
        load_chart_data() 的輸出（每日 x 每首歌的原始榜單列，尚未聚合成歌曲層級）。
    weather_df : pd.DataFrame | None
        weather_all.csv 讀入的結果，需含 date, temp, precip, sunshine 欄位。若為 None，
        代表沒有天氣資料，只回傳月份季節性統計，不含相關矩陣（對應「選配資料缺少時不能讓App崩潰」）。

    Returns
    -------
    dict
        {
            "daily_stats": 日期層級平均音訊特徵,
            "weather_merged": 日期層級 + 天氣（None 如果沒有天氣資料）,
            "monthly_stats": 月份(1-12)層級中位數音訊特徵，跨所有年份合併，代表「平均季節性樣貌」，
                              不是單一年份的趨勢,
            "correlation": 音訊特徵 x 天氣的相關矩陣（None 如果沒有天氣資料）,
            "weather_missing_dates": 天氣合併後仍缺值的日期數（None 如果沒有天氣資料）,
        }
    """
    daily_stats = (
        daily_chart_df.groupby("date")[MONTHLY_FEATURE_COLUMNS].mean().reset_index()
    )

    weather_merged = None
    correlation = None
    weather_missing_dates = None

    if weather_df is not None:
        weather = weather_df.copy()
        weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
        weather_merged = daily_stats.merge(weather, how="left", on="date")
        weather_missing_dates = int(weather_merged[WEATHER_COLUMNS].isna().any(axis=1).sum())
        correlation = weather_merged[MONTHLY_FEATURE_COLUMNS + WEATHER_COLUMNS].corr()

    base = weather_merged if weather_merged is not None else daily_stats
    base = base.copy()
    base["month"] = base["date"].dt.month
    monthly_stats = base.groupby("month")[MONTHLY_FEATURE_COLUMNS].median().reset_index()

    return {
        "daily_stats": daily_stats,
        "weather_merged": weather_merged,
        "monthly_stats": monthly_stats,
        "correlation": correlation,
        "weather_missing_dates": weather_missing_dates,
    }
