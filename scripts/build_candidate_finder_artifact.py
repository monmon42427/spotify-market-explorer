"""
離線腳本：對 Spotify12MSongs.csv（約120萬首歌）用 Logistic Regression 模型評分，
產生一份精簡的候選歌曲清單，供 app.py 的 Candidate Finder 頁面讀取。

為什麼要離線跑，不能放進 app.py：
- Spotify12MSongs.csv 完整讀取要 17 秒、佔 735MB 記憶體，Streamlit 每次互動都重跑會拖垮 App。
- data/README.md 也明確建議「不建議在每次 App 啟動時重新預測」。

使用方式：
    python scripts/build_candidate_finder_artifact.py

輸出：
    artifacts/candidate_songs.csv （只保留分數最高的一批候選歌曲，檔案很小，可以放進 GitHub）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_pipeline import (  # noqa: E402
    build_song_level_stats,
    classify_success_types,
    load_chart_data,
)
from modeling import fit_audio_clusters, train_success_model  # noqa: E402

CHART_DATA_PATH = PROJECT_ROOT / "data" / "tw_chart_data.csv"
LARGE_LIBRARY_PATH = PROJECT_ROOT / "data" / "Spotify12MSongs.csv"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "candidate_songs.csv"

# 只保留分數最高的這麼多首，讓 artifact 檔案維持精簡（可放進 GitHub），
# 同時在 App 裡依發行日期／音訊特徵篩選時還有足夠的候選池可以篩。
TOP_N = 2000

# 大型歌庫需要用到的欄位（對照 data/README.md）
LARGE_LIBRARY_COLUMNS = [
    "id",
    "name",
    "artists",
    "release_date",
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "duration_ms",
]

MODEL_FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "duration_ms",
]


def main():
    t0 = time.time()

    # --- 1. 用跟 App 完全一樣的流程訓練模型（重用 src/ 模組，不重寫邏輯） ---
    print("訓練模型（用台灣榜單資料）...")
    if not CHART_DATA_PATH.exists():
        print(f"錯誤：找不到 {CHART_DATA_PATH}，無法訓練模型。")
        sys.exit(1)

    raw = load_chart_data(str(CHART_DATA_PATH), country="TW")
    hot_stats = build_song_level_stats(raw)
    classified, _ = classify_success_types(hot_stats)
    cluster_bundle = fit_audio_clusters(classified)
    model_bundle = train_success_model(cluster_bundle.hot_stats_clustered)
    print(f"  模型訓練完成（n_train={model_bundle.n_train}, AUC={model_bundle.auc:.3f}）")

    # --- 2. 讀取大型歌庫 ---
    print(f"讀取 {LARGE_LIBRARY_PATH.name}（可能需要一段時間）...")
    if not LARGE_LIBRARY_PATH.exists():
        print(f"錯誤：找不到 {LARGE_LIBRARY_PATH}，無法產生候選歌曲清單。")
        sys.exit(1)

    library = pd.read_csv(LARGE_LIBRARY_PATH, usecols=LARGE_LIBRARY_COLUMNS)
    n_raw = len(library)
    print(f"  讀取完成，共 {n_raw} 筆")

    # --- 3. 清理資料：去重、缺值、範圍檢查 ---
    library = library.drop_duplicates(subset=["id"])
    n_after_dedup = len(library)

    library = library.dropna(subset=MODEL_FEATURE_COLUMNS + ["name", "artists"])
    n_after_dropna = len(library)

    library = library[library["duration_ms"] > 0]
    zero_one_cols = ["danceability", "energy", "valence", "acousticness", "speechiness", "instrumentalness"]
    for col in zero_one_cols:
        library = library[library[col].between(0, 1)]
    n_after_range_check = len(library)

    library["release_date"] = pd.to_datetime(library["release_date"], errors="coerce")

    # 重要修正：模型訓練資料（TW榜單歌曲）全部經過 instrumentalness<0.1 篩選，
    # StandardScaler 在這個特徵上學到的標準差非常小。如果直接把高器樂比例（接近純演奏曲）
    # 的歌套進同一個 scaler，z-score 會被推到破百個標準差，模型會給出失真的滿分。
    # 這裡套用跟訓練資料一致的範圍限制，讓模型只評分「屬於同一個母體」的歌曲，
    # 避免對超出訓練範圍的極端值做無意義的外推。
    from modeling import INSTRUMENTALNESS_OUTLIER_THRESHOLD

    n_before_domain_filter = len(library)
    library = library[library["instrumentalness"] < INSTRUMENTALNESS_OUTLIER_THRESHOLD]
    n_after_domain_filter = len(library)
    print(
        f"  套用模型適用範圍限制（instrumentalness<{INSTRUMENTALNESS_OUTLIER_THRESHOLD}）："
        f"{n_before_domain_filter} → {n_after_domain_filter}"
    )

    print(
        f"  資料清理：{n_raw} → 去重後 {n_after_dedup} → 去缺值後 {n_after_dropna} "
        f"→ 範圍檢查後 {n_after_range_check}"
    )

    # --- 4. 用同一個模型評分（向量化運算，120萬筆也很快） ---
    print("計算探索性分數...")
    X = library[MODEL_FEATURE_COLUMNS]
    X_scaled = model_bundle.scaler.transform(X)
    library["score"] = model_bundle.model.predict_proba(X_scaled)[:, 1]
    library["percentile"] = library["score"].rank(pct=True) * 100

    # 標記是否為台灣已上榜過的歌（只比對歌名，不比對藝人——因為兩份資料的 artists 欄位
    # 格式不同：Spotify12MSongs 是 "['Artist']" 這種 list 字串，TW榜單是逗號分隔的原始字串，
    # 直接比對藝人格式永遠對不起來。只比歌名是比較穩健的簡化，這只是輔助資訊，不影響分數。
    tw_titles = set(classified["title"].str.lower())
    library["already_charted_in_tw"] = library["name"].str.lower().isin(tw_titles)

    # --- 5. 只保留分數最高的 TOP_N 首，存成精簡 artifact ---
    top_candidates = library.sort_values("score", ascending=False).head(TOP_N)

    output_cols = [
        "name",
        "artists",
        "release_date",
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "speechiness",
        "instrumentalness",
        "duration_ms",
        "score",
        "percentile",
        "already_charted_in_tw",
    ]
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    top_candidates[output_cols].to_csv(OUTPUT_PATH, index=False)

    print(f"完成！已輸出 {len(top_candidates)} 首候選歌曲到 {OUTPUT_PATH}")
    print(f"檔案大小：{OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    print(f"總耗時：{time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
