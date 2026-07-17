"""
Taiwan Spotify Market Explorer
非官方作品。以台灣 Spotify 每日榜單為基礎的互動式市場探索工具。

執行方式：streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_pipeline import (  # noqa: E402
    build_monthly_stats,
    build_song_level_stats,
    classify_success_types,
    load_chart_data,
    validate_chart_schema,
)
from modeling import (  # noqa: E402
    CONFIRMED_CLUSTER_NAMES,
    SIMULATOR_FEATURE_COLUMNS,
    compute_pca_projection,
    fit_and_name_audio_clusters,
    score_audio_features,
    train_success_model,
)
from charts import (  # noqa: E402
    best_rank_histogram,
    cluster_radar_chart,
    days_on_chart_histogram,
    feature_contribution_chart,
    monthly_trend_chart,
    pca_scatter_chart,
    success_type_by_cluster_chart,
    success_type_distribution_chart,
    weather_correlation_heatmap,
)

DATA_DIR = Path(__file__).parent / "data"
CHART_DATA_PATH = DATA_DIR / "tw_chart_data.csv"
WEATHER_DATA_PATH = DATA_DIR / "weather_all.csv"
CANDIDATE_ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "candidate_songs.csv"

st.set_page_config(
    page_title="Taiwan Spotify Market Explorer",
    page_icon="🎧",
    layout="wide",
)


# ---------------------------------------------------------------------------
# 資料載入（快取，只在檔案沒變時重新計算一次）
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="讀取並整理台灣榜單資料中...")
def get_base_data():
    """讀取原始榜單 → 驗證 → 聚合成歌曲層級 → 用全部資料分類成功型態 → 分群 → PCA。

    回傳的 thresholds 是「固定門檻」，之後任何日期區間篩選都只會篩選顯示範圍，
    不會重新計算這裡的門檻（對應已確認的分析邏輯：門檻固定用全部資料算一次）。

    分群同理：cluster_name 只在這裡算一次（用全部530首歌中，排除 instrumentalness>=0.1
    離群值後的525首去訓練），之後的篩選一樣只影響顯示範圍，不會重新分群。
    """
    raw = load_chart_data(str(CHART_DATA_PATH), country="TW")
    validation = validate_chart_schema(raw)
    hot_stats = build_song_level_stats(raw)
    classified, thresholds = classify_success_types(hot_stats)

    cluster_bundle = fit_and_name_audio_clusters(classified)
    cluster_bundle = compute_pca_projection(cluster_bundle)
    model_bundle = train_success_model(cluster_bundle.hot_stats_clustered)

    # 把 cluster_name / PC1 / PC2 併回主表；被排除的離群曲目（器樂比例過高）標記清楚，不留空白造成誤解
    merged = classified.merge(
        cluster_bundle.hot_stats_clustered[["title", "artist", "cluster_name", "PC1", "PC2"]],
        on=["title", "artist"],
        how="left",
    )
    merged["cluster_name"] = merged["cluster_name"].fillna("未分群（器樂比例過高，已排除）")

    return merged, thresholds, validation, cluster_bundle, model_bundle, raw


@st.cache_data(show_spinner="讀取天氣資料中...")
def get_weather_data():
    """讀取天氣資料，檔案不存在時回傳 None（讓 Seasonality 頁面優雅降級，不崩潰）。"""
    if not WEATHER_DATA_PATH.exists():
        return None
    weather = pd.read_csv(WEATHER_DATA_PATH)
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    return weather


@st.cache_data(show_spinner="讀取候選歌曲清單中...")
def get_candidate_songs():
    """讀取離線腳本產生的候選歌曲精簡檔，檔案不存在時回傳 None。

    絕對不會在這裡重新讀取/評分 Spotify12MSongs.csv 原始檔——那是
    scripts/build_candidate_finder_artifact.py 的離線工作，跟 App 執行時機完全分開。
    """
    if not CANDIDATE_ARTIFACT_PATH.exists():
        return None
    df = pd.read_csv(CANDIDATE_ARTIFACT_PATH)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# 錯誤處理：必要資料檔案不存在時，清楚提示而不是讓 App 崩潰
# ---------------------------------------------------------------------------
if not CHART_DATA_PATH.exists():
    st.error(
        "找不到必要資料檔案 `data/TopSpotifySongsin73Countries.csv`。\n\n"
        "請將該檔案放進 `data/` 資料夾後重新整理頁面。"
    )
    st.stop()

try:
    hot_stats, thresholds, validation, cluster_bundle, model_bundle, raw_chart_data = get_base_data()
except Exception as e:  # noqa: BLE001 — 這裡刻意攔截所有例外，因為資料格式問題有很多種可能
    st.error(f"資料處理過程發生錯誤，App 無法啟動：\n\n`{e}`")
    st.stop()

if not validation["is_valid"]:
    st.error("資料驗證未通過，請檢查來源檔案：\n\n" + "\n".join(validation["issues"]))
    st.stop()

weather_data = get_weather_data()  # None 如果檔案不存在，Seasonality 頁面會自己處理這個狀況
candidate_songs = get_candidate_songs()  # None 如果 artifact 還沒產生，Candidate Finder 會自己處理這個狀況


# ---------------------------------------------------------------------------
# 側邊欄導覽
# ---------------------------------------------------------------------------
st.sidebar.title("🎧 Taiwan Spotify\nMarket Explorer")
st.sidebar.caption("非官方作品，僅供探索與分析用途")

page = st.sidebar.radio(
    "頁面",
    [
        "Market Overview",
        "Pattern Explorer",
        "Audio Feature Simulator",
        "Seasonality",
        "Candidate Finder",
    ],
)

if validation["warnings"]:
    with st.sidebar.expander(f"⚠️ 資料品質提醒（{len(validation['warnings'])}項）"):
        for w in validation["warnings"]:
            st.write(f"- {w}")


# ---------------------------------------------------------------------------
# Market Overview
# ---------------------------------------------------------------------------
def render_market_overview(hot_stats: pd.DataFrame):
    st.title("Market Overview")
    st.caption(
        "以「最高排名」與「上榜天數」定義的成功型態，是分析用的定義，"
        "不代表營收或商業成功。詳見 About / Methodology。"
    )

    # --- 互動：日期區間 + 成功型態篩選 ---
    st.subheader("篩選條件")
    col_date, col_type, col_cluster = st.columns([2, 2, 2])

    min_date = hot_stats["first_date"].min().date()
    max_date = hot_stats["last_date"].max().date()

    with col_date:
        date_range = st.date_input(
            "日期區間（篩選在此區間內有上榜紀錄的歌曲）",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    with col_type:
        selected_types = st.multiselect(
            "成功型態",
            options=["Evergreen", "Climber", "Midrunner", "Viral", "Others"],
            default=["Evergreen", "Climber", "Midrunner", "Viral", "Others"],
        )

    cluster_options = list(CONFIRMED_CLUSTER_NAMES.values()) + ["未分群（器樂比例過高，已排除）"]
    with col_cluster:
        selected_clusters = st.multiselect(
            "音訊風格族群",
            options=cluster_options,
            default=cluster_options,
            help="K-Means 分群結果，非正式音樂 genre，僅供風格輪廓參考。"
            "「未分群」是器樂比例過高、不納入分群模型的少數歌曲。",
        )

    # 日期區間篩選：只篩「顯示哪些歌」，不影響成功型態門檻（門檻已在 get_base_data 固定）
    if len(date_range) == 2:
        start, end = date_range
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        mask_date = (hot_stats["last_date"] >= start) & (hot_stats["first_date"] <= end)
    else:
        mask_date = pd.Series(True, index=hot_stats.index)

    mask_type = hot_stats["success_type"].isin(selected_types) if selected_types else pd.Series(
        True, index=hot_stats.index
    )
    mask_cluster = (
        hot_stats["cluster_name"].isin(selected_clusters)
        if selected_clusters
        else pd.Series(True, index=hot_stats.index)
    )

    filtered = hot_stats[mask_date & mask_type & mask_cluster]

    if filtered.empty:
        st.warning("目前篩選條件下沒有任何歌曲，請放寬篩選範圍。")
        return

    st.divider()

    # --- KPI ---
    st.subheader("關鍵指標")
    st.caption(
        f"觀察期間：**{start.date()} ~ {end.date()}**"
        "（使用者選擇的日期區間；歌曲本身的上榜生命週期可能更長，篩選只決定哪些歌曲被納入統計）"
    )
    k1, k2, k3 = st.columns(3)
    k1.metric("去重後歌曲數", f"{len(filtered)}")
    success_rate = filtered["success_class"].mean()
    k2.metric("成功歌曲比例", f"{success_rate:.1%}")
    k3.metric("中位上榜天數", f"{filtered['total_days'].median():.0f} 天")

    st.divider()

    # --- 圖表 ---
    st.subheader("成功型態與上榜表現分布")
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(success_type_distribution_chart(filtered), use_container_width=True)
    c2.plotly_chart(days_on_chart_histogram(filtered), use_container_width=True)
    c3.plotly_chart(best_rank_histogram(filtered), use_container_width=True)

    st.divider()

    # --- 動態洞察（2-3句，全部從 filtered 現場計算，沒有寫死數字）---
    st.subheader("重點洞察")
    type_counts = filtered["success_type"].value_counts()
    non_others_counts = type_counts.drop("Others", errors="ignore")

    insight_lines = [
        f"在目前篩選範圍內共有 **{len(filtered)}** 首去重歌曲，"
        f"其中 **{success_rate:.1%}** 屬於 Evergreen / Climber / Midrunner / Viral 這四種成功型態之一。",
    ]

    if not non_others_counts.empty:
        top_type = non_others_counts.idxmax()
        top_type_pct = non_others_counts.max() / len(filtered)
        insight_lines.append(
            f"四種成功型態中最常見的是 **{top_type}**，佔篩選範圍的 **{top_type_pct:.1%}**"
            f"（Others 代表未落入任何成功型態，不計入比較）。"
        )
    else:
        insight_lines.append("篩選範圍內目前沒有歌曲落入任何一種成功型態。")

    insight_lines.append(
        f"歌曲上榜天數中位數為 **{filtered['total_days'].median():.0f} 天**，"
        f"最佳排名中位數為第 **{filtered['best_rank'].median():.0f}** 名。"
    )
    for line in insight_lines:
        st.markdown(f"- {line}")


# ---------------------------------------------------------------------------
# Pattern Explorer
# ---------------------------------------------------------------------------
def render_pattern_explorer(hot_stats: pd.DataFrame, cluster_bundle):
    st.title("Pattern Explorer")
    st.caption(
        "音訊風格族群是 K-Means 分群結果，用來描述歌曲的音訊特徵輪廓，"
        "不是正式的音樂 genre 分類。"
    )

    st.subheader("篩選條件")
    row1_c1, row1_c2 = st.columns(2)
    row2_c1, row2_c2, row2_c3 = st.columns(3)

    all_types = ["Evergreen", "Climber", "Midrunner", "Viral", "Others"]
    cluster_options = list(CONFIRMED_CLUSTER_NAMES.values()) + ["未分群（器樂比例過高，已排除）"]

    with row1_c1:
        selected_types = st.multiselect("成功型態", options=all_types, default=all_types, key="pe_types")
    with row1_c2:
        selected_clusters = st.multiselect(
            "音訊風格族群", options=cluster_options, default=cluster_options, key="pe_clusters"
        )

    min_days, max_days = int(hot_stats["total_days"].min()), int(hot_stats["total_days"].max())
    min_rank, max_rank = int(hot_stats["best_rank"].min()), int(hot_stats["best_rank"].max())

    with row2_c1:
        days_range = st.slider("上榜天數區間", min_value=min_days, max_value=max_days, value=(min_days, max_days))
    with row2_c2:
        rank_range = st.slider("最佳排名區間", min_value=min_rank, max_value=max_rank, value=(min_rank, max_rank))
    with row2_c3:
        search_query = st.text_input("搜尋歌曲或藝人", value="", placeholder="輸入關鍵字...")

    mask = (
        hot_stats["success_type"].isin(selected_types)
        & hot_stats["cluster_name"].isin(selected_clusters)
        & hot_stats["total_days"].between(days_range[0], days_range[1])
        & hot_stats["best_rank"].between(rank_range[0], rank_range[1])
    )
    if search_query.strip():
        q = search_query.strip().lower()
        mask &= (
            hot_stats["title"].str.lower().str.contains(q, na=False)
            | hot_stats["artist"].str.lower().str.contains(q, na=False)
        )

    filtered = hot_stats[mask]

    if filtered.empty:
        st.warning("目前篩選條件下沒有任何歌曲，請放寬篩選範圍。")
        return

    st.caption(f"目前篩選出 {len(filtered)} 首歌曲（總數 {len(hot_stats)} 首）")
    st.divider()

    st.subheader("成功型態 × 音訊風格族群")
    st.plotly_chart(success_type_by_cluster_chart(filtered), use_container_width=True)

    st.divider()

    st.subheader("音訊風格族群特徵輪廓")
    st.caption("這張雷達圖顯示的是四個族群本身的整體特徵輪廓（固定不受篩選影響），幫助理解每個族群的音樂性格。")
    st.plotly_chart(
        cluster_radar_chart(cluster_bundle.cluster_stats, CONFIRMED_CLUSTER_NAMES),
        use_container_width=True,
    )

    st.divider()

    st.subheader("PCA 2D 散點圖")
    pca_df = filtered[filtered["cluster_name"] != "未分群（器樂比例過高，已排除）"]
    if pca_df.empty:
        st.info("目前篩選範圍內沒有已分群的歌曲可顯示於 PCA 散點圖。")
    else:
        st.plotly_chart(
            pca_scatter_chart(pca_df, cluster_bundle.pca_explained_variance),
            use_container_width=True,
        )

    st.divider()

    st.subheader("歌曲明細")
    display_cols = ["title", "artist", "best_rank", "total_days", "success_type", "cluster_name"]
    display_df = filtered[display_cols].rename(
        columns={
            "title": "歌名",
            "artist": "藝人",
            "best_rank": "最佳排名",
            "total_days": "上榜天數",
            "success_type": "成功型態",
            "cluster_name": "音訊風格族群",
        }
    )
    st.dataframe(
        display_df.sort_values("最佳排名"),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Audio Feature Simulator
# ---------------------------------------------------------------------------
def render_audio_feature_simulator(model_bundle):
    st.title("Audio Feature Simulator")

    st.info(
        "這個分數反映輸入音訊特徵與歷史上榜成功型態的相似程度，適合用於探索與形成假設，"
        "**不是**「這首歌有幾成機率會成功」的保證預測。",
        icon="ℹ️",
    )

    with st.expander("模型限制（建議先看過再操作）", expanded=False):
        st.markdown(
            f"- 這是 **Logistic Regression 探索性模型**，訓練樣本僅 {model_bundle.n_train} 首"
            f"（測試集 {model_bundle.n_test} 首），樣本數不大，結果會有一定波動。\n"
            f"- 測試集 Accuracy 約 **{model_bundle.accuracy:.1%}**、ROC AUC 約 **{model_bundle.auc:.3f}**，"
            "僅略高於隨機猜測，代表音訊特徵本身對「是否成功」的解釋力有限。\n"
            "- 模型只反映音訊特徵與既有榜單成功型態之間的**關聯**，不能解讀成因果關係。\n"
            "- `success_class` 是用最佳排名與上榜天數衍生的分析定義，不是營收或播放收益。"
        )

    st.subheader("輸入音訊特徵")
    col1, col2 = st.columns(2)

    with col1:
        danceability = st.slider("Danceability（舞動性）", 0.0, 1.0, 0.65, 0.01)
        energy = st.slider("Energy（能量）", 0.0, 1.0, 0.6, 0.01)
        valence = st.slider("Valence（情緒正向程度）", 0.0, 1.0, 0.5, 0.01)
        acousticness = st.slider("Acousticness（原聲程度）", 0.0, 1.0, 0.2, 0.01)
    with col2:
        speechiness = st.slider("Speechiness（口白程度）", 0.0, 1.0, 0.06, 0.01)
        instrumentalness = st.slider("Instrumentalness（器樂程度）", 0.0, 1.0, 0.0, 0.01)
        st.caption("Duration（歌曲長度）")
        d_col1, d_col2 = st.columns(2)
        minutes = d_col1.number_input("分鐘", min_value=0, max_value=15, value=3, step=1)
        seconds = d_col2.number_input("秒", min_value=0, max_value=59, value=30, step=1)

    duration_ms = (minutes * 60 + seconds) * 1000

    input_features = {
        "danceability": danceability,
        "energy": energy,
        "valence": valence,
        "acousticness": acousticness,
        "speechiness": speechiness,
        "instrumentalness": instrumentalness,
        "duration_ms": duration_ms,
    }

    result = score_audio_features(input_features, model_bundle)

    st.divider()
    st.subheader("探索性結果")

    k1, k2 = st.columns(2)
    k1.metric(
        "模型預估成功分數（探索性）",
        f"{result['score']:.1%}",
        help="Logistic Regression 對「是否落入四種成功型態之一」的預測機率，不是保證。",
    )
    k2.metric(
        "相對於訓練資料歌曲的 Percentile",
        f"第 {result['percentile']:.0f} 百分位",
        help=f"與訓練這個模型的 {model_bundle.n_train + model_bundle.n_test} 首歌相比，分數輸給多少比例的歌曲。",
    )

    st.plotly_chart(feature_contribution_chart(result["contributions"]), use_container_width=True)

    st.caption(
        "貢獻值是「標準化後的輸入值 × 模型係數」，只能看出方向與相對大小，"
        "不能拆解成「每個特徵各貢獻幾%的成功機率」這種精確歸因。"
    )


# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------
def render_seasonality(raw_chart_data: pd.DataFrame, weather_data: pd.DataFrame | None):
    st.title("Seasonality")
    st.caption(
        "這裡呈現的是「月份季節性」：把所有年份的資料依日曆月份（1-12月）合併後取中位數，"
        "代表平均季節樣貌，不是單一年度的走勢；與下方「每日天氣相關性」是兩件不同的事，"
        "相關不代表因果。"
    )

    monthly = build_monthly_stats(raw_chart_data, weather_df=weather_data)

    st.subheader("月份季節性趨勢比較")
    all_features = ["danceability", "energy", "valence", "acousticness", "liveness", "tempo"]
    selected = st.multiselect(
        "選擇要比較的音訊特徵（建議 2 個，方便比較趨勢）",
        options=all_features,
        default=["danceability", "valence"],
    )
    if len(selected) == 0:
        st.info("請至少選擇一個特徵。")
    else:
        st.plotly_chart(monthly_trend_chart(monthly["monthly_stats"], selected), use_container_width=True)

    st.divider()

    st.subheader("天氣與音訊特徵相關性")
    if weather_data is None:
        st.info(
            "找不到 `data/weather_all.csv`，這個區塊會自動隱藏，不影響其他功能。"
            "如果想看天氣相關性，把該檔案放進 `data/` 資料夾後重新整理頁面即可。"
        )
    else:
        if monthly["weather_missing_dates"]:
            st.warning(f"有 {monthly['weather_missing_dates']} 個日期缺少對應天氣資料，已在相關矩陣計算中自動排除。")
        st.plotly_chart(weather_correlation_heatmap(monthly["correlation"]), use_container_width=True)
        st.caption(
            "這張圖只顯示統計上的相關係數，不能直接推論「天氣造成音樂偏好改變」——"
            "可能有其他共同因素（例如季節性發片檔期）同時影響兩者。"
        )


# ---------------------------------------------------------------------------
# Candidate Finder
# ---------------------------------------------------------------------------
def render_candidate_finder(candidate_songs: pd.DataFrame | None):
    st.title("Candidate Finder")

    if candidate_songs is None:
        st.info(
            "還沒有產生候選歌曲清單。這個頁面讀取的是離線腳本處理過的精簡結果，"
            "不會在網頁裡直接處理120萬筆原始資料。\n\n"
            "請在專案根目錄執行：\n\n"
            "`python scripts/build_candidate_finder_artifact.py`\n\n"
            "（需要 `data/Spotify12MSongs.csv` 存在）執行完成後重新整理這個頁面即可。"
        )
        return

    st.warning(
        "**這是目前限制最大的一個功能，請謹慎解讀。** 這裡的模型是用530首「曾在台灣上榜」的歌"
        "訓練出來的，套用到120萬首風格差異極大的歌曲庫時，等於是要求模型對它完全沒見過的音樂類型"
        "做判斷。分數只反映音訊特徵組合與歷史成功型態的相似程度，**跟這首歌實際會不會紅沒有可靠的因果關係**，"
        "請把這裡的結果當成「可以進一步人工聽感篩選的候選池」，不是排行榜。",
        icon="⚠️",
    )

    st.subheader("篩選條件")
    col1, col2 = st.columns(2)

    valid_dates = candidate_songs["release_date"].dropna()
    with col1:
        if not valid_dates.empty:
            min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
            date_range = st.date_input("發行日期區間", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        else:
            date_range = None
    with col2:
        exclude_charted = st.checkbox("排除台灣已上榜過的歌（依歌名比對，僅供參考）", value=True)

    mask = pd.Series(True, index=candidate_songs.index)
    if date_range is not None and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        mask &= candidate_songs["release_date"].between(start, end) | candidate_songs["release_date"].isna()
    if exclude_charted:
        mask &= ~candidate_songs["already_charted_in_tw"]

    filtered = candidate_songs[mask].sort_values("score", ascending=False)

    st.caption(f"目前候選池：{len(filtered)} 首（來自離線腳本預先篩選出的 {len(candidate_songs)} 首高分候選）")

    display_df = filtered[
        ["name", "artists", "release_date", "score", "percentile", "danceability", "energy", "valence"]
    ].rename(
        columns={
            "name": "歌名",
            "artists": "藝人",
            "release_date": "發行日期",
            "score": "探索性分數",
            "percentile": "Percentile",
            "danceability": "舞動性",
            "energy": "能量",
            "valence": "情緒正向",
        }
    )
    st.dataframe(display_df.head(200), use_container_width=True, hide_index=True)


if page == "Market Overview":
    render_market_overview(hot_stats)
elif page == "Pattern Explorer":
    render_pattern_explorer(hot_stats, cluster_bundle)
elif page == "Audio Feature Simulator":
    render_audio_feature_simulator(model_bundle)
elif page == "Seasonality":
    render_seasonality(raw_chart_data, weather_data)
elif page == "Candidate Finder":
    render_candidate_finder(candidate_songs)
else:
    st.title(page.split("（")[0])
    st.info("這個頁面還在開發中，會在對應的 Phase 完成後上線。")
