"""
圖表模組。所有圖表一律用 Plotly（互動、有 hover），不使用 matplotlib/seaborn。

原因：
1. PROJECT_SPEC 要求圖表要有 hover 資訊，matplotlib 產生的是靜態圖片做不到。
2. matplotlib 需要伺服器端安裝中文字型（原程式用 macOS 的 'Heiti TC'，
   部署到 Linux 環境大機率沒有），Plotly 的文字是瀏覽器端渲染，不依賴伺服器字型。

每個函式只負責畫圖，資料的篩選/計算在呼叫端（app.py）完成，圖表函式不碰篩選邏輯。
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 統一色系：Spotify 綠 + 深灰，對應 PROJECT_SPEC 的視覺風格要求
SUCCESS_TYPE_COLORS = {
    "Evergreen": "#1DB954",   # Spotify 綠：長紅型
    "Climber": "#3D8BFD",     # 穩定型
    "Midrunner": "#FCB53B",   # 潛力型
    "Viral": "#E04F5F",       # 爆發型
    "Others": "#B3B3B3",      # 灰：未落入任何成功型態
}

SUCCESS_TYPE_ORDER = ["Evergreen", "Climber", "Midrunner", "Viral", "Others"]


def success_type_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """成功型態分布長條圖。比例與筆數都即時從傳入的 df 計算，不使用硬編碼數字。

    Parameters
    ----------
    df : pd.DataFrame
        需包含 success_type 欄位（通常是已篩選過的 hot_stats 子集）。
    """
    counts = df["success_type"].value_counts()
    counts = counts.reindex(SUCCESS_TYPE_ORDER).fillna(0).astype(int)
    proportions = (counts / counts.sum() * 100).round(1)

    plot_df = pd.DataFrame(
        {
            "success_type": counts.index,
            "count": counts.values,
            "percent": proportions.values,
        }
    )

    fig = px.bar(
        plot_df,
        x="success_type",
        y="count",
        color="success_type",
        color_discrete_map=SUCCESS_TYPE_COLORS,
        text=plot_df["percent"].map(lambda p: f"{p}%"),
        custom_data=["count", "percent"],
    )
    fig.update_traces(
        hovertemplate="成功型態：%{x}<br>歌曲數：%{customdata[0]}<br>佔比：%{customdata[1]}%<extra></extra>"
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="成功型態",
        yaxis_title="歌曲數",
        title=f"成功型態分布（樣本數 n={int(counts.sum())}）",
        margin=dict(t=50, b=10),
    )
    return fig


def days_on_chart_histogram(df: pd.DataFrame, bins: int = 30) -> go.Figure:
    """上榜天數分布直方圖。"""
    median_days = df["total_days"].median()

    fig = px.histogram(
        df,
        x="total_days",
        nbins=bins,
        color_discrete_sequence=["#1DB954"],
    )
    fig.update_traces(
        hovertemplate="上榜天數區間：%{x}<br>歌曲數：%{y}<extra></extra>"
    )
    fig.add_vline(
        x=median_days,
        line_dash="dash",
        line_color="#535353",
        annotation_text=f"中位數：{median_days:.0f} 天",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="上榜天數",
        yaxis_title="歌曲數",
        title=f"上榜天數分布（樣本數 n={len(df)}）",
        margin=dict(t=50, b=10),
    )
    return fig


def best_rank_histogram(df: pd.DataFrame, bins: int = 30) -> go.Figure:
    """最佳排名分布直方圖（數字越小代表名次越高）。"""
    median_rank = df["best_rank"].median()

    fig = px.histogram(
        df,
        x="best_rank",
        nbins=bins,
        color_discrete_sequence=["#3D8BFD"],
    )
    fig.update_traces(
        hovertemplate="最佳排名區間：%{x}<br>歌曲數：%{y}<extra></extra>"
    )
    fig.add_vline(
        x=median_rank,
        line_dash="dash",
        line_color="#535353",
        annotation_text=f"中位數：第 {median_rank:.0f} 名",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="最佳排名（數字越小名次越高）",
        yaxis_title="歌曲數",
        title=f"最佳排名分布（樣本數 n={len(df)}）",
        margin=dict(t=50, b=10),
    )
    return fig


CLUSTER_COLORS = {
    "高能量舞曲型": "#FF8F8F",
    "輕快節奏型": "#84994F",
    "抒情原音型": "#FCB53B",
    "慢節奏情感型": "#93BFC7",
}


def success_type_by_cluster_chart(df: pd.DataFrame) -> go.Figure:
    """成功型態 × 音訊風格族群的分布（堆疊長條圖）。

    Parameters
    ----------
    df : pd.DataFrame
        需包含 success_type 與 cluster_name 兩欄（已排除或已標記未分群的離群曲目）。
    """
    cross = (
        df.groupby(["success_type", "cluster_name"]).size().reset_index(name="count")
    )
    fig = px.bar(
        cross,
        x="success_type",
        y="count",
        color="cluster_name",
        color_discrete_map=CLUSTER_COLORS,
        category_orders={"success_type": SUCCESS_TYPE_ORDER},
        barmode="stack",
    )
    fig.update_traces(
        hovertemplate="成功型態：%{x}<br>音訊風格族群：%{fullData.name}<br>歌曲數：%{y}<extra></extra>"
    )
    fig.update_layout(
        xaxis_title="成功型態",
        yaxis_title="歌曲數",
        legend_title="音訊風格族群",
        title=f"成功型態 × 音訊風格族群分布（樣本數 n={len(df)}）",
        margin=dict(t=50, b=10),
    )
    return fig


def cluster_radar_chart(cluster_stats: pd.DataFrame, cluster_names: dict[int, str]) -> go.Figure:
    """四個音訊風格族群的特徵輪廓雷達圖（0-1 min-max 標準化後比較）。

    Parameters
    ----------
    cluster_stats : pd.DataFrame
        fit_audio_clusters() 回傳的 bundle.cluster_stats（未標準化，index 是 cluster_label）。
    cluster_names : dict[int, str]
        cluster_label -> 中文名稱，用 CONFIRMED_CLUSTER_NAMES。
    """
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(cluster_stats),
        columns=cluster_stats.columns,
        index=cluster_stats.index,
    )

    fig = go.Figure()
    categories = list(scaled.columns)
    for cluster_label in scaled.index:
        name = cluster_names.get(cluster_label, f"Cluster {cluster_label}")
        values = scaled.loc[cluster_label].tolist()
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=name,
                line_color=CLUSTER_COLORS.get(name),
                opacity=0.7,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="音訊風格族群特徵輪廓（各特徵已標準化至 0-1，僅供族群間相對比較）",
        margin=dict(t=50, b=10),
    )
    return fig


def feature_contribution_chart(contrib_df: pd.DataFrame) -> go.Figure:
    """音訊特徵對本次模擬輸入的貢獻方向圖（水平長條，正向/負向用色區分）。"""
    plot_df = contrib_df.sort_values("contribution")
    colors = ["#1DB954" if v > 0 else "#E04F5F" for v in plot_df["contribution"]]

    fig = go.Figure(
        go.Bar(
            x=plot_df["contribution"],
            y=plot_df["feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="特徵：%{y}<br>貢獻值：%{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#535353")
    fig.update_layout(
        xaxis_title="對本次輸入的貢獻（正值推高分數，負值拉低分數）",
        yaxis_title="",
        title="本次輸入的特徵貢獻概覽",
        margin=dict(t=50, b=10, l=10),
    )
    return fig


def pca_scatter_chart(df: pd.DataFrame, explained_variance: tuple[float, float]) -> go.Figure:
    """PCA 2D 散點圖，依音訊風格族群上色。

    Parameters
    ----------
    df : pd.DataFrame
        需包含 PC1、PC2、cluster_name、title、artist 欄位。
    explained_variance : tuple[float, float]
        (PC1解釋變異量, PC2解釋變異量)，會顯示在標題裡提醒使用者這只是部分還原。
    """
    var_sum = sum(explained_variance)
    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="cluster_name",
        color_discrete_map=CLUSTER_COLORS,
        hover_data={"title": True, "artist": True, "cluster_name": True, "PC1": False, "PC2": False},
    )
    fig.update_layout(
        title=(
            f"音訊風格族群 PCA 2D 投影（PC1+PC2 共解釋 {var_sum:.1%} 變異量，"
            f"僅為降維後的近似視覺化，不代表完整資訊）"
        ),
        legend_title="音訊風格族群",
        margin=dict(t=60, b=10),
    )
    return fig


def monthly_trend_chart(monthly_stats: pd.DataFrame, selected_features: list[str]) -> go.Figure:
    """月份季節性趨勢比較圖（z-score 標準化後比較，讓不同量級的特徵可以疊在一起看趨勢）。

    對應 original_analysis.py L793-808（標準化月份折線圖），但改成使用者可自選特徵。

    Parameters
    ----------
    monthly_stats : pd.DataFrame
        build_monthly_stats() 回傳的 monthly_stats（month 1-12 的中位數音訊特徵）。
    selected_features : list[str]
        使用者想比較的特徵（通常 2 個，但函式本身不限制數量）。
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(monthly_stats[selected_features])
    scaled_df = pd.DataFrame(scaled_values, columns=selected_features)
    scaled_df["month"] = monthly_stats["month"].values

    fig = go.Figure()
    palette = ["#53629E", "#FFA239", "#8BAE66", "#8CA9FF", "#92487A", "#B87C4C"]
    for i, feature in enumerate(selected_features):
        fig.add_trace(
            go.Scatter(
                x=scaled_df["month"],
                y=scaled_df[feature],
                mode="lines+markers",
                name=feature,
                line_color=palette[i % len(palette)],
                hovertemplate=f"{feature}<br>月份：%{{x}}<br>標準化中位數：%{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis=dict(title="月份（跨所有年份合併，代表平均季節性樣貌，不是單一年度趨勢）", tickmode="linear", tick0=1, dtick=1),
        yaxis_title="標準化中位數（z-score）",
        title="音訊特徵月份季節性趨勢",
        margin=dict(t=50, b=10),
    )
    return fig


def weather_correlation_heatmap(correlation: pd.DataFrame) -> go.Figure:
    """音訊特徵 x 天氣的相關矩陣熱力圖。只呈現相關性，不代表因果關係。"""
    fig = go.Figure(
        go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.index,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=correlation.round(2).values,
            texttemplate="%{text}",
            hovertemplate="%{y} x %{x}：%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="音訊特徵 × 天氣相關矩陣（僅為相關性，不代表因果關係）",
        margin=dict(t=50, b=10),
    )
    return fig
