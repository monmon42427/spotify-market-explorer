# Taiwan Spotify Market Explorer 🎧

以台灣 Spotify 每日榜單為基礎的互動式市場探索工具，讓內容與行銷人員從榜單表現、音訊特徵、成功型態與月份趨勢中形成假設，並可透過探索性機器學習模型理解一組音訊特徵和歷史成功歌曲的相似程度。

**這不是歌曲一定會成功的預測器，也不是 Spotify 官方工具。**

🔗 **Live demo**：https://spotify-market-explorer-jameslin0717.streamlit.app/

## 功能

- **Market Overview** — 觀察期間、去重歌曲數、成功歌曲比例、成功型態分布、上榜天數與最佳排名分布，全部依篩選條件即時計算
- **Pattern Explorer** — K-Means 音訊風格族群、PCA 2D 投影、成功型態 × 音訊風格族群交叉分布、可篩選排序的歌曲明細表
- **Audio Feature Simulator** — 手動輸入 7 個音訊特徵，即時取得 Logistic Regression 探索性分數、percentile、特徵貢獻方向
- **Seasonality** — 月份季節性趨勢比較、天氣與音訊特徵相關矩陣（選配，需要 `weather_all.csv`）
- **Candidate Finder** — 用同一個模型對外部歌曲庫評分，找出探索性候選歌曲（選配，資料需離線預先處理）

## 技術棧

Python・Streamlit・Plotly・scikit-learn（K-Means、PCA、Logistic Regression）・pandas

## 本機執行

```bash
git clone <your-repo-url>
cd spotify-market-explorer
pip install -r requirements.txt
streamlit run app.py
```

## 專案結構

```text
spotify-market-explorer/
├── app.py                              # Streamlit 主程式
├── src/
│   ├── data_pipeline.py                # 讀取、驗證、聚合、成功型態分類、月份/天氣統計
│   ├── modeling.py                     # K-Means 分群、PCA、Logistic Regression
│   └── charts.py                       # 所有 Plotly 圖表
├── scripts/
│   ├── extract_tw_subset.py            # 從73國原始資料篩出台灣資料（部署用小檔案）
│   └── build_candidate_finder_artifact.py  # 離線對大型歌曲庫評分，產生 Candidate Finder 用的精簡清單
├── data/                                # tw_chart_data.csv、weather_all.csv
├── artifacts/                           # candidate_songs.csv（離線腳本產生）
└── requirements.txt
```

## 方法與限制

### 資料來源

台灣 Spotify 每日榜單（2023-10-18 ~ 2025-06-11），共 530 首去重歌曲。天氣資料為選配，涵蓋同一期間的每日氣溫、降雨量、日照時數。

### 成功型態定義

依「最高排名」與「上榜天數」的分位數門檻，把歌曲分成 Evergreen（長紅）、Climber（穩定攀升）、Midrunner（中期）、Viral（爆發型）、Others（未落入前四者）五種型態。**這是分析用的定義，不是營收或播放收益，也不代表商業成功。** 門檻固定用全部資料計算一次，網頁上的日期區間篩選只影響顯示範圍，不會重新計算門檻或改變任何一首歌的型態標籤。

### 音訊風格族群

K-Means（k=4）分群，剔除 instrumentalness ≥ 0.1 的離群值（5首）後，對 9 個音訊特徵分群。四個族群的中文名稱是看過實際 centroid 特徵值後人工確認命名，**不是正式的音樂 genre 分類**。

### 模型限制（請務必詳讀）

Logistic Regression 探索性模型，7 個音訊特徵，訓練/測試切分 `test_size=0.3, random_state=17`：

- 訓練樣本僅 367 首（測試集 158 首），樣本數不大
- 測試集 Accuracy 約 55.7%、ROC AUC 約 0.563，僅略高於隨機猜測
- 模型只反映音訊特徵與既有榜單成功型態之間的**關聯**，不能解讀成因果關係
- Candidate Finder 把這個模型套用到風格差異極大的外部歌曲庫，可靠度又更低一層，結果僅供進一步人工篩選參考

## AI 協作說明

本專案的介面開發、重構、除錯與測試由 AI（Claude）協助完成；需求定義、分析方法選擇、模型結果驗證與商業判斷由專案作者負責。過程中的階段性紀錄見 `AI_COLLABORATION_LOG.md`。
