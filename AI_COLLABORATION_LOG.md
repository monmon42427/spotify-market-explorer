# AI Collaboration Log

這份紀錄說明本專案如何透過 AI（Claude）協作完成，記錄每個階段 AI 做了什麼、我做了什麼決定、如何驗證。

## Project Roles

**Human owner（我）**：定義產品需求（`PROJECT_SPEC.md`）、稽核原始分析程式並寫成 `MODEL_AUDIT.md`、確認每個分析邏輯決策（例如成功門檻是否隨篩選重算、cluster命名是否採用）、驗證所有數字、實際操作GitHub與Streamlit Cloud完成部署。

**AI coding partner（Claude）**：協助重構模組、產生Streamlit/Plotly程式、解釋錯誤與提出最小修正、撰寫測試、在我確認之前不擅自更換分析定義或捏造結果。

## Iteration Log

### Iteration 00 — Phase 0 稽核

**Problem**：現有 `original_analysis.py`（約1000行notebook式程式）要重構成可互動的Streamlit網頁，但不知道程式碼裡有哪些技術債、資料是否齊全。

**Prompt / instruction given to AI**：貼上 `CLAUDE_MASTER_PROMPT.md` 全文，要求先做Phase 0稽核，不要直接重寫。

**AI contribution**：實際讀取三份CSV跑出真實統計（TW榜單29,157列聚合成530首歌、天氣資料涵蓋完整期間、大型歌曲庫1,204,025列），比對程式碼找出3個技術風險：K-Means群集編號跟視覺化標題的k值對不上（L313-340確實寫錯）、大檔案+notebook式全域變數、多處硬編碼百分比。

**Human decision**：確認稽核結果，同意繼續往下做，並要求Claude在不大幅更動分析邏輯的前提下可以主動指出更好的做法。

**Validation**：親自核對Claude回報的筆數（530首、29157列等）跟自己記憶中的資料量級是否合理。

---

### Iteration 01 — 資料流程模組化

**Problem**：原程式的資料清理、聚合、成功型態分類全部混在一起、用全域變數，無法變成可重複呼叫、可快取的Streamlit函式。

**AI contribution**：拆成 `load_chart_data()`、`validate_chart_schema()`、`build_song_level_stats()`、`classify_success_types()` 四個純函式，並在每個函式後面實際跑真實資料驗證數字（530首歌、成功型態分布Others 275/Evergreen 104/Midrunner 71/Climber 55/Viral 25）。

**Human decision**：被問到「日期區間篩選時，成功門檻要不要重新計算」——選擇**門檻固定用全部資料算一次，篩選只影響顯示範圍**，理由是避免同一首歌在不同篩選條件下型態標籤跳動。

**Validation**：Claude寫了一個模擬測試，證明篩選後子集重新套用固定門檻，跟全集分類結果一致。

---

### Iteration 02 — 音訊風格族群分群

**Problem**：原程式K-Means分群的中文命名（如「高能量舞曲型」）是人工寫死對應到數字編號 0-3，重新訓練可能編號錯位。

**AI contribution**：分群邏輯完整複製原程式（9個特徵、剔除instrumentalness≥0.1離群值、k=4、random_state=42），但**不**直接沿用命名，而是先跑出真實centroid數值、畫成雷達圖，人工比對後才確認可以沿用原命名。

**Human decision**：看過雷達圖後確認四個族群命名合理，同意寫入 `CONFIRMED_CLUSTER_NAMES` 常數。

**Issue and correction**：無重大bug，但這個流程本身就是為了避免MODEL_AUDIT.md點出的「群集編號不保證跨版本一致」風險。

---

### Iteration 03 — Logistic Regression 模型與 Simulator

**Problem**：需要重現原程式的成功預測模型，並包裝成使用者可互動的Audio Feature Simulator。

**AI contribution**：逐字比照原程式的資料切分（`test_size=0.3, random_state=17, stratify=Y`），確認沒有啟用原程式裡被註解掉的「試多個random seed挑最高分」的迴圈。訓練後發現實際AUC是0.563，跟MODEL_AUDIT.md記載的「約0.607」有落差。

**Human decision**：確認App要顯示**實際訓練出來的數字（0.563）**，不採用文件裡可能過時的舊數字。

**Validation**：手算一次完整的log-odds→sigmoid轉換過程，跟sklearn的predict_proba結果比對，數字一致（0.5003 vs 0.5003）。

---

### Iteration 04 — Candidate Finder 與模型外推bug

**Problem**：把模型套用到120萬首外部歌曲庫，需要離線腳本產生精簡結果檔。

**Issue and correction**：**第一次跑出來的結果全部飽和在0.9999分**，前幾名都是冷門演奏曲。深入排查後發現：模型訓練資料的instrumentalness全部<0.1，StandardScaler學到的標準差極小，套到高器樂比例的歌曲時z-score被推到130幾個標準差，模型判斷嚴重失真。

**Human decision**：同意修正方案——對候選歌曲庫套用跟訓練資料一致的instrumentalness<0.1範圍限制，讓模型只評分「屬於同一個母體」的歌曲。修正後分數落在合理的0.79-0.92範圍。

**What I learned**：這是一個模型「外推到訓練範圍外」失真的真實案例，也解釋了為什麼Candidate Finder頁面要用比其他頁面更強烈的警語。

---

### Iteration 05 — 部署與GitHub

**Problem**：原始資料檔案498MB/345MB，GitHub免費方案放不下，需要精簡版才能部署到Streamlit Community Cloud。

**AI contribution**：寫 `scripts/extract_tw_subset.py` 把73國資料篩成只有TW的4.2MB小檔案，驗證過換資料來源後所有數字（530首歌、AUC 0.563等）完全一致。

**Human decision**：實際操作GitHub Desktop建repo、commit、push，過程中遇到「Create a New Repository建到空資料夾」的操作失誤，跟AI一起排查（透過「Show in Finder」直接確認GitHub Desktop實際連到哪個路徑）才定位到問題，刪除重建後成功。

**Validation**：實際打開部署後的網址（`spotify-market-explorer-jameslin0717.streamlit.app`），確認5個頁面都正常運作。

---

## Final Evidence Checklist

- [x] 可啟動的 App（本機 + 已部署到 Streamlit Community Cloud）
- [ ] 主要頁面截圖或短影片
- [x] Git commit history（GitHub repo）
- [x] AI Collaboration Log（本檔案）
- [x] 模型評估與限制（README.md「方法與限制」章節、App內每頁的警語）
- [ ] 一段 60–90 秒的面試說法（見下方）

## 60-90 秒面試說法草稿

> 這個專案是把我之前做的一份Spotify台灣市場探索性分析，重構成一個可互動的Streamlit網頁。我用生成式AI協作完成介面開發、重構跟除錯，但需求定義、分析邏輯的每個決策、跟模型結果驗證都是我自己把關的。
>
> 比較值得一提的是，我沒有把模型包裝成「準確的預測器」——訓練出來的Logistic Regression模型AUC只有0.563，我選擇誠實地把這個限制寫進網頁跟README裡，而不是隱藏它。過程中還實際發現並修正了一個模型外推的bug：把模型套用到120萬首風格差異很大的外部歌曲庫時，因為訓練資料的特徵分布太窄，套用時數值被推到超出訓練範圍一百多個標準差，導致分數全部飽和失真，我後來用「限制模型只評分屬於同一母體的歌曲」解決了這個問題。
>
> 這個專案讓我完整跑過一次資料科學專案的工作流程：資料清理、特徵工程、訓練/驗證切分、模型評估、發現並修正bug、最後誠實地把限制包裝進產品裡，並且完成從本機開發到雲端部署的完整流程。
