"""
離線腳本：從 TopSpotifySongsin73Countries.csv（498MB，73國）篩出台灣的資料，
存成一份小檔案，供部署到 Streamlit Community Cloud 使用（GitHub 放不下498MB的檔案）。

這不改變任何分析邏輯——data_pipeline.py 的 load_chart_data() 完全不用改，
只是讀取的來源檔案從「全部73國」換成「已經篩好的台灣資料」，篩選結果完全一樣。

使用方式：
    python scripts/extract_tw_subset.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_PATH = PROJECT_ROOT / "data" / "TopSpotifySongsin73Countries.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "tw_chart_data.csv"

# 跟 data_pipeline.py 的 REQUIRED_RAW_COLUMNS 完全一致，維持單一事實來源不重複定義容易漏改，
# 這裡直接 import 過來用。
import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data_pipeline import REQUIRED_RAW_COLUMNS  # noqa: E402


def main():
    if not SOURCE_PATH.exists():
        print(f"錯誤：找不到 {SOURCE_PATH}")
        sys.exit(1)

    print(f"讀取 {SOURCE_PATH.name}（498MB，會花一點時間）...")
    df = pd.read_csv(SOURCE_PATH, usecols=REQUIRED_RAW_COLUMNS)
    print(f"  讀取完成，共 {len(df)} 筆（73國）")

    tw_only = df[df["country"] == "TW"]
    print(f"  篩出 TW：{len(tw_only)} 筆")

    tw_only.to_csv(OUTPUT_PATH, index=False)
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"完成！已輸出到 {OUTPUT_PATH}（{size_mb:.1f} MB，GitHub 放得下）")


if __name__ == "__main__":
    main()
