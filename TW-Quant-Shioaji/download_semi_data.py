import yfinance as yf
import json
import os
from pathlib import Path

# 配置
SEMI_LIST_FILE = "TW-Quant-Shioaji/semi_10.json"
DATA_DIR = "TW-Quant-Shioaji/data/batch_semi"

def download_data():
    if not os.path.exists(SEMI_LIST_FILE):
        print(f"❌ 找不到清單: {SEMI_LIST_FILE}")
        return

    with open(SEMI_LIST_FILE, "r") as f:
        symbols = json.load(f)

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 開始下載 {len(symbols)} 檔半導體股票數據...")
    
    for sym in symbols:
        yf_sym = f"{sym}.TW"
        print(f"  📥 下載 {yf_sym}...")
        df = yf.download(yf_sym, period='3y', interval='1d', progress=False)
        if not df.empty:
            # 展平 MultiIndex
            import pandas as pd
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 儲存
            csv_path = os.path.join(DATA_DIR, f"{sym}.csv")
            df.to_csv(csv_path)
            print(f"    ✅ 已存至 {csv_path}")
        else:
            print(f"    ❌ {yf_sym} 無數據")

if __name__ == "__main__":
    download_data()
