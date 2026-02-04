import yfinance as yf
import pandas as pd
import json
import os
import time

def batch_download():
    with open("TW-Quant-Shioaji/components.json", "r") as f:
        symbols = json.load(f)
    
    os.makedirs("TW-Quant-Shioaji/data/batch", exist_ok=True)
    
    print(f"開始下載 {len(symbols)} 支標的歷史數據...")
    
    for i, sym in enumerate(symbols):
        path = f"TW-Quant-Shioaji/data/batch/{sym}.csv"
        if os.path.exists(path):
            continue
            
        try:
            df = yf.download(f"{sym}.TW", period='2y', progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.to_csv(path)
                print(f"[{i+1}/{len(symbols)}] {sym} 下載成功")
            else:
                print(f"[{i+1}/{len(symbols)}] {sym} 無數據")
            time.sleep(1) # 避免 API 限制
        except Exception as e:
            print(f"下載 {sym} 失敗: {e}")

if __name__ == "__main__":
    batch_download()
