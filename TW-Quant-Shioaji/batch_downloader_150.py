import yfinance as yf
import pandas as pd
import json
import os
import time

def batch_download_150():
    with open("TW-Quant-Shioaji/components_150.json", "r") as f:
        symbols = json.load(f)
    
    os.makedirs("TW-Quant-Shioaji/data/batch_150", exist_ok=True)
    
    print(f"開始下載 {len(symbols)} 支標的歷史數據...")
    
    for i, sym in enumerate(symbols):
        path = f"TW-Quant-Shioaji/data/batch_150/{sym}.csv"
        if os.path.exists(path):
            continue
            
        try:
            df = yf.download(f"{sym}.TW", period='2y', interval='1d', progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.to_csv(path)
                if (i+1) % 10 == 0:
                    print(f"進度: {i+1}/{len(symbols)} 已完成")
            time.sleep(0.5) 
        except Exception as e:
            print(f"下載 {sym} 失敗")

if __name__ == "__main__":
    batch_download_150()
