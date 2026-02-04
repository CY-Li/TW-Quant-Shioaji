import yfinance as yf
import json
import os
from pathlib import Path

# 定義板塊標的
SECTORS = {
    "電腦及週邊設備": ["2317", "2382", "3231", "2357", "2376"],
    "電子零組件": ["2327", "3037", "2368", "8046", "3044"],
    "航運業": ["2603", "2609", "2615", "2610", "2618"],
    "通信、網路與光電": ["2412", "3045", "2345", "2409", "3481"],
    "電機、生技與其他電子": ["1519", "1513", "1504", "1795", "2360"],
    "傳統產業與其他": ["2002", "1101", "1216", "1476", "9910"]
}

DATA_DIR = "TW-Quant-Shioaji/data/batch_sector"

def download_sector_data():
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    all_symbols = [s for list in SECTORS.values() for s in list]
    
    print(f"🚀 開始下載 {len(all_symbols)} 檔跨板塊股票數據...")
    
    for sym in all_symbols:
        yf_sym = f"{sym}.TW"
        print(f"  📥 下載 {yf_sym}...")
        df = yf.download(yf_sym, period='3y', interval='1d', progress=False)
        if not df.empty:
            if hasattr(df.columns, 'get_level_values'):
                df.columns = df.columns.get_level_values(0)
            csv_path = os.path.join(DATA_DIR, f"{sym}.csv")
            df.to_csv(csv_path)
            print(f"    ✅ 已存至 {csv_path}")
        else:
            print(f"    ❌ {yf_sym} 無數據")

if __name__ == "__main__":
    download_sector_data()
