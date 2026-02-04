import pandas as pd
import requests
import json

def fetch_taiwan_indices():
    """
    抓取 0050 與 0051 成分股清單的代理方法 (使用台灣證交所或財經網站數據)
    """
    print("正在抓取 0050 + 0051 成分股清單...")
    
    # 這裡我們使用常用的財經 API 接口或靜態清單作為備案
    # 為求穩定，我這裡先整理出核心權值股清單
    # 0050 核心 (部分清單)
    ft50 = ["2330", "2317", "2454", "2308", "2881", "2882", "2303", "3711", "2412", "2886", 
            "2891", "1301", "1303", "1216", "2002", "2884", "2892", "2885", "5880", "2357"]
    
    # 0051 核心 (部分清單)
    ft100 = ["2603", "2609", "2615", "2382", "2324", "3231", "2356", "2376", "2377", "2408",
             "2409", "3481", "2448", "3034", "3035", "3037", "8046", "2498", "2345", "6239"]
    
    all_symbols = sorted(list(set(ft50 + ft100)))
    
    with open("TW-Quant-Shioaji/components.json", "w") as f:
        json.dump(all_symbols, f)
    
    print(f"成功整理出 {len(all_symbols)} 支核心成分股標的。")
    return all_symbols

if __name__ == "__main__":
    fetch_taiwan_indices()
