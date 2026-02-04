import shioaji as sj
import pandas as pd
import json
from datetime import datetime, timedelta

# 配置 (使用使用者提供的測試 Key)
API_KEY = "DKGb3BFQ7SsX8Mb9y3bhrMmrgjjUPoD827Wjd6Ki5rKk"
SECRET_KEY = "125X7iLKE525gr2njqPu5zB83WDVnsfnY4mNydbkZSJF"

def fetch_historical_data(symbol, days=30):
    api = sj.Shioaji()
    try:
        print(f"正在連線並登入以抓取 {symbol} 歷史數據...")
        api.login(api_key=API_KEY, secret_key=SECRET_KEY)
        
        # 設定時間範圍 (抓取最近 30 天)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        print(f"抓取範圍: {start_date} 至 {end_date}")
        
        contract = api.Contracts.Stocks[symbol]
        kbars = api.kbars(
            contract=contract, 
            start=start_date, 
            end=end_date,
        )
        
        df = pd.DataFrame({**kbars})
        if df.empty:
            return {"status": "empty", "msg": "抓取成功但無數據回傳"}
            
        df.ts = pd.to_datetime(df.ts)
        
        # 儲存為 CSV 以供後續回測使用
        output_path = f"TW-Quant-Shioaji/data/{symbol}_history.csv"
        os.makedirs("TW-Quant-Shioaji/data", exist_ok=True)
        df.to_csv(output_path, index=False)
        
        return {
            "status": "success",
            "symbol": symbol,
            "rows": len(df),
            "start": str(df.ts.min()),
            "end": str(df.ts.max()),
            "file": output_path
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        api.logout()

if __name__ == "__main__":
    import os
    # 測試抓取 2330 台積電
    result = fetch_historical_data("2330")
    print(json.dumps(result, indent=2, ensure_ascii=False))
