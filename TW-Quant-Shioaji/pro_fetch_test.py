import shioaji as sj
import pandas as pd
import json
import os
from datetime import datetime, timedelta

# 配置最高權限 Key
API_KEY = "BVhtFhFdjJirnkxbey2PYMUxTBz3QNFnagjiViMCMV11"
SECRET_KEY = "fjPPAFN9FuYXzsgATtfbaoY52Y3nrx3tHpVMSak4U6g"

def fetch_tw_history(symbol, days=15):
    api = sj.Shioaji()
    try:
        print(f"正在以最高權限登入並抓取 {symbol} 數據...")
        api.login(api_key=API_KEY, secret_key=SECRET_KEY)
        
        # 獲取合約資訊
        contract = api.Contracts.Stocks[symbol]
        
        # 設定抓取範圍 (抓取最近 15 天)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        kbars = api.kbars(
            contract=contract, 
            start=start_date, 
            end=end_date,
        )
        
        df = pd.DataFrame({**kbars})
        if df.empty:
            return {"status": "empty", "msg": "無數據回傳"}
            
        df.ts = pd.to_datetime(df.ts)
        
        # 儲存數據
        os.makedirs("TW-Quant-Shioaji/data", exist_ok=True)
        output_path = f"TW-Quant-Shioaji/data/{symbol}_pro.csv"
        df.to_csv(output_path, index=False)
        
        return {
            "status": "success",
            "symbol": symbol,
            "rows": len(df),
            "start": str(df.ts.min()),
            "end": str(df.ts.max())
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        api.logout()

if __name__ == "__main__":
    # 先測試抓取台積電 (2330)
    result = fetch_tw_history("2330")
    print(json.dumps(result, indent=2, ensure_ascii=False))
