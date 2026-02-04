import shioaji as sj
import pandas as pd
import time
import json
import sys
import os

# 配置
API_KEY = "DKGb3BFQ7SsX8Mb9y3bhrMmrgjjUPoD827Wjd6Ki5rKk"
SECRET_KEY = "125X7iLKE525gr2njqPu5zB83WDVnsfnY4mNydbkZSJF"

class TWHistoricalEngine:
    def __init__(self):
        self.api = sj.Shioaji()
        self.is_logged_in = False

    def login(self):
        try:
            self.api.login(api_key=API_KEY, secret_key=SECRET_KEY)
            self.is_logged_in = True
            return True
        except:
            return False

    def get_kbars(self, symbol, start_date, end_date):
        """抓取台股 K 線數據"""
        if not self.is_logged_in:
            self.login()
        
        try:
            contract = self.api.Contracts.Stocks[symbol]
            kbars = self.api.kbars(
                contract=contract, 
                start=start_date, 
                end=end_date,
            )
            df = pd.DataFrame({**kbars})
            df.ts = pd.to_datetime(df.ts)
            return df
        except Exception as e:
            print(f"抓取 {symbol} 數據失敗: {e}")
            return pd.DataFrame()

def run_backtest_prototype(symbol):
    """
    台股版回測原型：RSI + 成交量反轉策略
    """
    engine = TWHistoricalEngine()
    # 注意：測試 Key 可能無法抓取完整歷史數據，這裡我們先建立邏輯框架
    print(f"--- 啟動 {symbol} 台股回測原型 ---")
    
    # 策略邏輯 (模擬)
    # 1. RSI < 35 (超賣)
    # 2. 當日成交量 > 5日均量 1.5倍
    # 3. 股價 > 200MA (長多格局)
    
    report = {
        "strategy": "TW-BPS-Reversal",
        "symbol": symbol,
        "metrics": {
            "win_rate": "待數據填充",
            "expectancy": "待數據填充",
            "status": "數據引擎建設中"
        },
        "next_step": "等待正式環境憑證以獲取歷史 K 線"
    }
    return report

if __name__ == "__main__":
    res = run_backtest_prototype("2330")
    print(json.dumps(res, indent=2, ensure_ascii=False))
