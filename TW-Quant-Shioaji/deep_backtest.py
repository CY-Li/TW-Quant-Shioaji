import shioaji as sj
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta

# 配置資訊
API_KEY = "BVhtFhFdjJirnkxbey2PYMUxTBz3QNFnagjiViMCMV11"
SECRET_KEY = "fjPPAFN9FuYXzsgATtfbaoY52Y3nrx3tHpVMSak4U6g"

# 將專案根目錄添加到路徑
sys.path.append(os.path.join(os.getcwd(), "TW-Quant-Shioaji"))
from core.analyzer import TWAnalyzer

def fetch_daily_data(api, symbol, years=2):
    """抓取兩年期的日線數據"""
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
        
        contract = api.Contracts.Stocks[symbol]
        kbars = api.kbars(contract=contract, start=start_date, end=end_date)
        df = pd.DataFrame({**kbars})
        df.ts = pd.to_datetime(df.ts)
        return df
    except Exception as e:
        print(f"抓取 {symbol} 失敗: {e}")
        return pd.DataFrame()

def backtest_logic(df, symbol):
    """執行回測邏輯 (日線優化版)"""
    analyzer = TWAnalyzer()
    
    # 1. 一次性計算所有指標
    df = analyzer.analyze_indicators(df)
    
    trades = []
    in_position = False
    entry_price = 0
    
    # 參數設定
    THRESHOLD = 55 # 稍微調低閾值以適應日線
    TAKE_PROFIT = 0.05 # 5% 停利
    STOP_LOSS = 0.03   # 3% 停損
    
    for i in range(100, len(df)): # 從第 100 天開始，確保指標穩定
        row = df.iloc[i]
        
        if not in_position:
            # 這裡我們需要重構 get_confidence_score 的邏輯
            # 直接使用計算好的指標
            score = 0
            if row['Close'] > row['MA200']: score += 20
            if row['MA5'] > row['MA20']: score += 15
            if row['RSI'] < 35: score += 25
            elif row['RSI'] > row['prev_rsi'] and row['RSI'] < 50: score += 10
            if row['MACD_Hist'] > 0 and df['MACD_Hist'].iloc[i-1] <= 0: score += 20
            if row['Vol_Ratio'] > 1.5: score += 10
            
            if score >= THRESHOLD:
                in_position = True
                entry_price = row['Close']
                trades.append({
                    'entry_time': str(row['ts']),
                    'entry_price': entry_price,
                    'score': score
                })
        else:
            curr_price = row['Close']
            pnl = (curr_price - entry_price) / entry_price
            if pnl >= TAKE_PROFIT or pnl <= -STOP_LOSS:
                in_position = False
                trades[-1].update({
                    'exit_time': str(row['ts']),
                    'exit_price': curr_price,
                    'pnl_pct': round(pnl * 100, 2)
                })
    
    completed = [t for t in trades if 'pnl_pct' in t]
    if not completed: return None
    
    return {
        "symbol": symbol,
        "trades_count": len(completed),
        "win_rate": f"{round(len([t for t in completed if t['pnl_pct'] > 0]) / len(completed) * 100, 2)}%",
        "avg_pnl": f"{round(np.mean([t['pnl_pct'] for t in completed]), 2)}%",
        "total_pnl": f"{round(np.sum([t['pnl_pct'] for t in completed]), 2)}%"
    }

def main():
    api = sj.Shioaji()
    api.login(api_key=API_KEY, secret_key=SECRET_KEY)
    
    sectors = {
        "半導體": "2330",
        "電子代工": "2317",
        "IC設計": "2454",
        "金融": "2881",
        "航運": "2603"
    }
    
    final_reports = []
    print("--- 啟動跨產業兩年期深度回測 ---", flush=True)
    
    for sector, sym in sectors.items():
        print(f"正在處理: {sector} ({sym})...", flush=True)
        df = fetch_daily_data(api, sym)
        if not df.empty:
            report = backtest_logic(df, sym)
            if report:
                report['sector'] = sector
                final_reports.append(report)
                
    print("\n" + "="*50, flush=True)
    print("台股兩年期多板塊回測總表", flush=True)
    print("="*50, flush=True)
    print(json.dumps(final_reports, indent=2, ensure_ascii=False))
    
    api.logout()

if __name__ == "__main__":
    main()
