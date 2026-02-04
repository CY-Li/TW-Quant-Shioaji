import pandas as pd
import numpy as np
import json
import os
import sys

# 將專案根目錄添加到路徑
sys.path.append(os.path.join(os.getcwd(), "TW-Quant-Shioaji"))
from core.analyzer import TWAnalyzer

def run_advanced_backtest(symbol):
    file_path = f"TW-Quant-Shioaji/data/{symbol}_pro.csv"
    if not os.path.exists(file_path):
        return {"error": f"Data file for {symbol} not found"}
        
    df = pd.read_csv(file_path)
    df['ts'] = pd.to_datetime(df['ts'])
    
    analyzer = TWAnalyzer()
    
    trades = []
    in_position = False
    entry_price = 0
    
    # 設置回測參數
    CONFIDENCE_THRESHOLD = 60 # 信心評分超過 60 才進場
    PROFIT_TARGET = 0.008     # 0.8% 停利 (考慮台股交易成本較高)
    STOP_LOSS = 0.01          # 1% 停損
    
    print(f"--- 啟動 {symbol} 進階大腦回測 ---")
    
    # 為了計算指標，我們需要從足夠的歷史數據開始 (例如 200 根 K 線)
    for i in range(200, len(df)):
        current_window = df.iloc[i-200:i+1].copy()
        
        if not in_position:
            score, factors = analyzer.get_confidence_score(current_window)
            
            if score >= CONFIDENCE_THRESHOLD:
                in_position = True
                entry_price = df['Close'].iloc[i]
                entry_time = df['ts'].iloc[i]
                trades.append({
                    'entry_time': str(entry_time),
                    'entry_price': entry_price,
                    'score': score,
                    'reasons': factors
                })
        else:
            current_price = df['Close'].iloc[i]
            pnl_pct = (current_price - entry_price) / entry_price
            
            # 出場邏輯
            if pnl_pct >= PROFIT_TARGET or pnl_pct <= -STOP_LOSS:
                in_position = False
                trades[-1].update({
                    'exit_time': str(df['ts'].iloc[i]),
                    'exit_price': current_price,
                    'pnl_pct': round(pnl_pct * 100, 2)
                })

    completed_trades = [t for t in trades if 'pnl_pct' in t]
    if not completed_trades:
        return {"symbol": symbol, "msg": "No completed trades"}

    wins = [t for t in completed_trades if t['pnl_pct'] > 0]
    win_rate = len(wins) / len(completed_trades)
    
    return {
        "symbol": symbol,
        "total_trades": len(completed_trades),
        "win_rate": f"{round(win_rate * 100, 2)}%",
        "avg_pnl": f"{round(np.mean([t['pnl_pct'] for t in completed_trades]), 2)}%",
        "total_accumulated_pnl": f"{round(np.sum([t['pnl_pct'] for t in completed_trades]), 2)}%",
        "best_trade": max([t['pnl_pct'] for t in completed_trades]),
        "worst_trade": min([t['pnl_pct'] for t in completed_trades])
    }

if __name__ == "__main__":
    # 執行台積電回測
    res = run_advanced_backtest("2330")
    print(json.dumps(res, indent=2, ensure_ascii=False))
