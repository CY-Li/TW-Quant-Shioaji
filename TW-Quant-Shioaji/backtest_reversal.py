import pandas as pd
import numpy as np
import json
import os

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def backtest_tw_reversal(symbol):
    file_path = f"TW-Quant-Shioaji/data/{symbol}_pro.csv"
    if not os.path.exists(file_path):
        return {"error": "Data file not found"}
        
    df = pd.read_csv(file_path)
    df['ts'] = pd.to_datetime(df['ts'])
    
    # 策略參數
    RSI_PERIOD = 14
    RSI_BUY = 30
    VOL_FACTOR = 1.5
    
    # 技術指標計算
    df['RSI'] = calculate_rsi(df['Close'], RSI_PERIOD)
    df['Vol_MA5'] = df['Volume'].rolling(20).mean() # 假設約 20 根 1m K線
    
    trades = []
    in_position = False
    entry_price = 0
    
    for i in range(20, len(df)):
        # 買入訊號：RSI < 30 且 成交量大於均量 1.5倍
        if not in_position:
            if df['RSI'].iloc[i] < RSI_BUY and df['Volume'].iloc[i] > df['Vol_MA5'].iloc[i] * VOL_FACTOR:
                in_position = True
                entry_price = df['Close'].iloc[i]
                entry_time = df['ts'].iloc[i]
                trades.append({'entry_time': str(entry_time), 'entry_price': entry_price, 'type': 'BUY'})
        
        # 賣出訊號：獲利 1% 或 損平 (簡單原型)
        elif in_position:
            current_price = df['Close'].iloc[i]
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= 0.01 or profit_pct <= -0.01:
                in_position = False
                trades[-1].update({
                    'exit_time': str(df['ts'].iloc[i]),
                    'exit_price': current_price,
                    'pnl_pct': round(profit_pct * 100, 2)
                })

    if not trades:
        return {"symbol": symbol, "msg": "No trades executed with current logic"}

    completed_trades = [t for t in trades if 'pnl_pct' in t]
    win_rate = len([t for t in completed_trades if t['pnl_pct'] > 0]) / len(completed_trades) if completed_trades else 0
    
    return {
        "symbol": symbol,
        "total_trades": len(completed_trades),
        "win_rate": f"{round(win_rate * 100, 2)}%",
        "avg_pnl": f"{round(np.mean([t['pnl_pct'] for t in completed_trades]), 2)}%",
        "trades": completed_trades[:5] # 只顯示前 5 筆
    }

if __name__ == "__main__":
    result = backtest_tw_reversal("2330")
    print(json.dumps(result, indent=2, ensure_ascii=False))
