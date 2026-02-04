import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

def run_swing_backtest_yf(symbol):
    yf_sym = f"{symbol}.TW"
    
    df = yf.download(yf_sym, period='2y', interval='1d', progress=False)
    mdf = yf.download("^TWII", period='2y', interval='1d', progress=False)
    
    if df.empty or mdf.empty: return None
        
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if isinstance(mdf.columns, pd.MultiIndex): mdf.columns = mdf.columns.get_level_values(0)

    # 指標計算
    df['MA200'] = df['Close'].rolling(200).mean()
    mdf['MA20'] = mdf['Close'].rolling(20).mean()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        return 100 - (100 / (1 + (gain / loss)))
    
    df['RSI'] = calculate_rsi(df['Close'])
    
    # 策略參數 (拉長週期)
    PROFIT_TARGET = 0.15 # 15% 停利
    STOP_LOSS = 0.07      # 7% 停損
    TRANS_COST = 0.005    # 0.5% 交易成本 (含手續費與交易稅)
    
    trades = []
    in_pos = False
    entry_p = 0
    
    for i in range(200, len(df)):
        current_date = df.index[i]
        m_rows = mdf[mdf.index <= current_date]
        if m_rows.empty: continue
        m_last = m_rows.iloc[-1]
        
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        if not in_pos:
            # 進場邏輯：長期多頭 + 大盤多頭 + 短線超賣 + 動能翻正
            if (row['Close'] > row['MA200'] and 
                m_last['Close'] > m_last['MA20'] and 
                row['RSI'] < 45 and 
                row['MACD_Hist'] > 0 and prev_row['MACD_Hist'] <= 0):
                
                in_pos = True
                entry_p = float(row['Close'])
                trades.append({'entry_date': str(current_date), 'entry_price': entry_p})
        else:
            pnl = (float(row['Close']) - entry_p) / entry_p
            if pnl >= PROFIT_TARGET or pnl <= -STOP_LOSS:
                in_pos = False
                net_pnl = pnl - TRANS_COST
                trades[-1].update({
                    'exit_date': str(current_date),
                    'net_pnl': round(net_pnl * 100, 2)
                })
    
    completed = [t for t in trades if 'net_pnl' in t]
    if not completed: return None
    
    win_rate = len([t for t in completed if t['net_pnl'] > 0]) / len(completed)
    return {
        "symbol": symbol,
        "net_total_pnl": f"{round(np.sum([t['net_pnl'] for t in completed]), 2)}%",
        "win_rate": f"{round(win_rate * 100, 2)}%",
        "trades_count": len(completed),
        "avg_net_pnl": f"{round(np.mean([t['net_pnl'] for t in completed]), 2)}%"
    }

if __name__ == "__main__":
    targets = ["2330", "2317", "2454", "2881", "2603", "2382", "2308"]
    results = []
    for s in targets:
        res = run_swing_backtest_yf(s)
        if res: results.append(res)
    print(json.dumps(results, indent=2, ensure_ascii=False))
