import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def run_ultimate_portfolio_backtest():
    # 載入 150 支標的
    with open("TW-Quant-Shioaji/components_150.json", "r") as f:
        symbols = json.load(f)
    
    data_dir = "TW-Quant-Shioaji/data/batch_150"
    initial_capital = 1000000
    capital = initial_capital
    max_positions = 10 # 擴大持倉上限以測試多樣性
    pos_size_pct = 0.1 # 每筆固定 10%
    
    # 策略參數 (優化版：更靈敏的反轉確認)
    TP = 0.15 # 15% 停利
    SL = 0.05 # 5% 停損
    COST = 0.005 # 0.5% 成本
    
    active_positions = []
    trade_log = []
    
    # 預加載數據
    all_data = {}
    all_dates = set()
    print("正在初始化海量數據指標...")
    for sym in symbols:
        p = f"{data_dir}/{sym}.csv"
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # 核心指標
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA200'] = df['Close'].rolling(200).mean()
            
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD_Hist'] = exp1 - exp2 - (exp1 - exp2).ewm(span=9, adjust=False).mean()
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            all_data[sym] = df
            all_dates.update(df.index.tolist())
    
    sorted_dates = sorted(list(all_dates))
    print(f"啟動全市場 150 支標的模擬，天數: {len(sorted_dates)}")
    
    for date in sorted_dates:
        # 1. 出場
        for pos in active_positions[:]:
            df = all_data[pos['symbol']]
            if date in df.index:
                curr_p = df.loc[date, 'Close']
                pnl = (curr_p - pos['entry_p']) / pos['entry_p']
                if pnl >= TP or pnl <= -SL:
                    pnl_with_cost = pnl - COST
                    capital += pos['invested'] * (1 + pnl_with_cost)
                    trade_log.append({'symbol': pos['symbol'], 'pnl': pnl_with_cost, 'date': date})
                    active_positions.remove(pos)
        
        # 2. 進場
        if len(active_positions) < max_positions:
            candidates = []
            for sym, df in all_data.items():
                if any(p['symbol'] == sym for p in active_positions): continue
                if date in df.index:
                    loc = df.index.get_loc(date)
                    if loc < 1: continue
                    row = df.iloc[loc]
                    prev = df.iloc[loc-1]
                    
                    # 靈魂條件：趨勢+動能+量能
                    if row['Close'] > row['MA20'] and row['MACD_Hist'] > 0 and prev['MACD_Hist'] <= 0 and row['Vol_Ratio'] > 1.2:
                        score = row['Vol_Ratio'] * 50 + (100 - row['RSI'])
                        candidates.append({'symbol': sym, 'score': score, 'price': row['Close']})
            
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            while len(active_positions) < max_positions and candidates:
                c = candidates.pop(0)
                invest = capital * pos_size_pct
                if invest > 0:
                    active_positions.append({'symbol': c['symbol'], 'entry_p': c['price'], 'invested': invest})
                    capital -= invest

    # 統計
    final_val = capital + sum([p['invested'] for p in active_positions])
    wr = len([t for t in trade_log if t['pnl'] > 0]) / len(trade_log) if trade_log else 0
    
    report = {
        "final_equity": round(final_val, 2),
        "total_return": f"{round((final_val - initial_capital)/initial_capital*100, 2)}%",
        "win_rate": f"{round(wr*100, 2)}%",
        "total_trades": len(trade_log)
    }
    
    with open("TW-Quant-Shioaji/portfolio_report_v150.json", "w") as f:
        json.dump(report, f, indent=2)
    return report

if __name__ == "__main__":
    run_ultimate_portfolio_backtest()
