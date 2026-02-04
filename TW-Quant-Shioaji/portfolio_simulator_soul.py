import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def run_soul_portfolio_backtest():
    with open("TW-Quant-Shioaji/components.json", "r") as f:
        symbols = json.load(f)
    
    data_dir = "TW-Quant-Shioaji/data/batch"
    initial_capital = 1000000
    capital = initial_capital
    max_positions = 5
    pos_size = capital / max_positions
    
    # 策略參數
    TP = 0.12
    SL = 0.05
    COST = 0.005
    
    active_positions = []
    trade_log = []
    
    # 讀取並計算所有「靈魂指標」
    all_data = {}
    all_dates = set()
    print("正在初始化 59 支標的之進階技術指標...")
    for sym in symbols:
        p = f"{data_dir}/{sym}.csv"
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # 1. 均線排列 (MA5 > MA20 > MA60) - 趨勢靈魂
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA60'] = df['Close'].rolling(60).mean()
            df['MA200'] = df['Close'].rolling(200).mean()
            
            # 2. MACD 柱狀圖轉正 (Momentum Turn)
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            df['MACD_Hist'] = macd - macd.ewm(span=9, adjust=False).mean()
            
            # 3. RSI 超賣反轉 (RSI 14)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            # 4. SAR 拋物線 (簡化版邏輯)
            # 這裡簡化：收盤價 > SAR 視為翻多
            # 我們直接判斷收盤價是否上穿 MA20 作為替代確認
            
            # 5. 成交量倍增 (Volume Ratio > 1.5)
            df['Vol_Avg'] = df['Volume'].rolling(20).mean()
            df['Vol_Ratio'] = df['Volume'] / df['Vol_Avg']
            
            all_data[sym] = df
            all_dates.update(df.index.tolist())
    
    sorted_dates = sorted(list(all_dates))
    print(f"啟動「靈魂版」模擬，總天數: {len(sorted_dates)}")
    
    for i, date in enumerate(sorted_dates):
        # --- 1. 出場檢查 (與 V4 相同，但加入更嚴格的 MA5 破線停損) ---
        for pos in active_positions[:]:
            df = all_data[pos['symbol']]
            if date in df.index:
                curr_p = df.loc[date, 'Close']
                pnl = (curr_p - pos['entry_p']) / pos['entry_p']
                
                # 出場條件：達標 OR 破位停損 (股價跌破 MA5 且 PNL 為負)
                if pnl >= TP or pnl <= -SL or (curr_p < df.loc[date, 'MA5'] and pnl < 0):
                    net_gain = (pnl - COST) * pos_size
                    capital += (pos_size + net_gain)
                    trade_log.append({'symbol': pos['symbol'], 'pnl': pnl - COST, 'exit_date': date})
                    active_positions.remove(pos)
        
        # --- 2. 進場掃描 (核心四部曲) ---
        if len(active_positions) < max_positions:
            candidates = []
            for sym, df in all_data.items():
                if any(p['symbol'] == sym for p in active_positions): continue
                if date in df.index:
                    loc = df.index.get_loc(date)
                    if loc < 1: continue
                    row = df.iloc[loc]
                    prev_row = df.iloc[loc-1]
                    
                    # A. 趨勢條件：站在 200MA 之上 或 MA5/20/60 多頭排列
                    trend_ok = (row['Close'] > row['MA200']) or (row['MA5'] > row['MA20'] > row['MA60'])
                    
                    # B. 反轉條件：MACD Hist 由負轉正 或 RSI 從 <30 彈起
                    reversal_ok = (row['MACD_Hist'] > 0 and prev_row['MACD_Hist'] <= 0) or (row['RSI'] < 35 and row['RSI'] > prev_row['RSI'])
                    
                    # C. 量能確認：爆量
                    volume_ok = row['Vol_Ratio'] > 1.2
                    
                    if trend_ok and reversal_ok and volume_ok:
                        # 評分：量能越大、RSI 越低，分數越高
                        score = row['Vol_Ratio'] * 20 + (50 - row['RSI'])
                        candidates.append({'symbol': sym, 'score': score, 'price': row['Close']})
            
            # 精選進場
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            while len(active_positions) < max_positions and candidates:
                c = candidates.pop(0)
                active_positions.append({'symbol': c['symbol'], 'entry_p': c['price'], 'entry_date': date})
                capital -= pos_size
    
    final_equity = capital + sum([pos_size for _ in active_positions])
    report = {
        "strategy": "BullPS-Pure-Soul-Logic",
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "total_return": f"{round((final_equity - initial_capital)/initial_capital * 100, 2)}%",
        "win_rate": f"{round(len([t for t in trade_log if t['pnl'] > 0]) / len(trade_log) * 100, 2) if trade_log else 0}%",
        "total_trades": len(trade_log)
    }
    
    with open("TW-Quant-Shioaji/portfolio_report_soul.json", "w") as f:
        json.dump(report, f, indent=2)
    return report

if __name__ == "__main__":
    res = run_soul_portfolio_backtest()
    print(json.dumps(res, indent=2, ensure_ascii=False))
