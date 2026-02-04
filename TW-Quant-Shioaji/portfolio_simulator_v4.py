import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def run_pro_portfolio_backtest():
    with open("TW-Quant-Shioaji/components.json", "r") as f:
        symbols = json.load(f)
    
    data_dir = "TW-Quant-Shioaji/data/batch"
    initial_capital = 1000000
    capital = initial_capital
    max_positions = 5 # 縮減持倉，精選標的
    pos_size = capital / max_positions
    
    # 策略參數 (優化版)
    TP = 0.12
    SL = 0.05
    COST = 0.005
    
    active_positions = []
    equity_curve = []
    trade_log = []
    
    # 讀取並計算進階指標
    all_data = {}
    all_dates = set()
    for sym in symbols:
        p = f"{data_dir}/{sym}.csv"
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # 1. 趨勢指標
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA200'] = df['Close'].rolling(200).mean()
            
            # 2. 動能指標 (MACD)
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD_Hist'] = exp1 - exp2 - (exp1 - exp2).ewm(span=9, adjust=False).mean()
            
            # 3. 反轉指標 (RSI)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            # 4. 成交量比率
            df['Vol_Avg'] = df['Volume'].rolling(20).mean()
            df['Vol_Ratio'] = df['Volume'] / df['Vol_Avg']
            
            all_data[sym] = df
            all_dates.update(df.index.tolist())
    
    # 大盤數據 (0050 代理)
    mdf = all_data.get('0050') if '0050' in all_data else None
    
    sorted_dates = sorted(list(all_dates))
    print(f"啟動完整邏輯模擬，總標的: {len(all_data)}，總天數: {len(sorted_dates)}", flush=True)
    
    count = 0
    total_days = len(sorted_dates)
    for date in sorted_dates:
        count += 1
        if count % 50 == 0:
            print(f"進度: {count}/{total_days} 日 ({round(count/total_days*100, 2)}%)", flush=True)
        # --- 1. 出場檢查 ---
        for pos in active_positions[:]:
            df = all_data[pos['symbol']]
            if date in df.index:
                curr_p = df.loc[date, 'Close']
                pnl = (curr_p - pos['entry_p']) / pos['entry_p']
                
                # 智能出場：達標 或 趨勢反轉 (MACD 轉負)
                if pnl >= TP or pnl <= -SL or df.loc[date, 'MACD_Hist'] < 0:
                    net_gain = (pnl - COST) * pos_size
                    capital += (pos_size + net_gain)
                    trade_log.append({'symbol': pos['symbol'], 'pnl': pnl - COST, 'exit_date': date})
                    active_positions.remove(pos)
        
        # --- 2. 進場掃描 ---
        if len(active_positions) < max_positions:
            candidates = []
            for sym, df in all_data.items():
                if any(p['symbol'] == sym for p in active_positions): continue
                if date in df.index:
                    row = df.loc[date]
                    prev_row = df.iloc[df.index.get_loc(date)-1] if df.index.get_loc(date) > 0 else row
                    
                    # 靈魂邏輯 A：長期趨勢必須向上
                    trend_ok = row['Close'] > row['MA200']
                    
                    # 靈魂邏輯 B：大盤不准走空 (這裡簡化)
                    
                    # 靈魂邏輯 C：技術共振 (RSI低檔 + MACD翻正 + 爆量)
                    reversal_ok = (row['RSI'] < 45) and (row['MACD_Hist'] > 0 and prev_row['MACD_Hist'] <= 0)
                    volume_ok = row['Vol_Ratio'] > 1.1
                    
                    if trend_ok and reversal_ok and volume_ok:
                        score = row['Vol_Ratio'] * 10 + (50 - row['RSI'])
                        candidates.append({'symbol': sym, 'score': score, 'price': row['Close']})
            
            # 精選進場
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            while len(active_positions) < max_positions and candidates:
                c = candidates.pop(0)
                active_positions.append({'symbol': c['symbol'], 'entry_p': c['price'], 'entry_date': date})
                capital -= pos_size
        
        # 3. 淨值記錄 (略)
    
    final_equity = capital + sum([pos_size for _ in active_positions])
    print(f"模擬完成。最終淨值: {final_equity}, 總交易數: {len(trade_log)}", flush=True)
    report = {
        "strategy": "BullPS-Full-Logic",
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "win_rate": f"{round(len([t for t in trade_log if t['pnl'] > 0]) / len(trade_log) * 100, 2) if trade_log else 0}%",
        "total_trades": len(trade_log)
    }
    
    with open("TW-Quant-Shioaji/portfolio_report_v4.json", "w") as f:
        json.dump(report, f, indent=2)
    return report

if __name__ == "__main__":
    run_pro_portfolio_backtest()
