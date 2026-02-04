import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

def run_single_backtest(df, mdf, params):
    ma_period = params['ma_period']
    rsi_threshold = params['rsi_threshold']
    tp = params['tp']
    sl = params['sl']
    cost = 0.005
    
    # 指標
    df['MA_Trend'] = df['Close'].rolling(ma_period).mean()
    mdf['MA_Market'] = mdf['Close'].rolling(20).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    trades = []
    in_pos = False
    entry_p = 0
    
    for i in range(ma_period, len(df)):
        m_row = mdf[mdf.index <= df.index[i]].tail(1)
        if m_row.empty: continue
        
        row = df.iloc[i]
        if not in_pos:
            if row['Close'] > row['MA_Trend'] and m_row['Close'].iloc[0] > m_row['MA_Market'].iloc[0] and row['RSI'] < rsi_threshold:
                in_pos = True
                entry_p = float(row['Close'])
                trades.append({'entry_p': entry_p})
        else:
            pnl = (float(row['Close']) - entry_p) / entry_p
            if pnl >= tp or pnl <= -sl or i == len(df) - 1: # 觸發條件或回測結束
                in_pos = False
                trades[-1] = pnl - cost
                
    results = [t for t in trades if isinstance(t, float)]
    if not results: return -999, 0
    
    total_pnl = np.sum(results)
    win_rate = len([r for r in results if r > 0]) / len(results)
    return total_pnl, win_rate

def grid_search(symbol):
    print(f"正在優化 {symbol} 參數組合...")
    yf_sym = f"{symbol}.TW"
    df = yf.download(yf_sym, period='2y', interval='1d', progress=False)
    mdf = yf.download("^TWII", period='2y', interval='1d', progress=False)
    
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if isinstance(mdf.columns, pd.MultiIndex): mdf.columns = mdf.columns.get_level_values(0)

    # 參數空間
    ma_ranges = [20, 60, 120]
    rsi_ranges = [35, 45, 55]
    tp_ranges = [0.05, 0.10, 0.15]
    sl_ranges = [0.03, 0.05, 0.08]
    
    best_pnl = -999
    best_params = {}
    
    total_iterations = len(ma_ranges) * rsi_ranges.__len__() * tp_ranges.__len__() * sl_ranges.__len__()
    count = 0
    
    for ma in ma_ranges:
        for rsi in rsi_ranges:
            for tp in tp_ranges:
                for sl in sl_ranges:
                    p = {'ma_period': ma, 'rsi_threshold': rsi, 'tp': tp, 'sl': sl}
                    pnl, wr = run_single_backtest(df, mdf, p)
                    if pnl > best_pnl:
                        best_pnl = pnl
                        best_params = {**p, 'win_rate': f"{wr*100:.1f}%", 'total_pnl': f"{pnl*100:.2f}%"}
                    count += 1
                    if count % 20 == 0: print(f"進度: {count}/{total_iterations}")

    return best_params

if __name__ == "__main__":
    summary = {}
    for s in ["2330", "2317", "2454"]:
        summary[s] = grid_search(s)
        
    print("\n" + "!"*50)
    print("台股參數優化 (Grid Search) 最終推薦")
    print("!"*50)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
