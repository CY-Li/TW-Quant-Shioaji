import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

# 將專案目錄添加到路徑以載入核心模組
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)
from enhanced_confirmation_system import EnhancedConfirmationSystem

def run_final_portfolio_backtest():
    with open("TW-Quant-Shioaji/components.json", "r") as f:
        symbols = json.load(f)
    
    data_dir = "TW-Quant-Shioaji/data/batch"
    initial_capital = 1000000
    capital = initial_capital
    max_positions = 5
    pos_size = capital / max_positions
    
    # 策略參數
    TP = 0.15
    SL = 0.05
    COST = 0.005
    
    active_positions = []
    trade_log = []
    
    # 初始化核心大腦
    conf_system = EnhancedConfirmationSystem()
    
    all_data = {}
    all_dates = set()
    
    print("正在準備「100% 靈魂版」指標與週線代理...")
    for sym in symbols:
        p = f"{data_dir}/{sym}.csv"
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # 1. 預計算所有原版指標 (對標集成分析器)
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA60'] = df['Close'].rolling(60).mean()
            df['MA200'] = df['Close'].rolling(200).mean()
            
            # 週線趨勢代理 (5週線 ~ 25日線)
            df['Weekly_Trend_Proxy'] = df['Close'].rolling(25).mean()
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            df['MACD'] = macd
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['MA20'] + (bb_std * 2)
            df['BB_Lower'] = df['MA20'] - (bb_std * 2)
            
            df['Volume_MA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            all_data[sym] = df
            all_dates.update(df.index.tolist())

    sorted_dates = sorted(list(all_dates))
    print(f"啟動「完全體」模擬，天數: {len(sorted_dates)}")
    
    for date in sorted_dates:
        # --- 1. 出場檢查 ---
        for pos in active_positions[:]:
            df = all_data[pos['symbol']]
            if date in df.index:
                curr_p = df.loc[date, 'Close']
                pnl = (curr_p - pos['entry_p']) / pos['entry_p']
                if pnl >= TP or pnl <= -SL:
                    net_gain = (pnl - COST) * pos_size
                    capital += (pos_size + net_gain)
                    trade_log.append({'symbol': pos['symbol'], 'pnl': pnl - COST, 'exit_date': date})
                    active_positions.remove(pos)
        
        # --- 2. 進場掃描 (多指標 + 多時框 + 增強確認) ---
        if len(active_positions) < max_positions:
            candidates = []
            for sym, df in all_data.items():
                if any(p['symbol'] == sym for p in active_positions): continue
                if date in df.index:
                    loc = df.index.get_loc(date)
                    if loc < 30: continue
                    row = df.iloc[loc]
                    
                    # A. 多時間框架分析 (日線 > MA200 且 股價 > 週線代理)
                    mtf_ok = (row['Close'] > row['MA200']) and (row['Close'] > row['Weekly_Trend_Proxy'])
                    
                    if mtf_ok:
                        # B. 增強確認系統 (直接調用原類別)
                        conf_res = conf_system.calculate_comprehensive_confirmation(df.iloc[:loc+1])
                        
                        # 進場門檻：強確認 (65分) 以上
                        if conf_res['total_score'] >= 65:
                            candidates.append({'symbol': sym, 'score': conf_res['total_score'], 'price': row['Close']})
            
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            while len(active_positions) < max_positions and candidates:
                c = candidates.pop(0)
                active_positions.append({'symbol': c['symbol'], 'entry_p': c['price'], 'entry_date': date})
                capital -= pos_size
                
    final_equity = capital + sum([pos_size for _ in active_positions])
    report = {
        "strategy": "BullPS-Final-Comprehensive",
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "total_return": f"{round((final_equity - initial_capital)/initial_capital * 100, 2)}%",
        "win_rate": f"{round(len([t for t in trade_log if t['pnl'] > 0]) / len(trade_log) * 100, 2) if trade_log else 0}%",
        "total_trades": len(trade_log)
    }
    
    with open("TW-Quant-Shioaji/portfolio_report_final.json", "w") as f:
        json.dump(report, f, indent=2)
    return report

if __name__ == "__main__":
    res = run_final_portfolio_backtest()
    print(json.dumps(res, indent=2, ensure_ascii=False))
