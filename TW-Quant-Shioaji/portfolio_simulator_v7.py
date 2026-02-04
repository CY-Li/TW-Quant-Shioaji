import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

# 將專案根目錄添加到路徑以載入核心模組
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "TW-Quant-Shioaji"))
from core.integrated_stock_analyzer import IntegratedStockAnalyzer
from backtester.engine import evaluate_smart_sar_exit, evaluate_exit_confidence

def run_legacy_brain_backtest():
    with open("TW-Quant-Shioaji/components_150.json", "r") as f:
        symbols = json.load(f)
    
    data_dir = "TW-Quant-Shioaji/data/batch_150"
    initial_capital = 1000000
    capital = initial_capital
    max_positions = 10
    pos_size_pct = 0.1
    COST = 0.005 # 交易成本
    
    analyzer = IntegratedStockAnalyzer()
    active_positions = [] # [{'symbol', 'entry_price', 'entry_date', 'shares', 'initial_analysis_snapshot'}]
    trade_log = []
    
import multiprocessing
from functools import partial

# ... (保留原有的 imports 和 setup)

def process_stock_data(sym, data_dir):
    """單一股票的數據處理函數，用於多進程調用"""
    try:
        p = f"{data_dir}/{sym}.csv"
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # 初始化 analyzer (每個進程需要獨立的實例或靜態調用)
            # 由於 analyzer 比較重，我們這裡只實例化一次或確保它是輕量的
            # 為了簡單起見，我們在這裡重新實例化，或者將 calculate_technical_indicators 變為靜態方法
            # 這裡假設 IntegratedStockAnalyzer 初始化不耗時
            analyzer = IntegratedStockAnalyzer() 
            df = analyzer.calculate_technical_indicators(df)
            return sym, df
    except Exception as e:
        print(f"Error processing {sym}: {e}")
    return sym, None

def run_legacy_brain_backtest():
    with open("TW-Quant-Shioaji/components_150.json", "r") as f:
        symbols = json.load(f)
    
    data_dir = "TW-Quant-Shioaji/data/batch_150"
    initial_capital = 1000000
    capital = initial_capital
    max_positions = 10
    pos_size_pct = 0.1
    COST = 0.005 # 交易成本
    
    analyzer = IntegratedStockAnalyzer()
    active_positions = [] 
    trade_log = []
    
    # 預加載數據 - 使用多進程優化
    all_data = {}
    all_dates = set()
    print(f"正在準備「Legacy Brain」全市場數據與指標 (使用 {multiprocessing.cpu_count()} 核心並行處理)...")
    
    # 建立進程池
    with multiprocessing.Pool() as pool:
        # 使用 partial 固定 data_dir 參數
        worker = partial(process_stock_data, data_dir=data_dir)
        # 並行執行
        results = pool.map(worker, symbols)
        
    # 收集結果
    for sym, df in results:
        if df is not None:
            all_data[sym] = df
            all_dates.update(df.index.tolist())

    sorted_dates = sorted(list(all_dates))
    print(f"啟動「靈魂復刻版」模擬，標的: {len(all_data)}，天數: {len(sorted_dates)}")
    
    for date in sorted_dates:
        # --- 1. 出場檢查 (嚴格執行 SAR 與 信心度模型) ---
        for pos in active_positions[:]:
            df = all_data[pos['symbol']]
            if date in df.index:
                loc = df.index.get_loc(date)
                # 準備當前分析快照
                current_window = df.iloc[:loc+1]
                # 模擬 analyzer 的分析結果
                current_analysis = {
                    'symbol': pos['symbol'],
                    'current_price': df.loc[date, 'Close'],
                    'sar': df.loc[date, 'SAR'],
                    'rsi': df.loc[date, 'RSI'],
                    'macd': df.loc[date, 'MACD'],
                    'macd_histogram': df.loc[date, 'MACD_Histogram'],
                    'volume_ratio': df.loc[date, 'Volume_Ratio'],
                    'ma20': df.loc[date, 'MA20'],
                    'ma5': df.loc[date, 'MA5'],
                    'confidence_factors': [] # 這裡為了加速回測，簡化因素獲取，但在邏輯中仍會判斷
                }
                
                # A. 智能 SAR 出場
                sar_res = evaluate_smart_sar_exit(pos, current_analysis, current_dt=date)
                # B. 信心度出場
                conf_res = evaluate_exit_confidence(pos, current_analysis)
                
                if sar_res['should_exit'] or conf_res['exit_confidence'] >= 0.8:
                    exit_p = df.loc[date, 'Close']
                    pnl = (exit_p - pos['entry_price']) / pos['entry_price'] - COST
                    capital += (pos['entry_price'] * pos['shares']) * (1 + pnl)
                    print(f"[{date.date()}] EXIT {pos['symbol']} at {exit_p:.2f} | PnL: {pnl*100:.2f}%")
                    trade_log.append({'symbol': pos['symbol'], 'pnl': pnl, 'exit_date': date})
                    active_positions.remove(pos)

        # --- 2. 進場掃描 (使用原版綜合評分) ---
        if len(active_positions) < max_positions:
            candidates = []
            for sym, df in all_data.items():
                if any(p['symbol'] == sym for p in active_positions): continue
                if date in df.index:
                    loc = df.index.get_loc(date)
                    if loc < 60: continue
                    # 執行原版進場評估
                    try:
                        # 測試日誌
                        # print(f"Checking {sym} on {date.date()}")
                        entry_advice, score, level, factors = analyzer.assess_entry_opportunity(df.iloc[:loc+1])
                        if score > 0:
                             print(f"[{date.date()}] {sym} Score: {score}")
                    except Exception as e:
                        print(f"評估 {sym} 時發生錯誤: {e}")
                        score = 0
                        factors = []
                    
                    if score >= 65: # 靈魂版閾值再次下修，捕捉更多反彈訊號
                        candidates.append({'symbol': sym, 'score': score, 'price': df.loc[date, 'Close'], 'factors': factors})
            
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            while len(active_positions) < max_positions and candidates:
                c = candidates.pop(0)
                invest = capital * pos_size_pct
                shares = invest / c['price']
                print(f"[{date.date()}] ENTRY {c['symbol']} at {c['price']:.2f} | Score: {c['score']}")
                active_positions.append({
                    'symbol': c['symbol'],
                    'entry_price': c['price'],
                    'entry_date': date,
                    'shares': shares,
                    'initial_analysis_snapshot': {'confidence_factors': c['factors']}
                })
                capital -= invest

    final_val = capital + sum([p['entry_price'] * p['shares'] for p in active_positions])
    wr = len([t for t in trade_log if t['pnl'] > 0]) / len(trade_log) if trade_log else 0
    
    report = {
        "strategy": "BullPS-Legacy-Brain-v7",
        "total_return": f"{round((final_val - initial_capital)/initial_capital * 100, 2)}%",
        "win_rate": f"{round(wr * 100, 2)}%",
        "total_trades": len(trade_log),
        "final_equity": round(final_val, 2)
    }
    
    with open("TW-Quant-Shioaji/portfolio_report_v7.json", "w") as f:
        json.dump(report, f, indent=2)
    return report

if __name__ == "__main__":
    res = run_legacy_brain_backtest()
    print(json.dumps(res, indent=2, ensure_ascii=False))
