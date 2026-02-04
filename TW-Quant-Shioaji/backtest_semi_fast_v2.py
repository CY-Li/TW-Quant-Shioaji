import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
import multiprocessing
from functools import partial

# 將專案路徑加入以引用模組
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "TW-Quant-Shioaji"))

from core.integrated_stock_analyzer import IntegratedStockAnalyzer
sys.path.append(os.path.join(os.getcwd(), "BullPS-v3"))
from backend.portfolio_manager import evaluate_smart_sar_exit, evaluate_exit_confidence

# 配置
DATA_DIR = "TW-Quant-Shioaji/data/batch_semi"
INDEX_FILE = f"{DATA_DIR}/twii.csv"
SEMI_LIST_FILE = "TW-Quant-Shioaji/semi_10.json"
REPORT_FILE = "TW-Quant-Shioaji/semi_backtest_optimized_report.json"

ENTRY_THRESHOLD_BASE = 65
CAPITAL = 1000000
MAX_POSITIONS = 5 # 既然只有10支，同時持有5支算多了
POS_SIZE_PCT = 0.2
COST = 0.005

def process_stock_data(sym, data_dir):
    try:
        print(f"  🧵 正在處理 {sym}...")
        p = f"{data_dir}/{sym}.csv"
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # 使用 Analyzer 計算技術指標
            analyzer = IntegratedStockAnalyzer() 
            df = analyzer.calculate_technical_indicators(df)
            print(f"  ✅ {sym} 指標計算完成")
            return sym, df
    except Exception as e:
        print(f"Error processing {sym}: {e}")
    return sym, None

def run_backtest():
    with open(SEMI_LIST_FILE, "r") as f:
        symbols = json.load(f)
    
    # 1. 預加載大盤數據
    mdf = pd.read_csv(INDEX_FILE)
    mdf['Date'] = pd.to_datetime(mdf['Date'])
    mdf.set_index('Date', inplace=True)
    mdf['MA5'] = mdf['Close'].rolling(5).mean()

    # 2. 預加載個股數據 (多進程)
    all_data = {}
    all_dates = set()
    print(f"🔄 正在計算指標 (使用 {multiprocessing.cpu_count()} 核心)...")
    
    with multiprocessing.Pool() as pool:
        worker = partial(process_stock_data, data_dir=DATA_DIR)
        results = pool.map(worker, symbols)
        
    for sym, df in results:
        if df is not None:
            all_data[sym] = df
            all_dates.update(df.index.tolist())

    sorted_dates = sorted(list(all_dates))
    # 只看最近兩年
    bt_start_date = datetime.now() - timedelta(days=2*365)
    sorted_dates = [d for d in sorted_dates if d >= bt_start_date]

    print(f"🚀 啟動半導體回測，總交易天數: {len(sorted_dates)}")
    if len(sorted_dates) == 0:
        print("❌ 錯誤: 沒有可用的回測日期。請檢查數據範圍。")
        return None
    
    capital = CAPITAL
    active_positions = []
    trade_log = []
    analyzer = IntegratedStockAnalyzer()
    # 預設中性情緒以加速回測，避免重複下載大盤數據
    analyzer.market_sentiment = {'sentiment': 'neutral', 'score': 50, 'factors': ['Backtest Neutral']}
    # 禁用 MTF 分析以加速回測，防止回圈內大量下載數據
    analyzer.mtf_analyzer.calculate_multi_timeframe_score = lambda sym: {
        'final_score': 50, 
        'trend_consistency': 0.5, 
        'recommendation': ['MTF Disabled for Speed']
    }

    total_days = len(sorted_dates)
    for idx, date in enumerate(sorted_dates):
        # 顯示每日進度
        print(f"  📅 處理日期: {date.date()} ({idx+1}/{total_days})", flush=True)
        
        # --- A. 出場檢查 ---
        for pos in active_positions[:]:
            df = all_data[pos['symbol']]
            if date in df.index:
                loc = df.index.get_loc(date)
                # 準備分析快照
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
                    'confidence_factors': [] # 為了速度
                }
                
                # 執行智能出場
                sar_res = evaluate_smart_sar_exit(pos, current_analysis, current_dt=date)
                conf_res = evaluate_exit_confidence(pos, current_analysis)
                
                if sar_res['should_exit'] or conf_res['exit_confidence'] >= 0.8:
                    exit_p = df.loc[date, 'Close']
                    pnl = (exit_p - pos['entry_price']) / pos['entry_price'] - COST
                    capital += (pos['entry_price'] * pos['shares']) * (1 + pnl)
                    trade_log.append({
                        'symbol': pos['symbol'], 
                        'entry_date': str(pos['entry_date'].date()),
                        'exit_date': str(date.date()),
                        'pnl': round(pnl * 100, 2)
                    })
                    active_positions.remove(pos)

        # --- B. 進場掃描 ---
        if len(active_positions) < MAX_POSITIONS:
            # 大盤溫度計
            if date in mdf.index:
                m_close = mdf.loc[date, 'Close']
                m_ma5 = mdf.loc[date, 'MA5']
                threshold = ENTRY_THRESHOLD_BASE * (1.15 if m_close <= m_ma5 else 1.0)
            else:
                threshold = ENTRY_THRESHOLD_BASE

            candidates = []
            for sym, df in all_data.items():
                if any(p['symbol'] == sym for p in active_positions): continue
                if date in df.index:
                    loc = df.index.get_loc(date)
                    if loc < 60: continue
                    
                    try:
                        # 模擬 Analyzer 評估
                        _, score, _, factors = analyzer.assess_entry_opportunity(df.iloc[:loc+1])
                        if score >= threshold:
                            candidates.append({
                                'symbol': sym, 
                                'score': score, 
                                'price': df.loc[date, 'Close'], 
                                'factors': factors
                            })
                    except:
                        pass
            
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            while len(active_positions) < MAX_POSITIONS and candidates:
                c = candidates.pop(0)
                invest = capital * POS_SIZE_PCT
                shares = invest / c['price']
                active_positions.append({
                    'symbol': c['symbol'],
                    'entry_price': c['price'],
                    'entry_date': date,
                    'shares': shares,
                    'initial_analysis_snapshot': {'confidence_factors': c['factors']}
                })
                capital -= invest

    final_val = capital + sum([all_data[p['symbol']].loc[sorted_dates[-1], 'Close'] * p['shares'] for p in active_positions if sorted_dates[-1] in all_data[p['symbol']].index])
    wr = len([t for t in trade_log if t['pnl'] > 0]) / len(trade_log) if trade_log else 0
    
    report = {
        "strategy": "BullPS-TW-Optimized-v1",
        "period": "2y (Semiconductor Focus)",
        "initial_capital": CAPITAL,
        "final_equity": round(final_val, 2),
        "total_return": f"{round((final_val - CAPITAL)/CAPITAL * 100, 2)}%",
        "win_rate": f"{round(wr * 100, 2)}%",
        "total_trades": len(trade_log),
        "trades": trade_log
    }
    
    with open(REPORT_FILE, "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report

if __name__ == "__main__":
    res = run_backtest()
    print(f"\n📊 回測報告摘要：")
    print(f"總投報率: {res['total_return']}")
    print(f"勝率: {res['win_rate']}")
    print(f"交易次數: {res['total_trades']}")
    print(f"最終資產: {res['final_equity']}")
