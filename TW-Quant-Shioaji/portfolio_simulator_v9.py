"""
Portfolio Simulator V9 - 勝率優化版 (預計算模式)
================================================
基於 V8 架構，新增：
1. 大盤過濾：大盤 RSI < 40 時不進場
2. 放量確認：進場當天成交量 > MA20 × 1.3
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
import multiprocessing
from functools import partial

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "TW-Quant-Shioaji"))

from core.integrated_stock_analyzer import IntegratedStockAnalyzer
from core.portfolio_logic import evaluate_smart_sar_exit, evaluate_exit_confidence

# ============ V9 參數 ============
MARKET_RSI_THRESHOLD = 40
VOLUME_CONFIRMATION = 1.3
ENTRY_SCORE_THRESHOLD = 65

CAPITAL = 1000000
MAX_POSITIONS = 5
COST = 0.005
# =================================

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def pre_analyze_stock_v9(sym, data_dir, bt_start_date):
    """預先分析單一股票的所有交易日分數"""
    try:
        p = f"{data_dir}/{sym}.csv"
        if not os.path.exists(p): 
            return sym, None
        
        df = pd.read_csv(p)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 成交量 MA20
        df['Volume_MA20'] = df['Volume'].rolling(20).mean()
        
        # 技術指標
        analyzer = IntegratedStockAnalyzer()
        # 簡化 MTF 避免 API 呼叫
        analyzer.mtf_analyzer.calculate_multi_timeframe_score = lambda s: {
            'final_score': 50, 'trend_consistency': 0.5, 'recommendation': []
        }
        analyzer.market_sentiment = {'sentiment': 'neutral', 'score': 50}
        
        df = analyzer.calculate_technical_indicators(df)
        
        # 預計算每日分數
        results = {}
        test_df = df[df.index >= bt_start_date - timedelta(days=90)]
        
        for i in range(len(test_df)):
            date = test_df.index[i]
            if date < bt_start_date: 
                continue
            
            upto_date = test_df.iloc[:i+1]
            if len(upto_date) < 60: 
                continue
            
            _, score, _, factors = analyzer.assess_entry_opportunity(upto_date)
            
            cur = upto_date.iloc[-1]
            results[date] = {
                'score': score,
                'factors': factors,
                'close': float(cur['Close']),
                'sar': float(cur['SAR']),
                'rsi': float(cur['RSI']),
                'macd': float(cur['MACD']),
                'volume': float(cur['Volume']),
                'volume_ma20': float(cur['Volume_MA20']) if not pd.isna(cur['Volume_MA20']) else 0,
                'ma20': float(cur['MA20']),
                'ma5': float(cur['MA5']),
            }
        
        return sym, results
    except Exception as e:
        print(f"Error pre-analyzing {sym}: {e}")
        return sym, None

def run_v9_backtest():
    # 載入監控清單
    with open("TW-Quant-Shioaji/strong_sectors_watchlist.json", "r") as f:
        watchlist = json.load(f)
    symbols = watchlist.get("all_codes", [])
    
    data_dir = "TW-Quant-Shioaji/data/batch_150"
    index_path = "TW-Quant-Shioaji/data/batch_semi/twii.csv"
    bt_start_date = datetime(2024, 2, 15)
    
    # 載入大盤
    print("載入大盤數據...")
    market_df = pd.read_csv(index_path)
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    market_df.set_index('Date', inplace=True)
    market_df['RSI'] = calculate_rsi(market_df['Close'])
    
    # 預分析所有股票 (使用多進程)
    print(f"預分析 {len(symbols)} 檔股票...")
    
    all_signals = {}
    with multiprocessing.Pool() as pool:
        worker = partial(pre_analyze_stock_v9, data_dir=data_dir, bt_start_date=bt_start_date)
        results = pool.map(worker, symbols)
    
    for sym, signals in results:
        if signals:
            all_signals[sym] = signals
    
    print(f"成功分析: {len(all_signals)} 檔")
    
    if not all_signals:
        print("錯誤：沒有成功分析任何股票")
        return None
    
    # 取得所有交易日
    all_dates = set()
    for signals in all_signals.values():
        all_dates.update(signals.keys())
    sorted_dates = sorted(list(all_dates))
    
    print(f"開始回測，天數: {len(sorted_dates)}")
    
    # 回測
    capital = CAPITAL
    active_positions = []
    trade_log = []
    
    filtered_by_market = 0
    filtered_by_volume = 0
    total_candidates = 0
    
    for date in sorted_dates:
        # 大盤 RSI
        market_rsi = None
        if date in market_df.index:
            market_rsi = market_df.loc[date, 'RSI']
        market_ok = (market_rsi is not None and not pd.isna(market_rsi) and market_rsi >= MARKET_RSI_THRESHOLD)
        
        # --- 出場 ---
        for pos in active_positions[:]:
            if pos['symbol'] not in all_signals:
                continue
            signals = all_signals[pos['symbol']]
            if date not in signals:
                continue
            
            cur = signals[date]
            # SAR 反轉出場
            if cur['close'] < cur['sar']:
                pnl = (cur['close'] - pos['entry_price']) / pos['entry_price'] - COST
                capital += (pos['entry_price'] * pos['shares']) * (1 + pnl)
                trade_log.append({
                    'symbol': pos['symbol'],
                    'pnl': round(pnl * 100, 2),
                    'entry_date': str(pos['entry_date'].date()),
                    'exit_date': str(date.date()),
                    'reason': 'SAR'
                })
                active_positions.remove(pos)
        
        # --- 進場 ---
        if len(active_positions) < MAX_POSITIONS:
            candidates = []
            for sym, signals in all_signals.items():
                if any(p['symbol'] == sym for p in active_positions):
                    continue
                if date not in signals:
                    continue
                
                cur = signals[date]
                score = cur['score']
                
                if score >= ENTRY_SCORE_THRESHOLD:
                    total_candidates += 1
                    
                    # 過濾器 1: 大盤 RSI
                    if not market_ok:
                        filtered_by_market += 1
                        continue
                    
                    # 過濾器 2: 放量確認
                    if cur['volume_ma20'] <= 0 or cur['volume'] < cur['volume_ma20'] * VOLUME_CONFIRMATION:
                        filtered_by_volume += 1
                        continue
                    
                    candidates.append({
                        'symbol': sym,
                        'score': score,
                        'price': cur['close'],
                        'volume_ratio': cur['volume'] / cur['volume_ma20']
                    })
            
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            
            for c in candidates:
                if len(active_positions) >= MAX_POSITIONS:
                    break
                invest = capital * 0.1
                shares = invest / c['price']
                active_positions.append({
                    'symbol': c['symbol'],
                    'entry_price': c['price'],
                    'entry_date': date,
                    'shares': shares
                })
                capital -= invest
    
    # 計算結果
    final_val = capital + sum([p['entry_price'] * p['shares'] for p in active_positions])
    winning = [t for t in trade_log if t['pnl'] > 0]
    wr = len(winning) / len(trade_log) if trade_log else 0
    avg_win = np.mean([t['pnl'] for t in winning]) if winning else 0
    losing = [t for t in trade_log if t['pnl'] <= 0]
    avg_loss = np.mean([t['pnl'] for t in losing]) if losing else 0
    
    report = {
        "strategy": "BullPS-TW-V9-WinRate-Optimizer",
        "parameters": {
            "market_rsi_threshold": MARKET_RSI_THRESHOLD,
            "volume_confirmation": VOLUME_CONFIRMATION,
            "entry_score_threshold": ENTRY_SCORE_THRESHOLD
        },
        "results": {
            "initial_capital": CAPITAL,
            "final_equity": round(final_val, 2),
            "total_return": f"{round((final_val - CAPITAL)/CAPITAL * 100, 2)}%",
            "win_rate": f"{round(wr * 100, 2)}%",
            "total_trades": len(trade_log),
            "avg_win": f"{round(avg_win, 2)}%",
            "avg_loss": f"{round(avg_loss, 2)}%"
        },
        "filter_stats": {
            "total_candidates": total_candidates,
            "filtered_by_market_rsi": filtered_by_market,
            "filtered_by_volume": filtered_by_volume,
            "passed": total_candidates - filtered_by_market - filtered_by_volume
        },
        "trades": trade_log
    }
    
    with open("TW-Quant-Shioaji/portfolio_report_v9.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*50)
    print("V9 回測完成!")
    print("="*50)
    print(f"總報酬: {report['results']['total_return']}")
    print(f"勝率: {report['results']['win_rate']}")
    print(f"交易數: {report['results']['total_trades']}")
    print(f"平均獲利: {report['results']['avg_win']}")
    print(f"平均虧損: {report['results']['avg_loss']}")
    print("-"*50)
    print(f"過濾統計:")
    print(f"  候選: {total_candidates}")
    print(f"  大盤過濾: {filtered_by_market}")
    print(f"  成交量過濾: {filtered_by_volume}")
    print(f"  通過: {report['filter_stats']['passed']}")
    print("="*50)
    
    return report

if __name__ == "__main__":
    run_v9_backtest()
