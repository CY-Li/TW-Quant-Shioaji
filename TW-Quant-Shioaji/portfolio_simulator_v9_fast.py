"""
Portfolio Simulator V9 Fast - 勝率優化版 (簡化版)
==================================================
快速驗證進場過濾器效果
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "TW-Quant-Shioaji"))
from core.integrated_stock_analyzer import IntegratedStockAnalyzer

# ============ V9 參數 ============
MARKET_RSI_THRESHOLD = 40
VOLUME_CONFIRMATION = 1.3
ENTRY_SCORE_THRESHOLD = 65
# =================================

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_v9_fast():
    # 載入監控清單
    with open("TW-Quant-Shioaji/strong_sectors_watchlist.json", "r") as f:
        symbols = json.load(f)['all_codes']
    
    data_dir = "TW-Quant-Shioaji/data/batch_150"
    index_path = "TW-Quant-Shioaji/data/batch_semi/twii.csv"
    
    # 載入大盤
    print("載入大盤...")
    market_df = pd.read_csv(index_path)
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    market_df.set_index('Date', inplace=True)
    market_df['RSI'] = calculate_rsi(market_df['Close'])
    
    # 載入個股
    print("載入個股數據...")
    all_data = {}
    all_dates = set()
    analyzer = IntegratedStockAnalyzer()
    
    for sym in symbols:
        p = f"{data_dir}/{sym}.csv"
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df['Volume_MA20'] = df['Volume'].rolling(20).mean()
            df = analyzer.calculate_technical_indicators(df)
            all_data[sym] = df
            all_dates.update(df.index.tolist())
        except Exception as e:
            print(f"Error {sym}: {e}")
    
    print(f"成功載入: {len(all_data)} 檔")
    
    # 回測設定
    initial_capital = 1000000
    capital = initial_capital
    max_positions = 10
    pos_size_pct = 0.1
    COST = 0.005
    
    active_positions = []
    trade_log = []
    filtered_by_market = 0
    filtered_by_volume = 0
    total_candidates = 0
    
    sorted_dates = sorted(list(all_dates))
    print(f"開始回測，天數: {len(sorted_dates)}")
    
    for i, date in enumerate(sorted_dates):
        if i % 50 == 0:
            print(f"進度: {i}/{len(sorted_dates)}")
        
        # 大盤 RSI
        market_rsi = market_df.loc[date, 'RSI'] if date in market_df.index else None
        market_ok = (market_rsi is not None and market_rsi >= MARKET_RSI_THRESHOLD)
        
        # --- 出場 ---
        for pos in active_positions[:]:
            df = all_data[pos['symbol']]
            if date not in df.index:
                continue
            
            cur = df.loc[date]
            # 簡化出場: SAR 反轉
            if cur['Close'] < cur['SAR']:
                exit_p = cur['Close']
                pnl = (exit_p - pos['entry_price']) / pos['entry_price'] - COST
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
        if len(active_positions) < max_positions:
            candidates = []
            for sym, df in all_data.items():
                if any(p['symbol'] == sym for p in active_positions):
                    continue
                if date not in df.index:
                    continue
                
                loc = df.index.get_loc(date)
                if loc < 60:
                    continue
                
                try:
                    _, score, _, factors = analyzer.assess_entry_opportunity(df.iloc[:loc+1])
                except:
                    continue
                
                if score >= ENTRY_SCORE_THRESHOLD:
                    total_candidates += 1
                    
                    # 過濾器 1: 大盤
                    if not market_ok:
                        filtered_by_market += 1
                        continue
                    
                    # 過濾器 2: 成交量
                    cur = df.loc[date]
                    if pd.isna(cur['Volume_MA20']) or cur['Volume'] < cur['Volume_MA20'] * VOLUME_CONFIRMATION:
                        filtered_by_volume += 1
                        continue
                    
                    candidates.append({
                        'symbol': sym,
                        'score': score,
                        'price': cur['Close']
                    })
            
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            
            for c in candidates:
                if len(active_positions) >= max_positions:
                    break
                invest = capital * pos_size_pct
                shares = invest / c['price']
                active_positions.append({
                    'symbol': c['symbol'],
                    'entry_price': c['price'],
                    'entry_date': date,
                    'shares': shares
                })
                capital -= invest
    
    # 結果
    final_val = capital + sum([p['entry_price'] * p['shares'] for p in active_positions])
    winning = [t for t in trade_log if t['pnl'] > 0]
    wr = len(winning) / len(trade_log) if trade_log else 0
    avg_win = np.mean([t['pnl'] for t in winning]) if winning else 0
    losing = [t for t in trade_log if t['pnl'] <= 0]
    avg_loss = np.mean([t['pnl'] for t in losing]) if losing else 0
    
    report = {
        "strategy": "BullPS-TW-V9-Fast",
        "parameters": {
            "market_rsi_threshold": MARKET_RSI_THRESHOLD,
            "volume_confirmation": VOLUME_CONFIRMATION,
            "entry_score_threshold": ENTRY_SCORE_THRESHOLD
        },
        "results": {
            "total_return": f"{round((final_val - initial_capital)/initial_capital * 100, 2)}%",
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
    run_v9_fast()
