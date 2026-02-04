"""
🇹🇼 Portfolio Simulator V10 - 大賺小賠優化版 (Erosion & Trailing Stop)
=====================================================================
基於 V9 強化進場，徹底改革出場邏輯：
1. 進場：延用 V9 (大盤 RSI > 40, 量比 > 1.3, 分數 >= 65)
2. 出場 A (信心侵蝕)：獲利 > 5% 後，忽略 SAR 震盪，直到分數跌破 40 才出場。
3. 出場 B (移動止盈)：獲利 > 12% 後，啟動移動止盈 (回落 5% 強制平倉)。
4. 出場 C (原始止損)：若獲利未達 5%，維持 SAR 快速止損。
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import yfinance as yf
from datetime import datetime, timedelta
import multiprocessing
from functools import partial

# 加入專案路徑
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "TW-Quant-Shioaji"))

from core.integrated_stock_analyzer import IntegratedStockAnalyzer

# ============ V10 參數 ============
MARKET_RSI_THRESHOLD = 40
VOLUME_CONFIRMATION = 1.3
ENTRY_SCORE_THRESHOLD = 65

# V10 新增出場參數
EXIT_SCORE_THRESHOLD = 40      # 信心崩潰門檻
PROFIT_PROTECTION_LV1 = 0.05   # 5% 獲利後進入「信心侵蝕」模式
PROFIT_PROTECTION_LV2 = 0.12   # 12% 獲利後啟動「移動止盈」
TRAILING_STOP_PCT = 0.05       # 從最高點回落 5% 出場

CAPITAL = 1000000
MAX_POSITIONS = 5
COST = 0.005 # 單邊交易成本
# =================================

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_stock_data(sym, data_dir):
    p = f"{data_dir}/{sym}.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        if len(df) > 200: return df
    
    # 數據補齊邏輯
    try:
        yf_df = yf.download(f"{sym}.TW", period='2y', interval='1d', progress=False)
        if not yf_df.empty:
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = yf_df.columns.get_level_values(0)
            yf_df.index.name = 'Date'
            yf_df.to_csv(p)
            return yf_df
    except: pass
    return None

def pre_analyze_stock_v10(sym, data_dir, bt_start_date):
    """V10 預分析：包含分數紀錄供侵蝕模型判斷"""
    try:
        df = get_stock_data(sym, data_dir)
        if df is None or len(df) < 60: return sym, None
        
        df['Volume_MA20'] = df['Volume'].rolling(20).mean()
        analyzer = IntegratedStockAnalyzer()
        # 簡化 MTF
        analyzer.mtf_analyzer.calculate_multi_timeframe_score = lambda s: {
            'final_score': 50, 'trend_consistency': 0.5, 'recommendation': []
        }
        analyzer.market_sentiment = {'sentiment': 'neutral', 'score': 50}
        df = analyzer.calculate_technical_indicators(df)
        
        results = {}
        test_df = df[df.index >= bt_start_date - timedelta(days=90)]
        
        for i in range(len(test_df)):
            date = test_df.index[i]
            if date < bt_start_date: continue
            
            upto_date = test_df.iloc[:i+1]
            if len(upto_date) < 60: continue
            
            _, score, _, factors = analyzer.assess_entry_opportunity(upto_date)
            cur = upto_date.iloc[-1]
            results[date] = {
                'score': score,
                'close': float(cur['Close']),
                'sar': float(cur['SAR']),
                'volume': float(cur['Volume']),
                'volume_ratio': float(cur['Volume'] / cur['Volume_MA20']) if cur['Volume_MA20'] > 0 else 0,
            }
        return sym, results
    except Exception as e:
        return sym, None

def run_v10_backtest(sector_name, symbols):
    print(f"\n💎 啟動 V10 優化回測: {sector_name}")
    data_dir = "TW-Quant-Shioaji/data/batch_150"
    index_path = "TW-Quant-Shioaji/data/batch_semi/twii.csv"
    bt_start_date = datetime(2024, 2, 15)
    
    # 載入大盤
    market_df = pd.read_csv(index_path)
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    market_df.set_index('Date', inplace=True)
    market_df['RSI'] = calculate_rsi(market_df['Close'])
    
    # 預分析
    all_signals = {}
    with multiprocessing.Pool() as pool:
        worker = partial(pre_analyze_stock_v10, data_dir=data_dir, bt_start_date=bt_start_date)
        results = pool.map(worker, symbols)
    
    for sym, signals in results:
        if signals: all_signals[sym] = signals
    
    if not all_signals: return None
    
    all_dates = sorted(list(set().union(*(s.keys() for s in all_signals.values()))))
    
    capital = CAPITAL
    active_positions = [] # {symbol, entry_price, shares, entry_date, peak_price, mode}
    trade_log = []
    
    for date in all_dates:
        market_rsi = market_df.loc[date, 'RSI'] if date in market_df.index else None
        market_ok = (market_rsi is not None and market_rsi >= MARKET_RSI_THRESHOLD)
        
        # --- V10 改良出場邏輯 ---
        for pos in active_positions[:]:
            sym = pos['symbol']
            if date not in all_signals[sym]: continue
            
            cur = all_signals[sym][date]
            price = cur['close']
            pos['peak_price'] = max(pos['peak_price'], price)
            
            # 計算當前獲利
            pnl_pct = (price - pos['entry_price']) / pos['entry_price']
            exit_reason = None
            
            # 1. 移動止盈 (Protection LV2)
            if pnl_pct >= PROFIT_PROTECTION_LV2:
                if price < pos['peak_price'] * (1 - TRAILING_STOP_PCT):
                    exit_reason = "Trailing Stop"
            
            # 2. 信心侵蝕 (Protection LV1)
            elif pnl_pct >= PROFIT_PROTECTION_LV1:
                # 獲利已達 5%，忽略 SAR，直到分數跌破門檻
                if cur['score'] < EXIT_SCORE_THRESHOLD:
                    exit_reason = "Erosion (Score < 40)"
            
            # 3. 原始 SAR 止損
            else:
                if price < cur['sar']:
                    exit_reason = "SAR Stop Loss"

            if exit_reason:
                final_pnl = (price - pos['entry_price']) / pos['entry_price'] - COST
                capital += (pos['entry_price'] * pos['shares']) * (1 + final_pnl)
                trade_log.append({
                    'symbol': sym,
                    'pnl': round(final_pnl * 100, 2),
                    'entry_date': str(pos['entry_date'].date()),
                    'exit_date': str(date.date()),
                    'reason': exit_reason
                })
                active_positions.remove(pos)

        # --- V9 進場邏輯 ---
        if len(active_positions) < MAX_POSITIONS:
            candidates = []
            for sym, signals in all_signals.items():
                if any(p['symbol'] == sym for p in active_positions): continue
                if date not in signals: continue
                
                cur = signals[date]
                if cur['score'] >= ENTRY_SCORE_THRESHOLD:
                    if market_ok and cur['volume_ratio'] >= VOLUME_CONFIRMATION:
                        candidates.append({'symbol': sym, 'score': cur['score'], 'price': cur['close']})
            
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            for c in candidates:
                if len(active_positions) >= MAX_POSITIONS: break
                invest = capital * 0.2
                shares = invest / c['price']
                active_positions.append({
                    'symbol': c['symbol'], 'entry_price': c['price'], 'shares': shares,
                    'entry_date': date, 'peak_price': c['price']
                })
                capital -= invest
                
    # 統計結果
    final_val = capital + sum([all_signals[p['symbol']][all_dates[-1]]['close'] * p['shares'] 
                               for p in active_positions if all_dates[-1] in all_signals[p['symbol']]])
    winning = [t for t in trade_log if t['pnl'] > 0]
    wr = len(winning) / len(trade_log) if trade_log else 0
    avg_win = np.mean([t['pnl'] for t in winning]) if winning else 0
    avg_loss = np.mean([t['pnl'] for t in trade_log if t['pnl'] <= 0]) if trade_log else 0
    
    return {
        "sector": sector_name,
        "total_return": f"{round((final_val - CAPITAL)/CAPITAL * 100, 2)}%",
        "win_rate": f"{round(wr * 100, 2)}%",
        "total_trades": len(trade_log),
        "avg_win": f"{round(avg_win, 2)}%",
        "avg_loss": f"{round(avg_loss, 2)}%",
        "pnl_ratio": f"{abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A"
    }

def main():
    with open("TW-Quant-Shioaji/v9_sector_watchlist.json", "r", encoding='utf-8') as f:
        watchlist = json.load(f)
    
    v10_results = []
    for sector, items in watchlist.items():
        codes = [item['code'] for item in items]
        res = run_v10_backtest(sector, codes)
        if res: v10_results.append(res)
    
    print("\n" + "🏆" * 25)
    print("🏆 V10 (大賺小賠優化版) 最終回測結果")
    print("🏆" * 25)
    print(f"{'板塊':<20} | {'報酬率':<8} | {'勝率':<8} | {'盈虧比':<6}")
    print("-" * 55)
    for r in v10_results:
        print(f"{r['sector']:<20} | {r['total_return']:<8} | {r['win_rate']:<8} | {r['pnl_ratio']:<6}")
    print("=" * 55)

if __name__ == "__main__":
    main()
