"""
🇹🇼 Portfolio Simulator V12 Soul - 靈魂回歸版 (理由侵蝕核心)
=========================================================
回歸最初核心：支撐簇 (Support Clusters) 抄底 + 理由侵蝕 (Reasoning Erosion) 出場。
不設硬性止損百分比，完全交由「技術理由消失」決定。
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

# ============ V12 Soul 參數 ============
# 降低大盤門檻，允許在恐慌末端的支撐區進場
MARKET_RSI_THRESHOLD = 35 
# 移除 1.3x 放量限制，改為靈敏的轉強信號
ENTRY_SCORE_THRESHOLD = 60
SUPPORT_RELIABILITY_MIN = 50 # 至少 2 個支撐位重疊

# 出場邏輯：理由侵蝕
EROSION_THRESHOLD = 0.7 # 當前分數低於進場時的 70% 時出場
ABSOLUTE_EXIT_SCORE = 35 # 無論如何分數低於 35 就撤退

CAPITAL = 1000000
MAX_POSITIONS = 5
COST = 0.005
# =======================================

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_stock_data(sym, data_dir):
    p = f"{data_dir}/{sym}.csv"
    force_download = False
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            # 檢查是否有 2023 年之前的數據 (確保回測有足夠歷史計算指標)
            if df.index.min() > pd.Timestamp("2023-01-01"):
                force_download = True
            else:
                return df
        except:
            force_download = True
    
    if force_download or not os.path.exists(p):
        print(f"   [Data] 下載 {sym}.TW 完整歷史數據...")
        try:
            yf_df = yf.download(f"{sym}.TW", period='5y', interval='1d', progress=False)
            if not yf_df.empty:
                if isinstance(yf_df.columns, pd.MultiIndex):
                    yf_df.columns = yf_df.columns.get_level_values(0)
                yf_df.index.name = 'Date'
                yf_df.to_csv(p)
                return yf_df
        except: pass
    return None

def pre_analyze_stock_v12(sym, data_dir, bt_start_date):
    """V12 預分析：核心在於紀錄完整的支撐強度與總分"""
    try:
        df = get_stock_data(sym, data_dir)
        if df is None or len(df) < 200: # 需要足夠長度計算 MA200
            return sym, None
        
        analyzer = IntegratedStockAnalyzer()
        # 設置當前符號
        analyzer.current_symbol = f"{sym}.TW"
        # 簡化 MTF 與市場情緒，避免回測中網路請求
        analyzer.mtf_analyzer.calculate_multi_timeframe_score = lambda s: {'final_score': 50, 'recommendation': []}
        analyzer.market_sentiment = {'sentiment': 'neutral', 'score': 50}
        
        # 使用完整的 df 計算指標 (確保 MA200 等有數據)
        df = analyzer.calculate_technical_indicators(df)
        if df is None: return sym, None
        
        results = {}
        # 只在回測區間內運行
        backtest_range = df[df.index >= bt_start_date]
        
        for date in backtest_range.index:
            # 取得該日期前的所有數據
            loc = df.index.get_loc(date)
            upto_date = df.iloc[:loc+1]
            
            if len(upto_date) < 60: continue
            
            try:
                # 評估進場機會
                entry_advice, score, level, factors = analyzer.assess_entry_opportunity(upto_date)
                
                cur = upto_date.iloc[-1]
                results[date] = {
                    'score': float(score),
                    'support_reliability': float(cur['Support_Reliability']) if 'Support_Reliability' in cur else 0,
                    'close': float(cur['Close']),
                    'sar': float(cur['SAR']) if 'SAR' in cur else 0,
                    'rsi': float(cur['RSI']) if 'RSI' in cur else 0,
                    'factors': factors
                }
            except Exception as e:
                # print(f"DEBUG {sym} at {date} failed: {e}")
                continue
                
        return sym, results
    except Exception as e:
        print(f"CRITICAL {sym} failed: {e}")
        return sym, None

def run_v12_backtest(sector_name, symbols):
    print(f"\n🔥 啟動 V12 Soul 回測 (理由侵蝕模型): {sector_name}")
    data_dir = "TW-Quant-Shioaji/data/batch_150"
    index_path = "TW-Quant-Shioaji/data/batch_semi/twii.csv"
    bt_start_date = datetime(2024, 2, 15)
    
    market_df = pd.read_csv(index_path)
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    market_df.set_index('Date', inplace=True)
    market_df['RSI'] = calculate_rsi(market_df['Close'])
    
    all_signals = {}
    with multiprocessing.Pool() as pool:
        worker = partial(pre_analyze_stock_v12, data_dir=data_dir, bt_start_date=bt_start_date)
        results = pool.map(worker, symbols)
    
    for sym, signals in results:
        if signals: 
            all_signals[sym] = signals
        else:
            print(f"   [Warning] {sym} 無分析信號")
    
    if not all_signals: 
        print(f"   [Error] {sector_name} 無法獲取任何信號")
        return None
    
    all_dates = sorted(list(set().union(*(s.keys() for s in all_signals.values()))))
    
    capital = CAPITAL
    active_positions = [] # {symbol, entry_price, shares, entry_date, entry_score}
    trade_log = []
    
    for date in all_dates:
        market_rsi = market_df.loc[date, 'RSI'] if date in market_df.index else None
        market_ok = (market_rsi is not None and market_rsi >= MARKET_RSI_THRESHOLD)
        
        # --- V12 理由侵蝕出場 ---
        for pos in active_positions[:]:
            sym = pos['symbol']
            if date not in all_signals[sym]: continue
            
            cur = all_signals[sym][date]
            current_score = cur['score']
            price = cur['close']
            
            exit_reason = None
            # 條件 1: 分數侵蝕到進場時的 70% 以下 (理由瓦解)
            if current_score < pos['entry_score'] * EROSION_THRESHOLD:
                exit_reason = f"Erosion (Score {current_score:.1f} < {pos['entry_score']*0.7:.1f})"
            
            # 條件 2: 絕對信心崩潰
            elif current_score < ABSOLUTE_EXIT_SCORE:
                exit_reason = f"Confidence Collapse ({current_score:.1f})"

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

        # --- V12 支撐簇進場 ---
        if len(active_positions) < MAX_POSITIONS:
            candidates = []
            for sym, signals in all_signals.items():
                if any(p['symbol'] == sym for p in active_positions): continue
                if date not in signals: continue
                
                cur = signals[date]
                # 核心進場：支撐強度 + 基礎分數
                if cur['score'] >= ENTRY_SCORE_THRESHOLD and cur['support_reliability'] >= SUPPORT_RELIABILITY_MIN:
                    if market_ok:
                        candidates.append({'symbol': sym, 'score': cur['score'], 'price': cur['close']})
            
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            for c in candidates:
                if len(active_positions) >= MAX_POSITIONS: break
                invest = capital * 0.2
                shares = invest / c['price']
                active_positions.append({
                    'symbol': c['symbol'], 'entry_price': c['price'], 'shares': shares,
                    'entry_date': date, 'entry_score': c['score']
                })
                capital -= invest
                
    # 統計
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
    
    v12_results = []
    for sector, items in watchlist.items():
        codes = [item['code'] for item in items]
        res = run_v12_backtest(sector, codes)
        if res: v12_results.append(res)
    
    print("\n" + "👻" * 25)
    print("👻 V12 Soul (理由侵蝕版) 最終回測結果")
    print("👻" * 25)
    print(f"{'板塊':<20} | {'報酬率':<8} | {'勝率':<8} | {'盈虧比':<6}")
    print("-" * 55)
    for r in v12_results:
        print(f"{r['sector']:<20} | {r['total_return']:<8} | {r['win_rate']:<8} | {r['pnl_ratio']:<6}")
    print("=" * 55)

if __name__ == "__main__":
    main()
