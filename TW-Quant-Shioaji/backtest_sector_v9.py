"""
🇹🇼 TW-Quant V9 Sector Backtest - Specialized Runner
=====================================================
針對特定板塊執行 V9 策略回測，並自動補齊缺失數據。
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

# ============ V9 參數 ============
MARKET_RSI_THRESHOLD = 40
VOLUME_CONFIRMATION = 1.3
ENTRY_SCORE_THRESHOLD = 65

CAPITAL = 1000000
MAX_POSITIONS = 5
COST = 0.005 # 交易手續費 + 滑價 (單邊)
# =================================

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_stock_data(sym, data_dir):
    """取得股票數據，若本地沒有則從 yfinance 下載"""
    p = f"{data_dir}/{sym}.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        # 檢查數據是否太舊 (例如少於 250 筆)
        if len(df) > 200:
            return df

    # 從 yfinance 下載
    print(f"   [Data] 下載 {sym}.TW 歷史數據...")
    try:
        yf_df = yf.download(f"{sym}.TW", period='2y', interval='1d', progress=False)
        if not yf_df.empty:
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = yf_df.columns.get_level_values(0)
            yf_df.index.name = 'Date'
            # 存檔供以後使用
            yf_df.to_csv(p)
            return yf_df
    except Exception as e:
        print(f"   [Error] 下載 {sym} 失敗: {e}")
    return None

def pre_analyze_stock(sym, data_dir, bt_start_date):
    """預先分析單一股票的所有交易日分數"""
    try:
        df = get_stock_data(sym, data_dir)
        if df is None or len(df) < 60: 
            return sym, None
        
        # 成交量 MA20
        df['Volume_MA20'] = df['Volume'].rolling(20).mean()
        
        # 技術指標
        analyzer = IntegratedStockAnalyzer()
        # 簡化 MTF 避免回測中呼叫外部 API
        analyzer.mtf_analyzer.calculate_multi_timeframe_score = lambda s: {
            'final_score': 50, 'trend_consistency': 0.5, 'recommendation': []
        }
        analyzer.market_sentiment = {'sentiment': 'neutral', 'score': 50}
        
        df = analyzer.calculate_technical_indicators(df)
        
        # 預計算每日分數
        results = {}
        # 只計算回測開始日期前 90 天到現在的數據
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
            }
        
        return sym, results
    except Exception as e:
        print(f"Error pre-analyzing {sym}: {e}")
        return sym, None

def run_sector_backtest(sector_name, symbols):
    print(f"\n🚀 開始回測板塊: {sector_name} ({len(symbols)} 檔標的)")
    
    data_dir = "TW-Quant-Shioaji/data/batch_150"
    index_path = "TW-Quant-Shioaji/data/batch_semi/twii.csv"
    bt_start_date = datetime(2024, 2, 15)
    
    # 1. 載入大盤
    market_df = pd.read_csv(index_path)
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    market_df.set_index('Date', inplace=True)
    market_df['RSI'] = calculate_rsi(market_df['Close'])
    
    # 2. 預分析所有股票
    all_signals = {}
    print(f"   正在分析指標 (使用多進程)...")
    with multiprocessing.Pool() as pool:
        worker = partial(pre_analyze_stock, data_dir=data_dir, bt_start_date=bt_start_date)
        results = pool.map(worker, symbols)
    
    for sym, signals in results:
        if signals:
            all_signals[sym] = signals
    
    print(f"   分析完成，成功標的: {len(all_signals)} 檔")
    if not all_signals: return None
    
    # 3. 取得所有交易日
    all_dates = set()
    for signals in all_signals.values():
        all_dates.update(signals.keys())
    sorted_dates = sorted(list(all_dates))
    
    # 4. 回測循環
    capital = CAPITAL
    active_positions = []
    trade_log = []
    
    filtered_by_market = 0
    filtered_by_volume = 0
    total_candidates = 0
    
    for date in sorted_dates:
        # 大盤 RSI 過濾
        market_rsi = market_df.loc[date, 'RSI'] if date in market_df.index else None
        market_ok = (market_rsi is not None and not pd.isna(market_rsi) and market_rsi >= MARKET_RSI_THRESHOLD)
        
        # --- 出場邏輯 ---
        for pos in active_positions[:]:
            sym = pos['symbol']
            if sym not in all_signals or date not in all_signals[sym]:
                continue
            
            cur = all_signals[sym][date]
            # SAR 反轉出場
            if cur['close'] < cur['sar']:
                pnl = (cur['close'] - pos['entry_price']) / pos['entry_price'] - COST
                capital += (pos['entry_price'] * pos['shares']) * (1 + pnl)
                trade_log.append({
                    'symbol': sym,
                    'pnl': round(pnl * 100, 2),
                    'entry_date': str(pos['entry_date'].date()),
                    'exit_date': str(date.date()),
                    'reason': 'SAR'
                })
                active_positions.remove(pos)
        
        # --- 進場邏輯 ---
        if len(active_positions) < MAX_POSITIONS:
            candidates = []
            for sym, signals in all_signals.items():
                if any(p['symbol'] == sym for p in active_positions):
                    continue
                if date not in signals:
                    continue
                
                cur = signals[date]
                if cur['score'] >= ENTRY_SCORE_THRESHOLD:
                    total_candidates += 1
                    # V9 過濾 1: 大盤 RSI
                    if not market_ok:
                        filtered_by_market += 1
                        continue
                    # V9 過濾 2: 成交量 1.3x
                    if cur['volume_ma20'] <= 0 or cur['volume'] < cur['volume_ma20'] * VOLUME_CONFIRMATION:
                        filtered_by_volume += 1
                        continue
                    
                    candidates.append({
                        'symbol': sym,
                        'score': cur['score'],
                        'price': cur['close']
                    })
            
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            for c in candidates:
                if len(active_positions) >= MAX_POSITIONS: break
                invest = capital * 0.2 # 每個部位佔剩餘資金 20% (Max 5 positions)
                shares = invest / c['price']
                active_positions.append({
                    'symbol': c['symbol'],
                    'entry_price': c['price'],
                    'entry_date': date,
                    'shares': shares
                })
                capital -= invest
    
    # 5. 計算結果
    final_val = capital + sum([all_signals[p['symbol']][sorted_dates[-1]]['close'] * p['shares'] 
                               for p in active_positions if sorted_dates[-1] in all_signals[p['symbol']]])
    winning = [t for t in trade_log if t['pnl'] > 0]
    wr = len(winning) / len(trade_log) if trade_log else 0
    
    report = {
        "sector": sector_name,
        "results": {
            "total_return": f"{round((final_val - CAPITAL)/CAPITAL * 100, 2)}%",
            "win_rate": f"{round(wr * 100, 2)}%",
            "total_trades": len(trade_log)
        },
        "filters": {
            "total_scored_ge_65": total_candidates,
            "blocked_by_market_rsi": filtered_by_market,
            "blocked_by_volume": filtered_by_volume,
            "passed_to_trade": total_candidates - filtered_by_market - filtered_by_volume
        }
    }
    return report

def main():
    with open("TW-Quant-Shioaji/v9_sector_watchlist.json", "r", encoding='utf-8') as f:
        watchlist = json.load(f)
    
    summary = []
    for sector, items in watchlist.items():
        codes = [item['code'] for item in items]
        report = run_sector_backtest(sector, codes)
        if report:
            summary.append(report)
            # 存檔個別報告
            safe_name = sector.replace("、", "_")
            with open(f"TW-Quant-Shioaji/sector_reports/v9_{safe_name}.json", "w", encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*50)
    print("📊 V9 板塊回測彙整報告")
    print("="*50)
    print(f"{'板塊':<20} | {'報酬率':<8} | {'勝率':<8} | {'交易數':<6}")
    print("-" * 50)
    for s in summary:
        print(f"{s['sector']:<20} | {s['results']['total_return']:<8} | {s['results']['win_rate']:<8} | {s['results']['total_trades']:<6}")
    print("="*50)

if __name__ == "__main__":
    os.makedirs("TW-Quant-Shioaji/sector_reports", exist_ok=True)
    main()
