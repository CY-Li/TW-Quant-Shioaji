import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
import yfinance as yf
from multiprocessing import Pool
from functools import partial

# 將專案路徑加入以引用模組
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "TW-Quant-Shioaji"))

from core.integrated_stock_analyzer import IntegratedStockAnalyzer
# 嘗試從 backtester.engine 載入，如果失敗則手動定義 (防止路徑問題)
try:
    from backtester.engine import evaluate_smart_sar_exit, evaluate_exit_confidence
except ImportError:
    # 這裡放一個簡單的回退邏輯，或者從 BullPS-v3 拷貝
    def evaluate_smart_sar_exit(pos, current, current_dt):
        return {'should_exit': current['current_price'] < current['sar']}
    def evaluate_exit_confidence(pos, current):
        return {'exit_confidence': 0.0}

def get_market_thermometer_history():
    """預先抓取大盤歷史以供回測使用"""
    m_df = yf.download("^TWII", period='2y', interval='1d', progress=False)
    if isinstance(m_df.columns, pd.MultiIndex):
        m_df.columns = m_df.columns.get_level_values(0)
    m_df['MA5'] = m_df['Close'].rolling(5).mean()
    return m_df

def process_stock(sym):
    yf_sym = f"{sym}.TW"
    df = yf.download(yf_sym, period='2y', interval='1d', progress=False)
    if df.empty: return sym, None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    analyzer = IntegratedStockAnalyzer()
    df = analyzer.calculate_technical_indicators(df)
    return sym, df

def run_semi_backtest():
    symbols = ["2330", "2454", "3711", "2303", "3661", "2408", "3034", "2449", "2344", "3443"]
    initial_capital = 1000000
    capital = initial_capital
    max_positions = 5
    pos_size_pct = 0.2 # 100萬分5份，每份20萬
    COST = 0.005
    ENTRY_THRESHOLD = 65
    
    print(f"🚀 啟動半導體 Top 10 近一年回測 (高勝率優化版)")
    
    # 1. 準備數據
    with Pool(processes=4) as pool:
        results = pool.map(process_stock, symbols)
    
    all_data = {s: d for s, d in results if d is not None}
    m_df = get_market_thermometer_history()
    
    # 過濾最近一年日期
    one_year_ago = datetime.now() - timedelta(days=365)
    all_dates = sorted([d for d in m_df.index if d >= one_year_ago])
    
    active_positions = []
    trade_log = []
    analyzer = IntegratedStockAnalyzer()
    
    # 預先計算每日市場情緒以加速回測
    print(f"正在預計算每日市場情緒...")
    daily_sentiment_cache = {}
    for date in all_dates:
        m_row = m_df.loc[date]
        m_close = m_row['Close']
        m_ma5 = m_row['MA5']
        
        score = 50
        factors = []
        if m_close > m_ma5:
            score += 15
            factors.append("大盤強勢 (Close > MA5)")
        else:
            score -= 15
            factors.append("大盤弱勢 (Close < MA5)")
            
        daily_sentiment_cache[date] = {
            'sentiment': 'bullish' if score >= 65 else ('bearish' if score <= 35 else 'neutral'),
            'score': score,
            'factors': factors
        }

    for date in all_dates:
        # A. 大盤溫度計判斷
        m_row = m_df.loc[date]
        m_close = m_row['Close']
        m_ma5 = m_row['MA5']
        
        # 強制注入當前日期的情緒，避免 analyzer 重新下載數據
        analyzer.market_sentiment = daily_sentiment_cache[date]
        
        # 動態門檻：大盤破5日線則提高門檻
        dynamic_threshold = ENTRY_THRESHOLD * (1.0 if m_close > m_ma5 else 1.15)
        
        # B. 出場檢查
        for pos in active_positions[:]:
            df = all_data[pos['symbol']]
            if date in df.index:
                curr_row = df.loc[date]
                # 簡單 SAR 出場邏輯
                if curr_row['Close'] < curr_row['SAR']:
                    exit_p = float(curr_row['Close'])
                    pnl = (exit_p - pos['entry_price']) / pos['entry_price'] - COST
                    capital += (pos['entry_price'] * pos['shares']) * (1 + pnl)
                    trade_log.append({'symbol': pos['symbol'], 'pnl': pnl, 'date': date})
                    active_positions.remove(pos)

        # C. 進場掃描
        if len(active_positions) < max_positions:
            candidates = []
            for sym, df in all_data.items():
                if any(p['symbol'] == sym for p in active_positions): continue
                if date in df.index:
                    loc = df.index.get_loc(date)
                    if loc < 60: continue
                    
                    # 評估
                    _, score, _, factors = analyzer.assess_entry_opportunity(df.iloc[:loc+1])
                    if score >= dynamic_threshold:
                        candidates.append({'symbol': sym, 'score': score, 'price': float(df.loc[date, 'Close'])})
            
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            while len(active_positions) < max_positions and candidates:
                c = candidates.pop(0)
                invest = capital * pos_size_pct
                if invest < 10000: break # 剩餘資金太少
                shares = invest / c['price']
                active_positions.append({
                    'symbol': c['symbol'],
                    'entry_price': c['price'],
                    'shares': shares
                })
                capital -= invest

    # 結算
    final_val = capital
    for pos in active_positions:
        df = all_data[pos['symbol']]
        final_val += float(df['Close'].iloc[-1]) * pos['shares']
        
    win_rate = len([t for t in trade_log if t['pnl'] > 0]) / len(trade_log) if trade_log else 0
    total_return = (final_val - initial_capital) / initial_capital
    
    report = {
        "period": "Past 1 Year",
        "stocks": symbols,
        "total_return": f"{total_return*100:.2f}%",
        "win_rate": f"{win_rate*100:.2f}%",
        "trades": len(trade_log),
        "final_equity": round(final_val, 2)
    }
    
    print("\n--- 回測報告 ---")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report

if __name__ == "__main__":
    run_semi_backtest()
