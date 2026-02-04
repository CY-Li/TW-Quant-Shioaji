import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
import yfinance as yf

# 將專案路徑加入以引用模組
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "TW-Quant-Shioaji"))

print("Importing IntegratedStockAnalyzer...")
from core.integrated_stock_analyzer import IntegratedStockAnalyzer
print("Imported IntegratedStockAnalyzer.")

def evaluate_smart_sar_exit(pos, current, current_dt):
    return {'should_exit': current['current_price'] < current['sar']}
def evaluate_exit_confidence(pos, current):
    return {'exit_confidence': 0.0}

def get_market_thermometer_history():
    print("下載大盤數據...")
    m_df = yf.download("^TWII", period='2y', interval='1d', progress=False)
    if isinstance(m_df.columns, pd.MultiIndex):
        m_df.columns = m_df.columns.get_level_values(0)
    m_df['MA5'] = m_df['Close'].rolling(5).mean()
    print("大盤數據下載完成")
    return m_df

def process_stock(sym):
    print(f"處理股票: {sym}")
    yf_sym = f"{sym}.TW"
    try:
        df = yf.download(yf_sym, period='2y', interval='1d', progress=False)
        if df.empty: 
            print(f"{sym}: 無數據")
            return sym, None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        analyzer = IntegratedStockAnalyzer()
        df = analyzer.calculate_technical_indicators(df)
        print(f"{sym}: 完成技術指標計算")
        return sym, df
    except Exception as e:
        print(f"{sym}: 錯誤 {e}")
        return sym, None

def run_semi_backtest():
    symbols = ["2330", "2454", "3711", "2303", "3661", "2408", "3034", "2449", "2344", "3443"]
    initial_capital = 1000000
    capital = initial_capital
    max_positions = 5
    pos_size_pct = 0.2 
    COST = 0.005
    # Lower threshold
    ENTRY_THRESHOLD = 55
    
    print(f"🚀 啟動半導體 Top 10 近一年回測 (優化版 v2: Threshold={ENTRY_THRESHOLD}, TP=10%, SL=5%)")
    
    results = []
    for sym in symbols:
        results.append(process_stock(sym))
    
    all_data = {s: d for s, d in results if d is not None}
    if not all_data:
        print("沒有成功下載任何股票數據")
        return

    m_df = get_market_thermometer_history()
    
    one_year_ago = datetime.now() - timedelta(days=365)
    all_dates = sorted([d for d in m_df.index if d >= one_year_ago])
    
    active_positions = []
    trade_log = []
    analyzer = IntegratedStockAnalyzer()
    
    # Mock MTF Analyzer with debug prints
    def mock_mtf_score(symbol, period='2y'):
        df = getattr(analyzer, 'current_df_context', None)
        if df is None or df.empty:
            return {'final_score': 0, 'trend_consistency': 0.5, 'overall_rating': 'No Data', 'recommendation': []}
        
        current_price = df['Close'].iloc[-1]
        ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns else current_price
        ma60 = df['MA60'].iloc[-1] if 'MA60' in df.columns else current_price
        
        score = 0
        if current_price > ma20 and ma20 > ma60:
            score = 80 
        elif current_price > ma60:
            score = 60 
        elif current_price < ma20 and ma20 < ma60:
            score = -80 
        elif current_price < ma60:
            score = -60 
            
        return {
            'final_score': score, 
            'trend_consistency': 0.8 if abs(score) > 50 else 0.4, 
            'overall_rating': 'Simulated Bullish' if score > 0 else 'Simulated Bearish',
            'recommendation': ['Trend Follow']
        }
    
    analyzer.mtf_analyzer.calculate_multi_timeframe_score = mock_mtf_score

    # Optimization: Patch detect_bullish_signals to only scan recent days
    original_detect = IntegratedStockAnalyzer.detect_bullish_signals
    def fast_detect_bullish_signals(self, df):
        if df is None or df.empty: return []
        start_idx = max(20, len(df) - 10)
        
        signals = []
        for i in range(start_idx, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            if current['RSI'] > 75: continue
            
            bullish_conditions = []
            if (current['MA5'] > current['MA20'] and prev['MA5'] <= prev['MA20'] and current['Volume_Ratio'] > 1.5):
                bullish_conditions.append("黃金交叉+放量")
            if (current['MACD_Histogram'] > 0 and prev['MACD_Histogram'] <= 0 and current['RSI'] < 70):
                bullish_conditions.append("MACD柱狀圖轉正+RSI未超買")
            if (current['K'] > current['D'] and prev['K'] <= prev['D'] and current['K'] < 20 and current['D'] < 25):
                bullish_conditions.append("KD低檔交叉")
            
            if len(bullish_conditions) > 0:
                 signals.append({
                    'date': df.index[i],
                    'price': current['Close'],
                    'conditions': bullish_conditions,
                    'days_ago': len(df) - 1 - i
                })
        return signals

    IntegratedStockAnalyzer.detect_bullish_signals = fast_detect_bullish_signals
    
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

    print("開始回測迴圈...")
    for date in all_dates:
        m_row = m_df.loc[date]
        m_close = m_row['Close']
        m_ma5 = m_row['MA5']
        
        analyzer.market_sentiment = daily_sentiment_cache[date]
        dynamic_threshold = ENTRY_THRESHOLD * (1.0 if m_close > m_ma5 else 1.15)
        
        # B. 出場檢查
        for pos in active_positions[:]:
            df = all_data[pos['symbol']]
            if date in df.index:
                curr_row = df.loc[date]
                current_price = float(curr_row['Close'])
                entry_price = pos['entry_price']
                
                # Calculate unrealized PnL
                pnl_pct = (current_price - entry_price) / entry_price
                
                should_exit = False
                exit_reason = ""
                
                # 1. Take Profit (+10%)
                if pnl_pct >= 0.10:
                    should_exit = True
                    exit_reason = "Take Profit"
                # 2. Stop Loss (-5%)
                elif pnl_pct <= -0.05:
                    should_exit = True
                    exit_reason = "Stop Loss"
                # 3. SAR Exit
                elif current_price < curr_row['SAR']:
                    should_exit = True
                    exit_reason = "SAR Exit"
                
                if should_exit:
                    pnl = pnl_pct - COST
                    capital += (entry_price * pos['shares']) * (1 + pnl)
                    trade_log.append({
                        'symbol': pos['symbol'], 
                        'pnl': pnl, 
                        'date': date.strftime('%Y-%m-%d'),
                        'reason': exit_reason
                    })
                    active_positions.remove(pos)

        # C. 進場掃描
        if len(active_positions) < max_positions:
            candidates = []
            for sym, df in all_data.items():
                if any(p['symbol'] == sym for p in active_positions): continue
                if date in df.index:
                    loc = df.index.get_loc(date)
                    if loc < 60: continue
                    
                    try:
                        # Inject context for MTF mock
                        analyzer.current_symbol = sym
                        analyzer.current_df_context = df.iloc[:loc+1]
                        
                        _, score, _, factors = analyzer.assess_entry_opportunity(df.iloc[:loc+1])
                        if score >= dynamic_threshold:
                            candidates.append({'symbol': sym, 'score': score, 'price': float(df.loc[date, 'Close'])})
                    except Exception as e:
                        pass
            
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            while len(active_positions) < max_positions and candidates:
                c = candidates.pop(0)
                invest = capital * pos_size_pct
                if invest < 10000: break 
                shares = invest / c['price']
                active_positions.append({
                    'symbol': c['symbol'],
                    'entry_price': c['price'],
                    'shares': shares
                })
                capital -= invest

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

if __name__ == "__main__":
    run_semi_backtest()
