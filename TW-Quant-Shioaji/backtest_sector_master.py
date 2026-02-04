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
from core.portfolio_logic import evaluate_smart_sar_exit, evaluate_exit_confidence

# 配置
DATA_DIR = "TW-Quant-Shioaji/data/batch_sector"
INDEX_FILE = "TW-Quant-Shioaji/data/batch_semi/twii.csv" # 延用之前的大盤數據
REPORT_DIR = "TW-Quant-Shioaji/sector_reports"
CAPITAL = 1000000
MAX_POSITIONS = 5
COST = 0.005

# --- V7 VCP 優化參數 ---
VCP_VOL_CONTRACTION_RATIO = 0.85
VCP_TIGHTNESS_THRESHOLD = 0.06
PROFIT_LOCK_PCT = 0.08
TRAILING_STOP_THRESHOLD = 0.20
TRAILING_STOP_MULT = 1.2

# 板塊定義
SECTORS = {
    "電腦及週邊設備": ["2317", "2382", "3231", "2357", "2376"],
    "電子零組件": ["2327", "3037", "2368", "8046", "3044"],
    "航運業": ["2603", "2609", "2615", "2610", "2618"],
    "通信、網路與光電": ["2412", "3045", "2345", "2409", "3481"],
    "電機、生技與其他電子": ["1519", "1513", "1504", "1795", "2360"],
    "傳統產業與其他": ["2002", "1101", "1216", "1476", "9910"]
}

def pre_analyze_stock_sector(sym, data_dir, bt_start_date):
    try:
        p = f"{data_dir}/{sym}.csv"
        if not os.path.exists(p): return sym, None
        
        df = pd.read_csv(p)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 1. 波動率指標
        df['Std20'] = df['Close'].rolling(20).std()
        df['Std60'] = df['Close'].rolling(60).std()
        df['Vol_Contraction'] = df['Std20'] / df['Std60']
        
        # 2. 價格緊湊度
        df['Range10'] = (df['High'].rolling(10).max() - df['Low'].rolling(10).min()) / df['Close']
        
        # 3. ATR 與基礎指標
        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Close'].shift())
        low_cp = np.abs(df['Low'] - df['Close'].shift())
        df['TR'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        analyzer = IntegratedStockAnalyzer()
        analyzer.mtf_analyzer.calculate_multi_timeframe_score = lambda s: {'final_score': 50, 'trend_consistency': 0.5, 'recommendation': []}
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
            
            vcp_bonus = 0
            cur = upto_date.iloc[-1]
            if cur['Vol_Contraction'] < VCP_VOL_CONTRACTION_RATIO: vcp_bonus += 10
            if cur['Range10'] < VCP_TIGHTNESS_THRESHOLD: vcp_bonus += 10
            
            prev_high = upto_date['High'].iloc[-11:-1].max()
            if cur['Close'] > prev_high and cur['Volume_Ratio'] > 1.3: vcp_bonus += 15

            score += vcp_bonus

            results[date] = {
                'score': score,
                'factors': factors,
                'close': float(cur['Close']),
                'sar': float(cur['SAR']),
                'rsi': float(cur['RSI']),
                'macd': float(cur['MACD']),
                'atr': float(cur['ATR']),
                'is_vcp': vcp_bonus >= 20
            }
        return sym, results
    except Exception as e:
        print(f"Error pre-analyzing {sym}: {e}")
        return sym, None

def run_sector_backtest(sector_name, symbols, mdf):
    bt_start_date = datetime.now() - timedelta(days=2*365)
    print(f"\n📁 --- 啟動板塊回測: {sector_name} ---")
    
    # 預計算
    all_analysis = {}
    all_dates = set()
    with multiprocessing.Pool() as pool:
        worker = partial(pre_analyze_stock_sector, data_dir=DATA_DIR, bt_start_date=bt_start_date)
        results = pool.map(worker, symbols)
        
    for sym, res in results:
        if res:
            all_analysis[sym] = res
            all_dates.update(res.keys())

    if not all_dates:
        print(f"  ❌ {sector_name} 無法獲取數據")
        return None

    sorted_dates = sorted(list(all_dates))
    capital = CAPITAL
    active_positions = []
    trade_log = []

    for date in sorted_dates:
        # A. 出場檢查
        for pos in active_positions[:]:
            sym = pos['symbol']
            if date in all_analysis[sym]:
                data = all_analysis[sym][date]
                current_p = data['close']
                current_atr = data['atr']
                pos['max_price'] = max(pos.get('max_price', 0), current_p)
                unrealized_pnl = (current_p - pos['entry_price']) / pos['entry_price']
                
                should_exit = False
                exit_reason = ""

                if unrealized_pnl >= TRAILING_STOP_THRESHOLD: pos['trailing_active'] = True
                if pos.get('trailing_active', False):
                    if current_p < pos['max_price'] - (current_atr * TRAILING_STOP_MULT):
                        should_exit = True
                        exit_reason = "V7移動止盈"
                
                if not should_exit and (pos['max_price'] / pos['entry_price'] - 1) >= PROFIT_LOCK_PCT:
                    if unrealized_pnl <= 0.01:
                        should_exit = True
                        exit_reason = "保本觸發"

                if not should_exit:
                    current_analysis = {
                        'symbol': sym, 'current_price': current_p, 'sar': data['sar'],
                        'rsi': data['rsi'], 'macd': data['macd'], 'confidence_factors': data['factors']
                    }
                    if current_p < data['sar']:
                        should_exit = True
                        exit_reason = "SAR反轉"
                    elif (pos['max_price'] / current_p - 1) > 0.1: # 簡單理由侵蝕模擬
                        should_exit = True
                        exit_reason = "回檔過深"

                if should_exit:
                    pnl = (current_p - pos['entry_price']) / pos['entry_price'] - COST
                    capital += (pos['entry_price'] * pos['shares']) * (1 + pnl)
                    trade_log.append({
                        'symbol': sym, 'pnl': round(pnl * 100, 2), 'reason': exit_reason, 'exit_date': str(date.date())
                    })
                    active_positions.remove(pos)

        # B. 進場掃描
        if len(active_positions) < MAX_POSITIONS:
            threshold = 65
            if date in mdf.index:
                if mdf.loc[date, 'Close'] <= mdf.loc[date, 'MA5']: threshold = 78
            
            candidates = []
            for sym, analysis in all_analysis.items():
                if any(p['symbol'] == sym for p in active_positions): continue
                if date in analysis:
                    d = analysis[date]
                    if d['score'] >= threshold:
                        priority = d['score'] + (20 if d['is_vcp'] else 0)
                        candidates.append({'symbol': sym, 'priority': priority, 'price': d['close'], 'atr': d['atr'], 'factors': d['factors']})
            
            candidates = sorted(candidates, key=lambda x: x['priority'], reverse=True)
            while len(active_positions) < MAX_POSITIONS and candidates:
                c = candidates.pop(0)
                total_equity = capital + sum([p['entry_price'] * p['shares'] for p in active_positions])
                risk_amount = total_equity * 0.02
                shares = risk_amount / (c['atr'] * 2)
                
                max_invest = total_equity * 0.25
                if (shares * c['price']) > max_invest: shares = max_invest / c['price']
                if (shares * c['price']) > capital: shares = capital / c['price']

                active_positions.append({
                    'symbol': c['symbol'], 'entry_price': c['price'], 'entry_date': date,
                    'shares': shares, 'entry_atr': c['atr'], 'max_price': c['price']
                })
                capital -= (shares * c['price'])

    final_val = capital + sum([all_analysis[p['symbol']][sorted_dates[-1]]['close'] * p['shares'] for p in active_positions if sorted_dates[-1] in all_analysis[p['symbol']]])
    total_ret = (final_val - CAPITAL) / CAPITAL * 100
    wr = len([t for t in trade_log if t['pnl'] > 0]) / len(trade_log) if trade_log else 0
    
    report = {
        "sector": sector_name,
        "total_return": f"{round(total_ret, 2)}%",
        "win_rate": f"{round(wr * 100, 2)}%",
        "total_trades": len(trade_log),
        "final_equity": round(final_val, 2)
    }
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(f"{REPORT_DIR}/{sector_name}.json", "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ 完成！回報率: {report['total_return']} | 交易數: {report['total_trades']}")
    return report

def main():
    mdf = pd.read_csv(INDEX_FILE)
    mdf['Date'] = pd.to_datetime(mdf['Date'])
    mdf.set_index('Date', inplace=True)
    mdf['MA5'] = mdf['Close'].rolling(5).mean()

    results = []
    for sector, symbols in SECTORS.items():
        res = run_sector_backtest(sector, symbols, mdf)
        if res: results.append(res)
    
    print("\n🏆 --- 各板塊績效總結 ---")
    print(f"{'板塊':<20} | {'回報率':<10} | {'勝率':<8} | {'交易數':<6}")
    print("-" * 55)
    for r in results:
        print(f"{r['sector']:<18} | {r['total_return']:<10} | {r['win_rate']:<8} | {r['total_trades']:<6}")

if __name__ == "__main__":
    main()
