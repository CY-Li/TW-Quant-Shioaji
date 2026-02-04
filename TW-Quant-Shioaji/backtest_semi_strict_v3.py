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
DATA_DIR = "TW-Quant-Shioaji/data/batch_semi"
INDEX_FILE = f"{DATA_DIR}/twii.csv"
SEMI_LIST_FILE = "TW-Quant-Shioaji/semi_10.json"
REPORT_FILE = "TW-Quant-Shioaji/semi_backtest_strict_v3.json"

ENTRY_THRESHOLD_BASE = 65
CAPITAL = 1000000
MAX_POSITIONS = 5
POS_SIZE_PCT = 0.2
COST = 0.005

def pre_analyze_stock(sym, data_dir, bt_start_date):
    """預先計算個股每日得分與因素，大幅提速"""
    try:
        p = f"{data_dir}/{sym}.csv"
        if not os.path.exists(p): return sym, None
        
        df = pd.read_csv(p)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        analyzer = IntegratedStockAnalyzer()
        # 禁用重型指標
        analyzer.mtf_analyzer.calculate_multi_timeframe_score = lambda s: {'final_score': 50, 'trend_consistency': 0.5, 'recommendation': []}
        analyzer.market_sentiment = {'sentiment': 'neutral', 'score': 50}
        
        df = analyzer.calculate_technical_indicators(df)
        
        results = {}
        # 只針對回測區間進行打分
        test_df = df[df.index >= bt_start_date - timedelta(days=60)]
        for i in range(len(test_df)):
            date = test_df.index[i]
            if date < bt_start_date: continue
            
            # 取得切片
            upto_date = test_df.iloc[:i+1]
            if len(upto_date) < 60: continue
            
            _, score, _, factors = analyzer.assess_entry_opportunity(upto_date)
            results[date] = {
                'score': score,
                'factors': factors,
                'close': float(upto_date['Close'].iloc[-1]),
                'sar': float(upto_date['SAR'].iloc[-1]),
                'rsi': float(upto_date['RSI'].iloc[-1]),
                'macd': float(upto_date['MACD'].iloc[-1])
            }
        return sym, results
    except Exception as e:
        print(f"Error analyzing {sym}: {e}")
        return sym, None

def run_backtest():
    with open(SEMI_LIST_FILE, "r") as f:
        symbols = json.load(f)
    
    mdf = pd.read_csv(INDEX_FILE)
    mdf['Date'] = pd.to_datetime(mdf['Date'])
    mdf.set_index('Date', inplace=True)
    mdf['MA5'] = mdf['Close'].rolling(5).mean()

    bt_start_date = datetime.now() - timedelta(days=2*365)
    
    print(f"🔄 正在預計算 10 檔指標與得分 (多進程)...")
    all_analysis = {}
    all_dates = set()
    
    with multiprocessing.Pool() as pool:
        worker = partial(pre_analyze_stock, data_dir=DATA_DIR, bt_start_date=bt_start_date)
        results = pool.map(worker, symbols)
        
    for sym, res in results:
        if res:
            all_analysis[sym] = res
            all_dates.update(res.keys())

    sorted_dates = sorted(list(all_dates))
    print(f"🚀 啟動台股嚴格回測，交易天數: {len(sorted_dates)}")
    
    capital = CAPITAL
    active_positions = []
    trade_log = []

    for date in sorted_dates:
        # --- A. 出場檢查 ---
        for pos in active_positions[:]:
            sym = pos['symbol']
            if date in all_analysis[sym]:
                data = all_analysis[sym][date]
                
                current_analysis = {
                    'symbol': sym,
                    'current_price': data['close'],
                    'sar': data['sar'],
                    'rsi': data['rsi'],
                    'macd': data['macd'],
                    'confidence_factors': data['factors']
                }
                
                sar_res = evaluate_smart_sar_exit(pos, current_analysis, current_dt=date)
                conf_res = evaluate_exit_confidence(pos, current_analysis)
                
                if sar_res['should_exit'] or conf_res['exit_confidence'] >= 0.8:
                    exit_p = data['close']
                    pnl = (exit_p - pos['entry_price']) / pos['entry_price'] - COST
                    capital += (pos['entry_price'] * pos['shares']) * (1 + pnl)
                    
                    reason = "SAR觸發" if sar_res['should_exit'] else f"理由侵蝕 ({conf_res['exit_confidence']})"
                    trade_log.append({
                        'symbol': sym, 
                        'pnl': round(pnl * 100, 2),
                        'reason': reason,
                        'exit_date': str(date.date())
                    })
                    active_positions.remove(pos)

        # --- B. 進場掃描 ---
        if len(active_positions) < MAX_POSITIONS:
            threshold = ENTRY_THRESHOLD_BASE
            if date in mdf.index:
                m_row = mdf.loc[date]
                if m_row['Close'] <= m_row['MA5']:
                    threshold = ENTRY_THRESHOLD_BASE * 1.15
            
            candidates = []
            for sym, analysis in all_analysis.items():
                if any(p['symbol'] == sym for p in active_positions): continue
                if date in analysis:
                    d = analysis[date]
                    if d['score'] >= threshold:
                        candidates.append({'symbol': sym, 'score': d['score'], 'price': d['close'], 'factors': d['factors']})
            
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

    final_val = capital + sum([all_analysis[p['symbol']][sorted_dates[-1]]['close'] * p['shares'] for p in active_positions if sorted_dates[-1] in all_analysis[p['symbol']]])
    wr = len([t for t in trade_log if t['pnl'] > 0]) / len(trade_log) if trade_log else 0
    
    report = {
        "strategy": "BullPS-TW-Strict-v3-Optimized",
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
    print(json.dumps(res, indent=2, ensure_ascii=False))
