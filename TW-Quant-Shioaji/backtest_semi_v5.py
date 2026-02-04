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
REPORT_FILE = "TW-Quant-Shioaji/semi_backtest_v5_benchmark_killer.json"

CAPITAL = 1000000
MAX_POSITIONS = 5
COST = 0.005

# --- V5 優化參數 ---
ATR_PERIOD = 14
TARGET_RISK_PCT = 0.02 # 每筆交易風險控制在總資產的 2% (ATR-based sizing)
ADX_TREND_THRESHOLD = 20
RS_LOOKBACK = 60 # 60天相對強度

def pre_analyze_stock_v5(sym, data_dir, bt_start_date):
    try:
        p = f"{data_dir}/{sym}.csv"
        if not os.path.exists(p): return sym, None
        
        df = pd.read_csv(p)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 計算 ATR (用於倉位控制與動態停損)
        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Close'].shift())
        low_cp = np.abs(df['Low'] - df['Close'].shift())
        df['TR'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['ATR'] = df['TR'].rolling(window=ATR_PERIOD).mean()
        
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
            
            # V5: 趨勢加成 (ADX)
            adx = float(upto_date['ADX'].iloc[-1])
            if adx > ADX_TREND_THRESHOLD:
                score += 5
                factors.append(f"強趨勢確認 (ADX:{adx:.1f})")

            results[date] = {
                'score': score,
                'factors': factors,
                'close': float(upto_date['Close'].iloc[-1]),
                'sar': float(upto_date['SAR'].iloc[-1]),
                'rsi': float(upto_date['RSI'].iloc[-1]),
                'macd': float(upto_date['MACD'].iloc[-1]),
                'atr': float(upto_date['ATR'].iloc[-1])
            }
        return sym, results
    except Exception as e:
        return sym, None

def run_backtest_v5():
    with open(SEMI_LIST_FILE, "r") as f:
        symbols = json.load(f)
    
    mdf = pd.read_csv(INDEX_FILE)
    mdf['Date'] = pd.to_datetime(mdf['Date'])
    mdf.set_index('Date', inplace=True)
    mdf['MA5'] = mdf['Close'].rolling(5).mean()

    bt_start_date = datetime.now() - timedelta(days=2*365)
    
    print(f"🔄 正在預計算 V5 邏輯 (ATR Sizing + Trend Filter)...")
    all_analysis = {}
    all_dates = set()
    
    with multiprocessing.Pool() as pool:
        worker = partial(pre_analyze_stock_v5, data_dir=DATA_DIR, bt_start_date=bt_start_date)
        results = pool.map(worker, symbols)
        
    for sym, res in results:
        if res:
            all_analysis[sym] = res
            all_dates.update(res.keys())

    sorted_dates = sorted(list(all_dates))
    print(f"🚀 啟動 V5 挑戰大盤回測，天數: {len(sorted_dates)}")
    
    capital = CAPITAL
    active_positions = []
    trade_log = []

    for date in sorted_dates:
        # --- A. 出場檢查 ---
        for pos in active_positions[:]:
            sym = pos['symbol']
            if date in all_analysis[sym]:
                data = all_analysis[sym][date]
                current_p = data['close']
                
                # V5 動態停損：使用 3x ATR 或 SAR 孰低
                atr_stop = pos['entry_price'] - (pos['entry_atr'] * 2.5)
                
                should_exit = False
                exit_reason = ""

                # 1. 獲利保本 (10% 以上回檔)
                unrealized_pnl = (current_p - pos['entry_price']) / pos['entry_price']
                if unrealized_pnl > 0.15: # 獲利豐厚時放寬
                    if current_p < atr_stop:
                        should_exit = True
                        exit_reason = "ATR動態停損"
                
                if not should_exit:
                    current_analysis = {
                        'symbol': sym, 'current_price': current_p, 'sar': data['sar'],
                        'rsi': data['rsi'], 'macd': data['macd'], 'confidence_factors': data['factors']
                    }
                    sar_res = evaluate_smart_sar_exit(pos, current_analysis, current_dt=date)
                    conf_res = evaluate_exit_confidence(pos, current_analysis)
                    
                    if sar_res['should_exit']:
                        should_exit = True
                        exit_reason = "SAR反轉"
                    elif conf_res['exit_confidence'] >= 0.85: # 提高耐受度，讓子彈飛
                        should_exit = True
                        exit_reason = f"理由侵蝕 ({conf_res['exit_confidence']})"

                if should_exit:
                    pnl = (current_p - pos['entry_price']) / pos['entry_price'] - COST
                    capital += (pos['entry_price'] * pos['shares']) * (1 + pnl)
                    trade_log.append({
                        'symbol': sym, 'pnl': round(pnl * 100, 2), 'reason': exit_reason, 'exit_date': str(date.date())
                    })
                    active_positions.remove(pos)

        # --- B. 進場掃描 ---
        if len(active_positions) < MAX_POSITIONS:
            # 大盤門檻 (V5 改為更積極，大盤好時 60，不好時 75)
            threshold = 60
            if date in mdf.index:
                if mdf.loc[date, 'Close'] <= mdf.loc[date, 'MA5']: threshold = 75
            
            candidates = []
            for sym, analysis in all_analysis.items():
                if any(p['symbol'] == sym for p in active_positions): continue
                if date in analysis:
                    d = analysis[date]
                    if d['score'] >= threshold:
                        candidates.append({'symbol': sym, 'score': d['score'], 'price': d['close'], 'factors': d['factors'], 'atr': d['atr']})
            
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            while len(active_positions) < MAX_POSITIONS and candidates:
                c = candidates.pop(0)
                
                # V5: ATR-Based Position Sizing
                # 公式: Position Size = (Total Capital * Risk%) / ATR
                # 假設 ATR 是停損距離，我們希望這筆交易如果虧損一個 ATR，損失不超過總資產的 1%
                risk_amount = (capital + sum([p['entry_price'] * p['shares'] for p in active_positions])) * 0.02
                stop_dist = c['atr'] * 2
                shares = risk_amount / stop_dist
                
                # 限制單一持倉最大權重不超過 25%
                max_invest = (capital + sum([p['entry_price'] * p['shares'] for p in active_positions])) * 0.25
                if (shares * c['price']) > max_invest:
                    shares = max_invest / c['price']
                
                if (shares * c['price']) > capital:
                    shares = capital / c['price']

                active_positions.append({
                    'symbol': c['symbol'], 'entry_price': c['price'], 'entry_date': date,
                    'shares': shares, 'entry_atr': c['atr'], 'initial_analysis_snapshot': {'confidence_factors': c['factors']}
                })
                capital -= (shares * c['price'])

    final_val = capital + sum([all_analysis[p['symbol']][sorted_dates[-1]]['close'] * p['shares'] for p in active_positions if sorted_dates[-1] in all_analysis[p['symbol']]])
    wr = len([t for t in trade_log if t['pnl'] > 0]) / len(trade_log) if trade_log else 0
    
    report = {
        "strategy": "BullPS-TW-V5-Benchmark-Killer",
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
    res = run_backtest_v5()
    print(json.dumps(res, indent=2, ensure_ascii=False))
