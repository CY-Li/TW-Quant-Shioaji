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
REPORT_FILE = "TW-Quant-Shioaji/semi_backtest_final_v4.json"

ENTRY_THRESHOLD_BASE = 65
CAPITAL = 1000000
MAX_POSITIONS = 5
POS_SIZE_PCT = 0.2
COST = 0.005

# --- 優化參數 ---
COOL_OFF_DAYS = 10      # 停損後冷卻期
PROFIT_LOCK_PCT = 0.08  # 獲利 8% 啟動保本
VOLUME_PENALTY = True   # 強化成交量權重

def pre_analyze_stock(sym, data_dir, bt_start_date):
    try:
        p = f"{data_dir}/{sym}.csv"
        if not os.path.exists(p): return sym, None
        
        df = pd.read_csv(p)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        analyzer = IntegratedStockAnalyzer()
        analyzer.mtf_analyzer.calculate_multi_timeframe_score = lambda s: {'final_score': 50, 'trend_consistency': 0.5, 'recommendation': []}
        analyzer.market_sentiment = {'sentiment': 'neutral', 'score': 50}
        
        df = analyzer.calculate_technical_indicators(df)
        
        results = {}
        test_df = df[df.index >= bt_start_date - timedelta(days=60)]
        for i in range(len(test_df)):
            date = test_df.index[i]
            if date < bt_start_date: continue
            
            upto_date = test_df.iloc[:i+1]
            if len(upto_date) < 60: continue
            
            # [Optimization 3] 強化成交量權重
            vol_ratio = float(upto_date['Volume_Ratio'].iloc[-1])
            _, score, _, factors = analyzer.assess_entry_opportunity(upto_date)
            
            if vol_ratio < 1.0:
                score -= 10 # 無量上漲扣分
                factors.append("成交量不足扣分")
            elif vol_ratio > 1.5:
                score += 5  # 放量加成
                factors.append("放量加成")

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
        return sym, None

def run_backtest():
    with open(SEMI_LIST_FILE, "r") as f:
        symbols = json.load(f)
    
    mdf = pd.read_csv(INDEX_FILE)
    mdf['Date'] = pd.to_datetime(mdf['Date'])
    mdf.set_index('Date', inplace=True)
    mdf['MA5'] = mdf['Close'].rolling(5).mean()

    bt_start_date = datetime.now() - timedelta(days=2*365)
    
    print(f"🔄 正在預計算指標與 V4 優化邏輯...")
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
    print(f"🚀 啟動台股 V4 終極優化回測，天數: {len(sorted_dates)}")
    
    capital = CAPITAL
    active_positions = []
    trade_log = []
    cool_off_tracker = {s: None for s in symbols} # 追蹤冷卻期

    for date in sorted_dates:
        # --- A. 出場檢查 ---
        for pos in active_positions[:]:
            sym = pos['symbol']
            if date in all_analysis[sym]:
                data = all_analysis[sym][date]
                current_p = data['close']
                
                # [Optimization 2] 獲利保本機制
                max_p_since_entry = max(pos.get('max_price', 0), current_p)
                pos['max_price'] = max_p_since_entry
                unrealized_pnl = (current_p - pos['entry_price']) / pos['entry_price']
                
                should_exit = False
                exit_reason = ""

                # 1. 檢查 8% 保本
                if (max_p_since_entry / pos['entry_price'] - 1) >= PROFIT_LOCK_PCT:
                    if unrealized_pnl <= 0.01: # 跌回 +1% 時強制出場保本
                        should_exit = True
                        exit_reason = "獲利回吐保本觸發"

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
                    elif conf_res['exit_confidence'] >= 0.8:
                        should_exit = True
                        exit_reason = f"理由侵蝕 ({conf_res['exit_confidence']})"

                if should_exit:
                    pnl = (current_p - pos['entry_price']) / pos['entry_price'] - COST
                    capital += (pos['entry_price'] * pos['shares']) * (1 + pnl)
                    trade_log.append({
                        'symbol': sym, 'pnl': round(pnl * 100, 2), 'reason': exit_reason, 'exit_date': str(date.date())
                    })
                    active_positions.remove(pos)
                    # [Optimization 1] 設置冷卻期
                    cool_off_tracker[sym] = date + timedelta(days=COOL_OFF_DAYS)

        # --- B. 進場掃描 ---
        if len(active_positions) < MAX_POSITIONS:
            # 大盤門檻
            threshold = ENTRY_THRESHOLD_BASE
            if date in mdf.index:
                if mdf.loc[date, 'Close'] <= mdf.loc[date, 'MA5']: threshold = 75
            
            candidates = []
            for sym, analysis in all_analysis.items():
                if any(p['symbol'] == sym for p in active_positions): continue
                # [Optimization 1] 冷卻期檢查
                if cool_off_tracker[sym] and date < cool_off_tracker[sym]:
                    # 除非分數超高 (>85)，否則冷卻中
                    if analysis[date]['score'] < 85: continue
                
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
                    'symbol': c['symbol'], 'entry_price': c['price'], 'entry_date': date,
                    'shares': shares, 'max_price': c['price'], 'initial_analysis_snapshot': {'confidence_factors': c['factors']}
                })
                capital -= invest

    final_val = capital + sum([all_analysis[p['symbol']][sorted_dates[-1]]['close'] * p['shares'] for p in active_positions if sorted_dates[-1] in all_analysis[p['symbol']]])
    wr = len([t for t in trade_log if t['pnl'] > 0]) / len(trade_log) if trade_log else 0
    
    report = {
        "strategy": "BullPS-TW-V4-Ultimate",
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
