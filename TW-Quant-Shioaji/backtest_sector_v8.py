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
INDEX_FILE = "TW-Quant-Shioaji/data/batch_semi/twii.csv" # 共用大盤數據
SECTOR_LIST_FILE = "TW-Quant-Shioaji/sector_lists.json"
REPORT_DIR = "TW-Quant-Shioaji/sector_reports_v8"

if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

CAPITAL = 1000000
MAX_POSITIONS = 5
COST = 0.005

# --- V8 Alpha 優化參數 (保持一致) ---
VCP_VOL_CONTRACTION_RATIO = 0.85 
VCP_TIGHTNESS_THRESHOLD = 0.07
ADX_THRESHOLD = 18
PROFIT_LOCK_PCT = 0.08
TRAILING_STOP_THRESHOLD = 0.20
TRAILING_STOP_MULT = 2.5 

def pre_analyze_stock_v8(sym, data_dir, bt_start_date):
    try:
        p = f"{data_dir}/{sym}.csv"
        if not os.path.exists(p): return sym, None
        
        df = pd.read_csv(p)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 1. VCP 波動率指標
        df['Std20'] = df['Close'].rolling(20).std()
        df['Std60'] = df['Close'].rolling(60).std()
        df['Vol_Contraction'] = df['Std20'] / df['Std60']
        
        # 2. 價格緊湊度
        df['Range10'] = (df['High'].rolling(10).max() - df['Low'].rolling(10).min()) / df['Close']
        
        # 3. ATR 與 ADX
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
        # 預留緩衝期
        test_df = df[df.index >= bt_start_date - timedelta(days=90)]
        for i in range(len(test_df)):
            date = test_df.index[i]
            if date < bt_start_date: continue
            
            upto_date = test_df.iloc[:i+1]
            if len(upto_date) < 60: continue
            
            _, score, _, factors = analyzer.assess_entry_opportunity(upto_date)
            
            # --- V8 混合邏輯 ---
            vcp_bonus = 0
            cur = upto_date.iloc[-1]
            adx = float(cur['ADX'])
            
            if cur['Vol_Contraction'] < VCP_VOL_CONTRACTION_RATIO:
                vcp_bonus += 10
                factors.append(f"VCP波動收斂")
            
            if cur['Range10'] < VCP_TIGHTNESS_THRESHOLD:
                vcp_bonus += 10
                factors.append(f"VCP價格緊湊")

            if adx > ADX_THRESHOLD:
                vcp_bonus += 5
                factors.append(f"ADX趨勢確認({adx:.1f})")
            
            score += vcp_bonus

            results[date] = {
                'score': score,
                'factors': factors,
                'close': float(cur['Close']),
                'sar': float(cur['SAR']),
                'rsi': float(cur['RSI']),
                'macd': float(cur['MACD']),
                'atr': float(cur['ATR']),
                'adx': adx,
                'is_vcp': vcp_bonus >= 20
            }
        return sym, results
    except Exception as e:
        return sym, None

def run_sector_backtest(sector_name, symbols):
    print(f"\n🚀 啟動板塊回測: {sector_name} (Stocks: {symbols})")
    
    mdf = pd.read_csv(INDEX_FILE)
    mdf['Date'] = pd.to_datetime(mdf['Date'])
    mdf.set_index('Date', inplace=True)
    mdf['MA5'] = mdf['Close'].rolling(5).mean()

    # 設定回測時間：2年
    bt_start_date = datetime.now() - timedelta(days=730)
    
    all_analysis = {}
    all_dates = set()
    
    # 使用 Pool 進行並行預計算
    with multiprocessing.Pool() as pool:
        worker = partial(pre_analyze_stock_v8, data_dir=DATA_DIR, bt_start_date=bt_start_date)
        results = pool.map(worker, symbols)
        
    for sym, res in results:
        if res:
            all_analysis[sym] = res
            all_dates.update(res.keys())

    sorted_dates = sorted(list(all_dates))
    print(f"   -> 交易天數: {len(sorted_dates)}")
    
    capital = CAPITAL
    active_positions = []
    trade_log = []

    for date in sorted_dates:
        # A. 出場
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
                    trailing_stop = pos['max_price'] - (current_atr * TRAILING_STOP_MULT)
                    if current_p < trailing_stop:
                        should_exit = True
                        exit_reason = f"V8移動止盈"
                
                if not should_exit and (pos['max_price'] / pos['entry_price'] - 1) >= PROFIT_LOCK_PCT:
                    if unrealized_pnl <= 0.01:
                        should_exit = True
                        exit_reason = "保本觸發"

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
                    elif conf_res['exit_confidence'] >= 0.88:
                        should_exit = True
                        exit_reason = "信心侵蝕"

                if should_exit:
                    pnl = (current_p - pos['entry_price']) / pos['entry_price'] - COST
                    capital += (pos['entry_price'] * pos['shares']) * (1 + pnl)
                    trade_log.append({
                        'symbol': sym, 'pnl': round(pnl * 100, 2), 'reason': exit_reason, 
                        'entry_date': str(pos['entry_date'].date()),
                        'exit_date': str(date.date())
                    })
                    active_positions.remove(pos)

        # B. 進場
        if len(active_positions) < MAX_POSITIONS:
            threshold = 60
            if date in mdf.index:
                if mdf.loc[date, 'Close'] <= mdf.loc[date, 'MA5']: threshold = 75
            
            candidates = []
            for sym, analysis in all_analysis.items():
                if any(p['symbol'] == sym for p in active_positions): continue
                if date in analysis:
                    d = analysis[date]
                    if d['score'] >= threshold:
                        priority = d['score'] + (15 if d['is_vcp'] else 0) + (10 if d['adx'] > 25 else 0)
                        candidates.append({
                            'symbol': sym, 'score': d['score'], 'priority': priority, 
                            'price': d['close'], 'factors': d['factors'], 'atr': d['atr']
                        })
            
            candidates = sorted(candidates, key=lambda x: x['priority'], reverse=True)
            while len(active_positions) < MAX_POSITIONS and candidates:
                c = candidates.pop(0)
                total_equity = capital + sum([p['entry_price'] * p['shares'] for p in active_positions])
                risk_amount = total_equity * 0.035 
                shares = risk_amount / (c['atr'] * 2.5)
                
                max_invest = total_equity * 0.30
                if (shares * c['price']) > max_invest: shares = max_invest / c['price']
                if (shares * c['price']) > capital: shares = capital / c['price']

                active_positions.append({
                    'symbol': c['symbol'], 'entry_price': c['price'], 'entry_date': date,
                    'shares': shares, 'entry_atr': c['atr'], 'max_price': c['price'],
                    'initial_analysis_snapshot': {'confidence_factors': c['factors']}
                })
                capital -= (shares * c['price'])

    final_val = capital + sum([all_analysis[p['symbol']][sorted_dates[-1]]['close'] * p['shares'] for p in active_positions if sorted_dates[-1] in all_analysis[p['symbol']]])
    wr = len([t for t in trade_log if t['pnl'] > 0]) / len(trade_log) if trade_log else 0
    
    report = {
        "sector": sector_name,
        "strategy": "BullPS-TW-V8-Market-Alpha",
        "final_equity": round(final_val, 2),
        "total_return": f"{round((final_val - CAPITAL)/CAPITAL * 100, 2)}%",
        "win_rate": f"{round(wr * 100, 2)}%",
        "total_trades": len(trade_log),
        "trades": trade_log
    }
    
    out_file = f"{REPORT_DIR}/{sector_name}_report.json"
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ 完成: {out_file} | Return: {report['total_return']}")
    return report

def run_all_sectors():
    with open(SECTOR_LIST_FILE, "r") as f:
        sectors = json.load(f)
    
    summary = {}
    for sector, syms in sectors.items():
        res = run_sector_backtest(sector, syms)
        summary[sector] = res['total_return']
        
    print("\n🏁 所有板塊回測完成！")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    run_all_sectors()
