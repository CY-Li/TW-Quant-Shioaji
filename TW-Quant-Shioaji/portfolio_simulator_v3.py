import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def run_portfolio_backtest():
    with open("TW-Quant-Shioaji/components.json", "r") as f:
        symbols = json.load(f)
    
    data_dir = "TW-Quant-Shioaji/data/batch"
    initial_capital = 1000000
    capital = initial_capital
    max_positions = 10
    pos_size = capital / max_positions
    
    # 策略參數
    MA_PERIOD = 20
    RSI_BUY = 55
    TP = 0.15
    SL = 0.05
    COST = 0.005
    
    active_positions = [] # [{'symbol', 'entry_p', 'entry_date'}]
    equity_curve = []
    trade_log = []
    
    # 讀取並對齊所有數據的日期
    all_data = {}
    all_dates = set()
    for sym in symbols:
        p = f"{data_dir}/{sym}.csv"
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            # 計算指標
            df['MA20'] = df['Close'].rolling(MA_PERIOD).mean()
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            all_data[sym] = df
            all_dates.update(df.index.tolist())
    
    sorted_dates = sorted(list(all_dates))
    
    print(f"開始執行組合模擬，總天數: {len(sorted_dates)}")
    
    for date in sorted_dates:
        # 1. 檢查出場
        for pos in active_positions[:]:
            df = all_data[pos['symbol']]
            if date in df.index:
                curr_p = df.loc[date, 'Close']
                pnl = (curr_p - pos['entry_p']) / pos['entry_p']
                
                if pnl >= TP or pnl <= -SL:
                    net_gain = (pnl - COST) * pos_size
                    capital += (pos_size + net_gain)
                    trade_log.append({'symbol': pos['symbol'], 'pnl': pnl - COST, 'exit_date': date})
                    active_positions.remove(pos)
        
        # 2. 檢查進場
        if len(active_positions) < max_positions:
            candidates = []
            for sym in symbols:
                if any(p['symbol'] == sym for p in active_positions): continue
                df = all_data.get(sym)
                if df is not None and date in df.index:
                    row = df.loc[date]
                    if row['Close'] > row['MA20'] and row['RSI'] < RSI_BUY:
                        # 簡單排名：RSI 越低優先
                        candidates.append({'symbol': sym, 'rsi': row['RSI'], 'price': row['Close']})
            
            # 按 RSI 排序
            candidates = sorted(candidates, key=lambda x: x['rsi'])
            while len(active_positions) < max_positions and candidates:
                c = candidates.pop(0)
                active_positions.append({'symbol': c['symbol'], 'entry_p': c['price'], 'entry_date': date})
                capital -= pos_size
        
        # 3. 記錄每日淨值
        current_equity = capital
        for pos in active_positions:
            df = all_data[pos['symbol']]
            if date in df.index:
                current_equity += (df.loc[date, 'Close'] / pos['entry_p']) * pos_size
            else:
                current_equity += pos_size
        equity_curve.append({'date': str(date), 'equity': current_equity})

    # 結算
    final_equity = equity_curve[-1]['equity']
    total_return = (final_equity - initial_capital) / initial_capital
    
    report = {
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "total_return": f"{round(total_return * 100, 2)}%",
        "win_rate": f"{round(len([t for t in trade_log if t['pnl'] > 0]) / len(trade_log) * 100, 2) if trade_log else 0}%",
        "total_trades": len(trade_log)
    }
    
    with open("TW-Quant-Shioaji/portfolio_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("組合回測完成！")
    return report

if __name__ == "__main__":
    run_portfolio_backtest()
