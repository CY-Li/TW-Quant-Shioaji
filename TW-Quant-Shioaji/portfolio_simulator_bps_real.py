import pandas as pd
import numpy as np
import json
import os

def run_real_bps_portfolio_backtest():
    with open("TW-Quant-Shioaji/components_150.json", "r") as f:
        symbols = json.load(f)
    
    data_dir = "TW-Quant-Shioaji/data/batch_150"
    initial_capital = 1000000
    capital = initial_capital
    max_positions = 5
    
    # BPS 策略參數
    WIDTH = 5.0 # 假設 5 點價差 (台股點數)
    CREDIT_PROXY = 1.0 # 假設平均每口收 1.0 點 (20% 寬度)
    TP_LEVEL = 0.5 # 50% 獲利停利 (Spread 降至 0.5)
    
    active_positions = [] # [{'symbol', 'entry_p', 'short_strike', 'entry_credit'}]
    trade_log = []
    
    all_data = {}
    all_dates = set()
    for sym in symbols:
        p = f"{data_dir}/{sym}.csv"
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df['MA20'] = df['Close'].rolling(20).mean()
            # 簡化版 1SD 預期跌幅 (日波幅 * sqrt(30))
            df['Volatility'] = df['Close'].pct_change().rolling(20).std()
            df['Expected_Move'] = df['Close'] * df['Volatility'] * np.sqrt(21)
            all_data[sym] = df
            all_dates.update(df.index.tolist())

    sorted_dates = sorted(list(all_dates))
    
    for date in sorted_dates:
        # 1. BPS 出場檢查 (關鍵修正)
        for pos in active_positions[:]:
            df = all_data[pos['symbol']]
            if date in df.index:
                curr_price = df.loc[date, 'Close']
                
                # 條件 A: 股價跌破履約價 (Assignment Risk) -> 強制出場
                if curr_price <= pos['short_strike']:
                    loss = (pos['short_strike'] - curr_price) + pos['entry_credit'] # 簡化虧損計算
                    loss = min(loss, WIDTH) - pos['entry_credit'] # 最大損失限制在寬度
                    capital += (WIDTH - loss) * 1000 # 假設一口
                    trade_log.append({'symbol': pos['symbol'], 'result': 'LOSS', 'pnl': -loss})
                    active_positions.remove(pos)
                    continue
                
                # 條件 B: 獲利達標 (50% 權利金收回)
                # 這裡用股價反彈幅度來模擬權利金收回 (簡單線性模型)
                price_move = (curr_price - pos['entry_price']) / pos['expected_move']
                if price_move >= 0.5: # 股價往上彈了預期跌幅的一半，通常權利金已收回 > 50%
                    pnl = pos['entry_credit'] * 0.5
                    capital += (WIDTH + pnl) * 1000
                    trade_log.append({'symbol': pos['symbol'], 'result': 'WIN', 'pnl': pnl})
                    active_positions.remove(pos)

        # 2. BPS 進場檢查
        if len(active_positions) < max_positions:
            for sym in symbols:
                if any(p['symbol'] == sym for p in active_positions): continue
                df = all_data.get(sym)
                if df is not None and date in df.index:
                    row = df.loc[date]
                    # 進場條件：原本的靈魂邏輯 (這部分沒錯)
                    if row['Close'] > row['MA20']: 
                        # 找到一個安全點位 (1SD 之外)
                        short_strike = row['Close'] - row['Expected_Move']
                        active_positions.append({
                            'symbol': sym,
                            'entry_price': row['Close'],
                            'short_strike': short_strike,
                            'entry_credit': CREDIT_PROXY,
                            'expected_move': row['Expected_Move'],
                            'invested': WIDTH * 1000
                        })
                        capital -= WIDTH * 1000
                        if len(active_positions) >= max_positions: break

    # 結算
    wr = len([t for t in trade_log if t['pnl'] > 0]) / len(trade_log) if trade_log else 0
    final_val = capital + len(active_positions) * WIDTH * 1000
    
    report = {
        "strategy": "True-BPS-Logic-Backtest",
        "win_rate": f"{round(wr*100, 2)}%",
        "total_trades": len(trade_log),
        "final_equity": round(final_val, 2),
        "total_return": f"{round((final_val - initial_capital)/initial_capital*100, 2)}%"
    }
    
    with open("TW-Quant-Shioaji/portfolio_report_true_bps.json", "w") as f:
        json.dump(report, f, indent=2)
    return report

if __name__ == "__main__":
    run_real_bps_portfolio_backtest()
