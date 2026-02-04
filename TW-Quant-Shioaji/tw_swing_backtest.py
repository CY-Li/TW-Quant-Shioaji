import shioaji as sj
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta

# 配置
API_KEY = "BVhtFhFdjJirnkxbey2PYMUxTBz3QNFnagjiViMCMV11"
SECRET_KEY = "fjPPAFN9FuYXzsgATtfbaoY52Y3nrx3tHpVMSak4U6g"

def run_swing_backtest(symbol):
    api = sj.Shioaji()
    api.login(api_key=API_KEY, secret_key=SECRET_KEY)
    
    try:
        # 1. 獲取數據
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
        
        # 抓取個股與大盤數據 (0050 用來當大盤過濾器)
        stock_contract = api.Contracts.Stocks[symbol]
        market_contract = api.Contracts.Stocks["0050"] 
        
        kbars = api.kbars(
            contract=stock_contract, 
            start=start_date, 
            end=end_date,
            unit=sj.constant.KBarTime.Day
        )
        df = pd.DataFrame({**kbars})
        df.ts = pd.to_datetime(df.ts)
        
        mkbars = api.kbars(
            contract=market_contract, 
            start=start_date, 
            end=end_date,
            unit=sj.constant.KBarTime.Day
        )
        mdf = pd.DataFrame({**mkbars})
        mdf.ts = pd.to_datetime(mdf.ts)
        
        # 指標計算
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
        mdf['MA20'] = mdf['Close'].rolling(20).mean()
        
        # 策略參數
        PROFIT_TARGET = 0.12 # 12% 停利
        STOP_LOSS = 0.06      # 6% 停損
        TRANS_COST = 0.005    # 0.5% 交易成本 (含稅與手續費)
        
        trades = []
        in_pos = False
        entry_p = 0
        
        for i in range(200, len(df)):
            # 找到大盤對應日期的狀態
            m_row = mdf[mdf['ts'] <= df['ts'].iloc[i]].tail(1)
            if m_row.empty: continue
            
            if not in_pos:
                # 買入條件：個股站在 200MA 上 + 大盤站在 20MA 上 (趨勢濾網)
                if df['Close'].iloc[i] > df['MA200'].iloc[i] and m_row['Close'].iloc[0] > m_row['MA20'].iloc[0]:
                    # 配合 RSI 反轉
                    # 這裡為了展示趨勢濾網效果，我們簡化進場
                    in_pos = True
                    entry_p = df['Close'].iloc[i]
                    trades.append({'entry_date': str(df['ts'].iloc[i]), 'entry_price': entry_p})
            else:
                pnl = (df['Close'].iloc[i] - entry_p) / entry_p
                if pnl >= PROFIT_TARGET or pnl <= -STOP_LOSS:
                    in_pos = False
                    net_pnl = pnl - TRANS_COST
                    trades[-1].update({
                        'exit_date': str(df['ts'].iloc[i]),
                        'net_pnl': round(net_pnl * 100, 2)
                    })
        
        completed = [t for t in trades if 'net_pnl' in t]
        if not completed: return None
        
        win_rate = len([t for t in completed if t['net_pnl'] > 0]) / len(completed)
        return {
            "symbol": symbol,
            "net_total_pnl": f"{round(np.sum([t['net_pnl'] for t in completed]), 2)}%",
            "win_rate": f"{round(win_rate * 100, 2)}%",
            "trades_count": len(completed),
            "avg_net_pnl": f"{round(np.mean([t['net_pnl'] for t in completed]), 2)}%"
        }
        
    finally:
        api.logout()

if __name__ == "__main__":
    # 測試鴻海與台積電
    results = []
    for s in ["2317", "2330"]:
        res = run_swing_backtest(s)
        if res: results.append(res)
    print(json.dumps(results, indent=2, ensure_ascii=False))
