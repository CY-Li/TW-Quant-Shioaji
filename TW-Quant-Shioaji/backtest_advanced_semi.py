import os
import sys
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# 將專案路徑加入以引用模組
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "TW-Quant-Shioaji"))

from core.integrated_stock_analyzer import IntegratedStockAnalyzer

# 配置
SEMI_LIST_FILE = "TW-Quant-Shioaji/semi_10.json"
BACKTEST_REPORT = "TW-Quant-Shioaji/semi_backtest_report_2yr.json"
ENTRY_THRESHOLD = 65
PROFIT_TARGET = 0.15 # 15%
STOP_LOSS = 0.07     # 7%
TRANS_COST = 0.005   # 0.5% (手續費 + 稅)

def run_backtest_for_symbol(analyzer, symbol):
    yf_sym = f"{symbol}.TW"
    print(f"🔍 正在回測 {yf_sym}...")
    
    # 下載 2 年 + 1 年 (為了技術指標計算) 的數據
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365) # 抓 3 年數據，確保有足夠歷史計算指標
    
    df = yf.download(yf_sym, start=start_date, end=end_date, interval='1d', progress=False)
    if df.empty or len(df) < 250:
        print(f"  ⚠️ {symbol} 數據不足")
        return None
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 預先計算所有日期的指標 (一次性計算以提升效率)
    df = analyzer.calculate_technical_indicators(df)
    
    # 執行回測：只看最近兩年的交易
    bt_start_date = end_date - timedelta(days=2*365)
    df_test = df[df.index >= bt_start_date]
    
    if df_test.empty:
        return None

    trades = []
    in_position = False
    entry_data = None
    
    # 遍歷每一天
    for i in range(len(df_test)):
        current_date = df_test.index[i]
        row = df_test.iloc[i]
        
        # 取得直到當天的數據切片供 Analyzer 評估 (模擬真實每日掃描)
        # 其實 calculate_technical_indicators 已經算好了，我們只需要檢查當天的值
        
        if not in_position:
            # 這裡我們模擬 Analyzer 的 assess_entry_opportunity 邏輯
            # 由於 Analyzer 內部的 assess 主要是針對最新一根，我們可以直接讀取 pre-calculated 的列
            # 或是手動執行簡化版邏輯 (避免每回圈重複計算)
            
            # 使用 Analyzer 的得分邏輯 (從 df 中提取)
            # 注意：這裡假設 Analyzer 已經在 calculate_technical_indicators 裡把分數算進去
            # 或是我們手動調用 assess_entry_opportunity 傳入截至目前的 df
            
            history_upto_now = df[df.index <= current_date]
            if len(history_upto_now) < 60: continue
            
            entry_advice, score, level, factors = analyzer.assess_entry_opportunity(history_upto_now)
            
            if score >= ENTRY_THRESHOLD:
                in_position = True
                entry_data = {
                    'entry_date': str(current_date.date()),
                    'entry_price': float(row['Close']),
                    'score_at_entry': score,
                    'factors': factors
                }
        else:
            # 持倉中，檢查出場條件
            pnl_pct = (float(row['Close']) - entry_data['entry_price']) / entry_data['entry_price']
            
            # 1. 停利 2. 停損 3. 或是理由侵蝕 (這裡簡化，先用固定停利停損)
            if pnl_pct >= PROFIT_TARGET or pnl_pct <= -STOP_LOSS:
                net_pnl = pnl_pct - TRANS_COST
                trades.append({
                    **entry_data,
                    'exit_date': str(current_date.date()),
                    'exit_price': float(row['Close']),
                    'net_pnl': round(net_pnl * 100, 2),
                    'duration': (current_date - pd.to_datetime(entry_data['entry_date'])).days
                })
                in_position = False
                entry_data = None
                
    return {
        "symbol": symbol,
        "trades": trades,
        "total_pnl": round(sum([t['net_pnl'] for t in trades]), 2) if trades else 0,
        "win_rate": round(len([t for t in trades if t['net_pnl'] > 0]) / len(trades) * 100, 2) if trades else 0,
        "trade_count": len(trades)
    }

def main():
    if not os.path.exists(SEMI_LIST_FILE):
        print(f"❌ 找不到清單: {SEMI_LIST_FILE}")
        return

    with open(SEMI_LIST_FILE, "r") as f:
        semi_list = json.load(f)

    analyzer = IntegratedStockAnalyzer()
    results = []
    
    print(f"🚀 開始半導體 10 檔回測 (2年)... 進場門檻: {ENTRY_THRESHOLD}")
    
    for symbol in semi_list:
        res = run_backtest_for_symbol(analyzer, symbol)
        if res:
            results.append(res)
            print(f"  ✅ {symbol} 完成. 累計盈虧: {res['total_pnl']}% | 交易次數: {res['trade_count']}")

    # 總結
    final_report = {
        "timestamp": str(datetime.now()),
        "config": {
            "period": "2y",
            "entry_threshold": ENTRY_THRESHOLD,
            "profit_target": PROFIT_TARGET,
            "stop_loss": STOP_LOSS
        },
        "results": results,
        "overall_pnl": round(np.mean([r['total_pnl'] for r in results]), 2) if results else 0
    }

    with open(BACKTEST_REPORT, "w", encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"\n🏁 回測完畢！總體平均盈虧: {final_report['overall_pnl']}%")
    print(f"報告已存至 {BACKTEST_REPORT}")

if __name__ == "__main__":
    main()
