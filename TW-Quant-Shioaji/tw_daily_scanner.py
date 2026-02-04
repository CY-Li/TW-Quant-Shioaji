#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🇹🇼 TW-Quant Daily Scanner (Soul Edition)
使用 yfinance 獲取最新數據，並套用靈魂復刻版邏輯產生每日信號。
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pathlib import Path

# 將專案路徑加入以引用模組
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "TW-Quant-Shioaji"))

from core.integrated_stock_analyzer import IntegratedStockAnalyzer

# 配置
COMPONENTS_FILE = "TW-Quant-Shioaji/components_150.json"
OUTPUT_REPORT = "TW-Quant-Shioaji/tw_signals_today.json"
ENTRY_SCORE_THRESHOLD = 65

# V9 核心過濾參數
MARKET_RSI_THRESHOLD = 40
VOLUME_CONFIRMATION = 1.3

def get_market_v9_status():
    """獲取 V9 邏輯所需的大盤 RSI 狀態"""
    try:
        m_df = yf.download("^TWII", period='30d', interval='1d', progress=False)
        if m_df.empty: return False, 0
        
        if isinstance(m_df.columns, pd.MultiIndex):
            m_df.columns = m_df.columns.get_level_values(0)
            
        close = m_df['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        is_ok = current_rsi >= MARKET_RSI_THRESHOLD
        return is_ok, current_rsi
    except:
        return False, 0

def scan_market():
    print(f"🚀 啟動台股每日掃描 (V9 勝率優化版) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 獲取大盤 V9 狀態
    market_ok, market_rsi = get_market_v9_status()
    print(f"🌡️  大盤 RSI: {market_rsi:.1f} | V9 准許進場: {'YES' if market_ok else 'NO (過濾器啟動)'}")

    if not market_ok:
        print("⚠️  由於大盤 RSI 低於門檻 (40)，V9 策略今日將採取極度保守態度。")

    if not os.path.exists(COMPONENTS_FILE):
        print(f"❌ 找不到成分股文件: {COMPONENTS_FILE}")
        return

    with open(COMPONENTS_FILE, "r") as f:
        symbols = json.load(f)
    
    analyzer = IntegratedStockAnalyzer()
    signals = []
    all_scores = []
    
    filtered_by_market = 0
    filtered_by_volume = 0
    
    print(f"正在掃描 {len(symbols)} 檔標的...")
    
    for i, sym in enumerate(symbols):
        try:
            yf_sym = f"{sym}.TW"
            df = yf.download(yf_sym, period='1y', interval='1d', progress=False)
            
            if df.empty or len(df) < 60:
                continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 計算 V9 所需的成交量均線
            df['Volume_MA20'] = df['Volume'].rolling(20).mean()
            
            # 計算技術指標
            df = analyzer.calculate_technical_indicators(df)
            
            # 評估基礎分數
            entry_advice, score, level, factors = analyzer.assess_entry_opportunity(df)
            
            current_price = float(df['Close'].iloc[-1])
            current_vol = float(df['Volume'].iloc[-1])
            avg_vol = float(df['Volume_MA20'].iloc[-1])
            
            # V9 過濾邏輯
            vol_ok = current_vol >= avg_vol * VOLUME_CONFIRMATION
            
            # 記錄所有分數 (即使沒過 V9 過濾器也記錄，供分析參考)
            all_scores.append({
                "symbol": sym,
                "score": round(score, 2),
                "price": current_price,
                "vol_ratio": round(current_vol / avg_vol, 2) if avg_vol > 0 else 0,
                "v9_pass": market_ok and vol_ok and score >= ENTRY_SCORE_THRESHOLD
            })

            # 只有完全符合 V9 邏輯的才列入信號
            if score >= ENTRY_SCORE_THRESHOLD:
                if not market_ok:
                    filtered_by_market += 1
                    continue
                if not vol_ok:
                    filtered_by_volume += 1
                    continue
                    
                print(f"✅ [V9 SIGNAL] {sym} | 分數: {score:.1f} | 價格: {current_price} | 量比: {current_vol/avg_vol:.2f}")
                signals.append({
                    "symbol": sym,
                    "name": yf_sym,
                    "score": round(score, 2),
                    "price": current_price,
                    "advice": entry_advice,
                    "factors": factors,
                    "vol_ratio": round(current_vol / avg_vol, 2),
                    "timestamp": str(datetime.now())
                })
            
        except Exception as e:
            pass
            
        if (i + 1) % 20 == 0:
            print(f"進度: {i + 1}/{len(symbols)}")

    # 排序
    all_scores = sorted(all_scores, key=lambda x: x['score'], reverse=True)
    
    print("\n📊 --- V9 掃描統計 ---")
    print(f"符合基礎門檻 (>={ENTRY_SCORE_THRESHOLD}): {len(signals) + filtered_by_market + filtered_by_volume}")
    print(f"遭大盤過濾: {filtered_by_market}")
    print(f"遭成交量過濾: {filtered_by_volume}")
    print(f"最終 V9 信號數: {len(signals)}")
    print("----------------------\n")

    signals = sorted(signals, key=lambda x: x['score'], reverse=True)
    
    report = {
        "date": datetime.now().strftime('%Y-%m-%d'),
        "v9_mode": True,
        "count": len(signals),
        "signals": signals,
        "market_rsi": round(market_rsi, 2)
    }
    
    with open(OUTPUT_REPORT, "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report

if __name__ == "__main__":
    scan_market()
