#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bull Put Spread (BPS) 策略優化器
專門針對選擇權交易設計，計算最佳點位與風險評估
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

class BPSOptimizer:
    def __init__(self):
        self.tz = pytz.timezone('Asia/Taipei')

    def get_expected_move(self, symbol, days_to_expiry=30):
        """
        計算預期漲跌幅 (Expected Move)
        公式: Price * IV * sqrt(T/365)
        """
        try:
            ticker = yf.Ticker(symbol)
            # 獲取當前價格
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            
            # 獲取 IV (這在 yfinance 中比較難直接取得，我們用 30 天歷史波動率代替，或從期權鏈計算)
            hist = ticker.history(period='1y')
            returns = hist['Close'].pct_change().dropna()
            annual_vol = returns.std() * np.sqrt(252)
            
            # 預期漲跌幅 (1 標準差)
            expected_move = current_price * annual_vol * np.sqrt(days_to_expiry / 365)
            
            return {
                'current_price': current_price,
                'annual_vol': annual_vol,
                'expected_move': expected_move,
                'lower_1sd': current_price - expected_move,
                'upper_1sd': current_price + expected_move
            }
        except Exception as e:
            print(f"Error calculating expected move for {symbol}: {e}")
            return None

    def suggest_bps_strikes(self, symbol, days_to_expiry=30):
        """
        建議 BPS 的進場點位 (Short Put & Long Put)
        """
        em_data = self.get_expected_move(symbol, days_to_expiry)
        if not em_data:
            return None
        
        current_price = em_data['current_price']
        lower_1sd = em_data['lower_1sd']
        
        # 建議 Short Put 點位在 1SD 之外或近期強力支撐位
        # 這裡可以加入更多邏輯，例如找整數關卡
        short_put_strike = floor_to_half(lower_1sd)
        long_put_strike = short_put_strike - 5 # 假設 5 點價差
        
        # 計算建議到期日 (約 30-45 天後最近的週五)
        expiry_date = (datetime.now() + timedelta(days=days_to_expiry)).strftime('%Y-%m-%d')

        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'short_put_strike': short_put_strike,
            'long_put_strike': long_put_strike,
            'suggested_expiry': expiry_date, # 新增建議到期日
            'safety_margin_pct': round((current_price - short_put_strike) / current_price * 100, 2),
            'expected_move_1sd': round(em_data['expected_move'], 2)
        }

    def check_earnings_risk(self, symbol):
        """
        檢查近期是否有財報風險 (如微軟事件)
        """
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            if calendar is not None and not calendar.empty:
                # 獲取財報日期
                earnings_date = calendar.iloc[0, 0]
                if isinstance(earnings_date, datetime):
                    days_to_earnings = (earnings_date.replace(tzinfo=None) - datetime.now()).days
                    return {
                        'earnings_date': earnings_date.strftime('%Y-%m-%d'),
                        'days_to_earnings': days_to_earnings,
                        'high_risk': days_to_earnings < 14 # 兩週內有財報視為高風險
                    }
            return {'high_risk': False, 'note': '無財報資訊'}
        except:
            return {'high_risk': False, 'note': '無法獲取財報資訊'}

def floor_to_half(val):
    return np.floor(val * 2) / 2

if __name__ == "__main__":
    optimizer = BPSOptimizer()
    # 測試
    print(optimizer.suggest_bps_strikes("MSFT"))
    print(optimizer.check_earnings_risk("MSFT"))
