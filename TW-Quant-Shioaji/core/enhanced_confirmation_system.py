#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
強化確認機制系統 (TW Edition)
多層次確認系統，提升進場信號的可靠性
已同步美股 ML 動態權重機制
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
import os
import sys

# TW系統路徑處理 (移除 backend.path_manager 依賴，直接使用相對路徑)
warnings.filterwarnings('ignore')

class EnhancedConfirmationSystem:
    def __init__(self):
        # 初始權重
        self.base_weights = {
            'technical_sync': 0.25,      # 技術指標同步確認
            'price_action': 0.25,        # 價格行為確認
            'volume_confirmation': 0.20,  # 成交量確認
            'momentum_confirmation': 0.15, # 動量確認
            'structure_confirmation': 0.15 # 結構確認
        }
        self.confirmation_weights = self.load_dynamic_weights()
        
        # 載入信號級別的權重 (ML 優化結果)
        self.signal_weights = self.load_signal_weights()

    def load_dynamic_weights(self):
        """[Optimization 2] 載入 ML 產生的動態類別權重"""
        try:
            return self.base_weights
        except Exception:
            return self.base_weights

    def load_signal_weights(self):
        """[Optimization 2] 載入 ML 產生的個別信號權重"""
        try:
            # 嘗試從 core 目錄讀取
            weight_path = os.path.join(os.path.dirname(__file__), 'dynamic_weights.json')
            if os.path.exists(weight_path):
                with open(weight_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # print(f"✅ TW-System 已載入動態信號權重: {len(data)} 個信號")
                    return data
            return {}
        except Exception as e:
            # print(f"⚠️ 無法載入 dynamic_weights.json: {e}")
            return {}

    def get_signal_multiplier(self, signal_name):
        """[Optimization 2] 獲取指定信號的權重加成 (預設 1.0)"""
        return self.signal_weights.get(signal_name, 1.0)
    
    def calculate_technical_sync_confirmation(self, df):
        """
        技術指標同步確認
        檢查多個技術指標是否同步發出多頭信號
        """
        try:
            confirmation_score = 0
            confirmation_factors = []
            max_score = 100
            
            current_idx = len(df) - 1
            if current_idx < 20:
                return {'score': 0, 'factors': ['數據不足'], 'max_score': max_score}
            
            # 1. RSI多頭確認 (25分)
            rsi_current = df['RSI'].iloc[-1]
            rsi_prev = df['RSI'].iloc[-2]
            
            if 30 <= rsi_current <= 60 and rsi_current > rsi_prev:
                score = 25 * self.get_signal_multiplier("RSI健康上升")
                confirmation_score += score
                confirmation_factors.append("RSI健康上升")
            elif rsi_current < 30 and rsi_current > rsi_prev:
                score = 20 * self.get_signal_multiplier("RSI超賣反彈")
                confirmation_score += score
                confirmation_factors.append("RSI超賣反彈")
            elif rsi_current > 70:
                confirmation_score -= 15
                confirmation_factors.append("RSI超買風險")
            
            # 2. MACD多頭確認 (25分)
            macd_current = df['MACD'].iloc[-1]
            macd_signal = df['MACD_Signal'].iloc[-1]
            macd_hist_current = df['MACD_Histogram'].iloc[-1]
            macd_hist_prev = df['MACD_Histogram'].iloc[-2]
            
            if macd_current > macd_signal and macd_hist_current > 0:
                score = 25 * self.get_signal_multiplier("MACD強勢多頭")
                confirmation_score += score
                confirmation_factors.append("MACD強勢多頭")
            elif macd_hist_current > macd_hist_prev and macd_hist_current > 0:
                score = 20 * self.get_signal_multiplier("MACD柱狀圖轉強")
                confirmation_score += score
                confirmation_factors.append("MACD柱狀圖轉強")
            elif macd_current > macd_signal:
                score = 15 * self.get_signal_multiplier("MACD黃金交叉")
                confirmation_score += score
                confirmation_factors.append("MACD黃金交叉")
            
            # 3. 移動平均線確認 (25分)
            ma5 = df['MA5'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]
            ma30 = df['MA30'].iloc[-1] if 'MA30' in df.columns else ma20
            current_price = df['Close'].iloc[-1]
            
            if current_price > ma5 > ma20 > ma30:
                score = 25 * self.get_signal_multiplier("完美多頭排列")
                confirmation_score += score
                confirmation_factors.append("完美多頭排列")
            elif current_price > ma5 > ma20:
                score = 20 * self.get_signal_multiplier("短期多頭排列")
                confirmation_score += score
                confirmation_factors.append("短期多頭排列")
            elif ma5 > ma20:
                score = 15 * self.get_signal_multiplier("均線黃金交叉")
                confirmation_score += score
                confirmation_factors.append("均線黃金交叉")
            elif current_price < ma20:
                confirmation_score -= 10
                confirmation_factors.append("價格低於關鍵均線")
            
            # 4. 布林通道確認 (25分)
            bb_upper = df['BB_Upper'].iloc[-1]
            bb_lower = df['BB_Lower'].iloc[-1]
            bb_middle = df['BB_Middle'].iloc[-1]
            
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            
            if 0.2 <= bb_position <= 0.5:
                score = 25 * self.get_signal_multiplier("布林通道理想位置")
                confirmation_score += score
                confirmation_factors.append("布林通道理想位置")
            elif bb_position <= 0.2:
                score = 20 * self.get_signal_multiplier("布林通道超賣區")
                confirmation_score += score
                confirmation_factors.append("布林通道超賣區")
            elif bb_position >= 0.8:
                confirmation_score -= 15
                confirmation_factors.append("布林通道超買區")
            
            return {
                'score': min(confirmation_score, max_score),
                'factors': confirmation_factors,
                'max_score': max_score
            }
            
        except Exception as e:
            return {'score': 0, 'factors': [f'計算錯誤: {str(e)}'], 'max_score': 100}
    
    def calculate_price_action_confirmation(self, df):
        """
        價格行為確認
        分析K線形態和價格結構
        """
        try:
            confirmation_score = 0
            confirmation_factors = []
            max_score = 100
            
            if len(df) < 10:
                return {'score': 0, 'factors': ['數據不足'], 'max_score': max_score}
            
            # 1. K線形態分析 (40分)
            recent_candles = df.tail(5)
            
            for i in range(len(recent_candles)):
                open_price = recent_candles['Open'].iloc[i]
                close_price = recent_candles['Close'].iloc[i]
                high_price = recent_candles['High'].iloc[i]
                low_price = recent_candles['Low'].iloc[i]
                
                body_size = abs(close_price - open_price)
                total_range = high_price - low_price
                
                if total_range > 0:
                    body_ratio = body_size / total_range
                    upper_shadow = high_price - max(open_price, close_price)
                    lower_shadow = min(open_price, close_price) - low_price
                    
                    # 錘子線/倒錘子線
                    if body_ratio < 0.3 and lower_shadow > body_size * 2:
                        confirmation_score += 15
                        confirmation_factors.append("錘子線形態")
                    
                    # 十字星
                    elif body_ratio < 0.1:
                        confirmation_score += 10
                        confirmation_factors.append("十字星形態")
                    
                    # 長紅K
                    elif close_price > open_price and body_ratio > 0.7:
                        confirmation_score += 8
                        confirmation_factors.append("強勢長紅K")
            
            # 2. 支撐阻力分析 (30分)
            recent_lows = df['Low'].tail(20)
            recent_highs = df['High'].tail(20)
            current_price = df['Close'].iloc[-1]
            
            # 找出關鍵支撐位
            support_levels = []
            for i in range(1, len(recent_lows) - 1):
                if (recent_lows.iloc[i] <= recent_lows.iloc[i-1] and 
                    recent_lows.iloc[i] <= recent_lows.iloc[i+1]):
                    support_levels.append(recent_lows.iloc[i])
            
            if support_levels:
                nearest_support = max([s for s in support_levels if s <= current_price], default=0)
                if nearest_support > 0:
                    support_distance = (current_price - nearest_support) / current_price
                    if support_distance <= 0.02:  # 2%以內
                        confirmation_score += 30
                        confirmation_factors.append("接近關鍵支撐位")
                    elif support_distance <= 0.05:  # 5%以內
                        confirmation_score += 20
                        confirmation_factors.append("靠近支撐位")
            
            # 3. 趨勢結構分析 (30分)
            # 檢查是否形成上升底部
            lows_5d = df['Low'].tail(5)
            if len(lows_5d) >= 3:
                if lows_5d.iloc[-1] > lows_5d.iloc[0]:
                    confirmation_score += 20
                    confirmation_factors.append("上升底部結構")
                elif lows_5d.iloc[-1] > lows_5d.iloc[-2]:
                    confirmation_score += 10
                    confirmation_factors.append("底部抬升")
            
            # 檢查突破形態
            resistance_level = df['High'].tail(10).max()
            if current_price >= resistance_level * 0.98:
                confirmation_score += 10
                confirmation_factors.append("接近突破阻力")
            
            return {
                'score': min(confirmation_score, max_score),
                'factors': confirmation_factors,
                'max_score': max_score
            }
            
        except Exception as e:
            return {'score': 0, 'factors': [f'計算錯誤: {str(e)}'], 'max_score': 100}
    
    def calculate_volume_confirmation(self, df):
        """
        成交量確認
        分析成交量與價格的配合度
        """
        try:
            confirmation_score = 0
            confirmation_factors = []
            max_score = 100
            
            if len(df) < 10:
                return {'score': 0, 'factors': ['數據不足'], 'max_score': max_score}
            
            # 1. 成交量趨勢分析 (40分)
            volume_ratio = df['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df.columns else 1
            volume_ma5 = df['Volume'].tail(5).mean()
            volume_ma20 = df['Volume'].tail(20).mean()
            current_volume = df['Volume'].iloc[-1]
            
            if volume_ratio > 2.0:
                confirmation_score += 40
                confirmation_factors.append("爆量確認")
            elif volume_ratio > 1.5:
                confirmation_score += 30
                confirmation_factors.append("明顯放量")
            elif volume_ratio > 1.2:
                confirmation_score += 20
                confirmation_factors.append("溫和放量")
            elif volume_ratio < 0.8:
                confirmation_score -= 10
                confirmation_factors.append("成交量萎縮")
            
            # 2. 量價配合分析 (35分)
            price_change = df['Close'].pct_change().iloc[-1]
            volume_change = df['Volume'].pct_change().iloc[-1]
            
            if price_change > 0 and volume_change > 0:
                if price_change > 0.02 and volume_change > 0.5:
                    confirmation_score += 35
                    confirmation_factors.append("價漲量增完美配合")
                elif price_change > 0 and volume_change > 0.2:
                    confirmation_score += 25
                    confirmation_factors.append("價漲量增良好配合")
            elif price_change > 0 and volume_change < -0.2:
                confirmation_score -= 15
                confirmation_factors.append("價漲量縮警告")
            
            # 3. 成交量結構分析 (25分)
            if current_volume > volume_ma5 > volume_ma20:
                confirmation_score += 25
                confirmation_factors.append("成交量多頭排列")
            elif current_volume > volume_ma5:
                confirmation_score += 15
                confirmation_factors.append("成交量短期放大")
            
            return {
                'score': min(confirmation_score, max_score),
                'factors': confirmation_factors,
                'max_score': max_score
            }
            
        except Exception as e:
            return {'score': 0, 'factors': [f'計算錯誤: {str(e)}'], 'max_score': 100}
    
    def calculate_comprehensive_confirmation(self, df):
        """
        綜合確認評分
        整合所有確認機制
        """
        try:
            # 獲取各項確認評分
            technical_sync = self.calculate_technical_sync_confirmation(df)
            price_action = self.calculate_price_action_confirmation(df)
            volume_conf = self.calculate_volume_confirmation(df)
            
            # 計算加權總分
            total_score = (
                technical_sync['score'] * self.confirmation_weights['technical_sync'] +
                price_action['score'] * self.confirmation_weights['price_action'] +
                volume_conf['score'] * self.confirmation_weights['volume_confirmation']
            )
            
            # 合併確認因素
            all_factors = []
            all_factors.extend(technical_sync['factors'])
            all_factors.extend(price_action['factors'])
            all_factors.extend(volume_conf['factors'])
            
            # 確定確認等級
            if total_score >= 80:
                confirmation_level = "極強確認"
            elif total_score >= 65:
                confirmation_level = "強確認"
            elif total_score >= 50:
                confirmation_level = "中等確認"
            elif total_score >= 35:
                confirmation_level = "弱確認"
            else:
                confirmation_level = "無確認"
            
            return {
                'total_score': total_score,
                'confirmation_level': confirmation_level,
                'technical_sync': technical_sync,
                'price_action': price_action,
                'volume_confirmation': volume_conf,
                'all_factors': all_factors,
                'detailed_scores': {
                    '技術指標同步': technical_sync['score'],
                    '價格行為': price_action['score'],
                    '成交量確認': volume_conf['score']
                }
            }
            
        except Exception as e:
            return {
                'total_score': 0,
                'confirmation_level': "計算錯誤",
                'all_factors': [f'錯誤: {str(e)}'],
                'detailed_scores': {}
            }
