#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多時間框架分析系統
整合日線、週線、月線的技術分析
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class MultiTimeframeAnalyzer:
    def __init__(self):
        self.analyzer = None  # 延遲初始化，避免循環導入
        self.timeframes = {
            'daily': '1d',
            'weekly': '1wk', 
            'monthly': '1mo'
        }
        
        # 時間框架權重
        self.timeframe_weights = {
            'monthly': 0.4,   # 月線權重最高，決定主趨勢
            'weekly': 0.35,   # 週線權重次之，決定中期趨勢
            'daily': 0.25     # 日線權重最低，決定短期進場時機
        }
    
    def get_multi_timeframe_data(self, symbol, period='2y'):
        """
        獲取多時間框架數據，包含錯誤處理
        """
        try:
            # 檢查股票代號有效性
            if not symbol or symbol in ['UNKNOWN', '$UNKNOWN']:
                return None

            ticker = yf.Ticker(symbol)

            # 獲取日線數據
            daily_data = ticker.history(period=period, interval='1d')

            # 獲取週線數據
            weekly_data = ticker.history(period=period, interval='1wk')

            # 獲取月線數據
            monthly_data = ticker.history(period=period, interval='1mo')

            # 檢查數據完整性
            if daily_data.empty:
                print(f"  ⚠️  {symbol}: 日線數據為空")
                return None

            if weekly_data.empty:
                print(f"  ⚠️  {symbol}: 週線數據為空，使用日線數據替代")
                weekly_data = daily_data.resample('W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()

            if monthly_data.empty:
                print(f"  ⚠️  {symbol}: 月線數據為空，使用日線數據替代")
                monthly_data = daily_data.resample('M').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()

            # 確保有足夠的數據
            if len(daily_data) < 30:
                print(f"  ⚠️  {symbol}: 日線數據不足 ({len(daily_data)}天)")
                return None

            return {
                'daily': daily_data,
                'weekly': weekly_data,
                'monthly': monthly_data
            }

        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "delisted" in error_msg.lower():
                print(f"  ❌ {symbol}: 股票可能已下市")
            elif "No data found" in error_msg:
                print(f"  ❌ {symbol}: 無數據")
            else:
                print(f"  ❌ {symbol}: 多時間框架數據獲取失敗 - {error_msg}")
            return None
    
    def analyze_timeframe_trend(self, df, timeframe):
        """
        分析單一時間框架的趨勢
        """
        try:
            # 延遲導入避免循環依賴
            if self.analyzer is None:
                from core.integrated_stock_analyzer import IntegratedStockAnalyzer
                self.analyzer = IntegratedStockAnalyzer()

            # 計算技術指標
            df_with_indicators = self.analyzer.calculate_technical_indicators(df)
            if df_with_indicators is None:
                return None
            
            trend_score = 0
            trend_factors = []
            
            current_price = df_with_indicators['Close'].iloc[-1]
            ma5 = df_with_indicators['MA5'].iloc[-1]
            ma20 = df_with_indicators['MA20'].iloc[-1]
            ma30 = df_with_indicators['MA30'].iloc[-1] if 'MA30' in df_with_indicators.columns else ma20
            
            # 1. 移動平均線趨勢分析
            if current_price > ma5 > ma20 > ma30:
                trend_score += 40
                trend_factors.append(f"{timeframe}完美多頭排列")
            elif current_price > ma5 > ma20:
                trend_score += 30
                trend_factors.append(f"{timeframe}短期多頭排列")
            elif ma5 > ma20:
                trend_score += 20
                trend_factors.append(f"{timeframe}均線黃金交叉")
            elif current_price < ma5 < ma20 < ma30:
                trend_score -= 40
                trend_factors.append(f"{timeframe}完美空頭排列")
            elif current_price < ma5 < ma20:
                trend_score -= 30
                trend_factors.append(f"{timeframe}短期空頭排列")
            
            # 2. 價格動量分析
            price_momentum_5 = (current_price - df_with_indicators['Close'].iloc[-6]) / df_with_indicators['Close'].iloc[-6] * 100 if len(df_with_indicators) > 5 else 0
            price_momentum_20 = (current_price - df_with_indicators['Close'].iloc[-21]) / df_with_indicators['Close'].iloc[-21] * 100 if len(df_with_indicators) > 20 else 0
            
            if price_momentum_5 > 5:
                trend_score += 20
                trend_factors.append(f"{timeframe}短期強勢上漲")
            elif price_momentum_5 > 2:
                trend_score += 10
                trend_factors.append(f"{timeframe}短期溫和上漲")
            elif price_momentum_5 < -5:
                trend_score -= 20
                trend_factors.append(f"{timeframe}短期急跌")
            
            if price_momentum_20 > 10:
                trend_score += 20
                trend_factors.append(f"{timeframe}中期強勢上漲")
            elif price_momentum_20 > 5:
                trend_score += 10
                trend_factors.append(f"{timeframe}中期溫和上漲")
            elif price_momentum_20 < -10:
                trend_score -= 20
                trend_factors.append(f"{timeframe}中期下跌趨勢")
            
            # 3. RSI趨勢分析
            rsi = df_with_indicators['RSI'].iloc[-1]
            if 40 <= rsi <= 70:
                trend_score += 10
                trend_factors.append(f"{timeframe}RSI健康區間")
            elif rsi < 30:
                trend_score += 5  # 超賣可能反彈
                trend_factors.append(f"{timeframe}RSI超賣")
            elif rsi > 80:
                trend_score -= 15
                trend_factors.append(f"{timeframe}RSI嚴重超買")
            
            # 4. MACD趨勢分析
            macd = df_with_indicators['MACD'].iloc[-1]
            macd_signal = df_with_indicators['MACD_Signal'].iloc[-1]
            macd_hist = df_with_indicators['MACD_Histogram'].iloc[-1]
            
            if macd > macd_signal and macd_hist > 0:
                trend_score += 15
                trend_factors.append(f"{timeframe}MACD多頭")
            elif macd > macd_signal:
                trend_score += 10
                trend_factors.append(f"{timeframe}MACD黃金交叉")
            elif macd < macd_signal and macd_hist < 0:
                trend_score -= 15
                trend_factors.append(f"{timeframe}MACD空頭")
            
            # 確定趨勢方向
            if trend_score >= 60:
                trend_direction = "強多頭"
            elif trend_score >= 30:
                trend_direction = "多頭"
            elif trend_score >= 10:
                trend_direction = "弱多頭"
            elif trend_score >= -10:
                trend_direction = "中性"
            elif trend_score >= -30:
                trend_direction = "弱空頭"
            elif trend_score >= -60:
                trend_direction = "空頭"
            else:
                trend_direction = "強空頭"
            
            return {
                'timeframe': timeframe,
                'trend_score': trend_score,
                'trend_direction': trend_direction,
                'trend_factors': trend_factors,
                'current_price': current_price,
                'ma5': ma5,
                'ma20': ma20,
                'rsi': rsi,
                'macd': macd,
                'price_momentum_5': price_momentum_5,
                'price_momentum_20': price_momentum_20
            }
            
        except Exception as e:
            return {
                'timeframe': timeframe,
                'trend_score': 0,
                'trend_direction': "分析失敗",
                'trend_factors': [f"錯誤: {str(e)}"],
                'error': str(e)
            }
    
    def calculate_multi_timeframe_score(self, symbol, period='2y'):
        """
        計算多時間框架綜合評分
        """
        try:
            # 獲取多時間框架數據
            mtf_data = self.get_multi_timeframe_data(symbol, period)
            if not mtf_data:
                return None
            
            # 分析各時間框架趨勢
            timeframe_analysis = {}
            for timeframe, data in mtf_data.items():
                analysis = self.analyze_timeframe_trend(data, timeframe)
                if analysis:
                    timeframe_analysis[timeframe] = analysis
            
            if not timeframe_analysis:
                return None
            
            # 計算加權綜合評分
            total_weighted_score = 0
            total_weight = 0
            
            for timeframe, analysis in timeframe_analysis.items():
                weight = self.timeframe_weights.get(timeframe, 0)
                weighted_score = analysis['trend_score'] * weight
                total_weighted_score += weighted_score
                total_weight += weight
            
            if total_weight > 0:
                final_score = total_weighted_score / total_weight
            else:
                final_score = 0
            
            # 確定整體趨勢一致性
            trend_consistency = self.calculate_trend_consistency(timeframe_analysis)
            
            # 確定最終評級
            if final_score >= 50 and trend_consistency >= 0.7:
                overall_rating = "強烈多頭一致"
            elif final_score >= 30 and trend_consistency >= 0.6:
                overall_rating = "多頭趨勢"
            elif final_score >= 10 and trend_consistency >= 0.5:
                overall_rating = "弱多頭"
            elif final_score >= -10 and trend_consistency >= 0.4:
                overall_rating = "趨勢不明"
            elif final_score >= -30:
                overall_rating = "弱空頭"
            elif final_score >= -50:
                overall_rating = "空頭趨勢"
            else:
                overall_rating = "強烈空頭"
            
            return {
                'symbol': symbol,
                'final_score': final_score,
                'overall_rating': overall_rating,
                'trend_consistency': trend_consistency,
                'timeframe_analysis': timeframe_analysis,
                'recommendation': self.generate_mtf_recommendation(final_score, trend_consistency, timeframe_analysis)
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'final_score': 0,
                'overall_rating': "分析失敗",
                'error': str(e)
            }
    
    def calculate_trend_consistency(self, timeframe_analysis):
        """
        計算趨勢一致性
        """
        try:
            if not timeframe_analysis:
                return 0
            
            # 獲取各時間框架的趨勢方向
            trend_directions = []
            for analysis in timeframe_analysis.values():
                direction = analysis['trend_direction']
                if '多頭' in direction:
                    trend_directions.append(1)
                elif '空頭' in direction:
                    trend_directions.append(-1)
                else:
                    trend_directions.append(0)
            
            if not trend_directions:
                return 0
            
            # 計算一致性（標準差的倒數）
            if len(set(trend_directions)) == 1:
                # 完全一致
                return 1.0
            else:
                # 計算變異係數
                std_dev = np.std(trend_directions)
                mean_val = np.mean(np.abs(trend_directions))
                if mean_val > 0:
                    consistency = 1 - (std_dev / (mean_val + 1))
                    return max(0, consistency)
                else:
                    return 0.5
                    
        except Exception:
            return 0
    
    def generate_mtf_recommendation(self, final_score, trend_consistency, timeframe_analysis):
        """
        生成多時間框架建議
        """
        try:
            recommendations = []
            
            # 基於綜合評分的建議
            if final_score >= 40 and trend_consistency >= 0.7:
                recommendations.append("多時間框架強烈看多，建議積極進場")
            elif final_score >= 20 and trend_consistency >= 0.6:
                recommendations.append("多時間框架看多，建議進場")
            elif final_score >= 0 and trend_consistency >= 0.5:
                recommendations.append("多時間框架中性偏多，謹慎進場")
            elif final_score < 0:
                recommendations.append("多時間框架偏空，建議觀望")
            
            # 基於時間框架分析的具體建議
            if 'monthly' in timeframe_analysis:
                monthly = timeframe_analysis['monthly']
                if monthly['trend_score'] >= 30:
                    recommendations.append("月線趨勢良好，適合中長期持有")
                elif monthly['trend_score'] <= -30:
                    recommendations.append("月線趨勢不佳，避免長期持有")
            
            if 'weekly' in timeframe_analysis:
                weekly = timeframe_analysis['weekly']
                if weekly['trend_score'] >= 20:
                    recommendations.append("週線支持上漲，中期看好")
                elif weekly['trend_score'] <= -20:
                    recommendations.append("週線壓力較大，中期謹慎")
            
            if 'daily' in timeframe_analysis:
                daily = timeframe_analysis['daily']
                if daily['trend_score'] >= 15:
                    recommendations.append("日線短期強勢，進場時機良好")
                elif daily['trend_score'] <= -15:
                    recommendations.append("日線短期弱勢，等待更好時機")
            
            return recommendations if recommendations else ["多時間框架分析無明確建議"]
            
        except Exception:
            return ["建議生成失敗"]
