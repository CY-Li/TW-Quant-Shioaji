#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合股票分析程式
資深股票交易員設計：結合Long Days和Long Signal Price分析
找出被低估、可抄底、接下來30天有高機率上漲修正的股票
"""

import json
import pandas as pd
import sys
import os

# 設置控制台編碼以支持 Unicode 字符
if sys.platform.startswith('win'):
    try:
        # 嘗試設置 UTF-8 編碼
        os.system('chcp 65001 > nul')
    except:
        pass
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
import time
import pytz
warnings.filterwarnings('ignore')
from pathlib import Path
from core.enhanced_confirmation_system import EnhancedConfirmationSystem
from core.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from core.bps_optimizer import BPSOptimizer

class IntegratedStockAnalyzer:
    def __init__(self, watchlist_file='TW-Quant-Shioaji/tw_watchlist.json'):
        self.watchlist_file = watchlist_file
        self.watchlist = self.load_watchlist()
        self.stocks = self.watchlist.get('stocks', [])
        self.market_sentiment = None  # 市場情緒指標
        self.confirmation_system = EnhancedConfirmationSystem()  # 強化確認系統
        self.mtf_analyzer = MultiTimeframeAnalyzer()  # 多時間框架分析器
        self.bps_optimizer = BPSOptimizer()  # BPS 策略優化器
        
    def load_watchlist(self):
        try:
            with open(self.watchlist_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ 找不到 {self.watchlist_file}")
            return {"stocks": []}
    
    def get_stock_info(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            name = info.get('longName', info.get('shortName', symbol))
            if '.TW' in symbol:
                market = 'TWSE'
            elif '.HK' in symbol:
                market = 'HKEX'
            else:
                market = 'US'
            return {'symbol': symbol, 'name': name, 'market': market}
        except Exception as e:
            return {'symbol': symbol, 'name': symbol, 'market': 'Unknown'}
    
    def get_stock_data(self, symbol, period='60d', max_retries=3):
        # 檢查股票代號有效性
        if not symbol or symbol in ['UNKNOWN', '$UNKNOWN'] or symbol.startswith('$'):
            return None

        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, timeout=60)

                if data.empty:
                    if attempt == 0:  # 只在第一次嘗試時顯示警告
                        try:
                            print(f"  ⚠️  {symbol}: 無數據返回")
                        except UnicodeEncodeError:
                            print(f"  WARNING {symbol}: 無數據返回")
                    return None

                if len(data) < 30:
                    if attempt == 0:
                        try:
                            print(f"  ⚠️  {symbol}: 數據不足 ({len(data)}天)")
                        except UnicodeEncodeError:
                            print(f"  WARNING {symbol}: 數據不足 ({len(data)}天)")
                    return None

                return data

            except Exception as e:
                error_msg = str(e)

                if attempt == 0:  # 只在第一次嘗試時顯示詳細錯誤
                    try:
                        if "404" in error_msg or "delisted" in error_msg.lower():
                            print(f"  ❌ {symbol}: 股票可能已下市")
                            return None  # 不重試下市股票
                        elif "No data found" in error_msg:
                            print(f"  ❌ {symbol}: 無數據")
                            return None  # 不重試無數據股票
                        elif "timeout" in error_msg.lower():
                            print(f"  ⏳ {symbol}: 請求超時，重試中...")
                        else:
                            print(f"  ❌ {symbol}: 獲取數據失敗 - {error_msg}")
                    except UnicodeEncodeError:
                        if "404" in error_msg or "delisted" in error_msg.lower():
                            print(f"  ERROR {symbol}: 股票可能已下市")
                            return None
                        elif "No data found" in error_msg:
                            print(f"  ERROR {symbol}: 無數據")
                            return None
                        elif "timeout" in error_msg.lower():
                            print(f"  TIMEOUT {symbol}: 請求超時，重試中...")
                        else:
                            print(f"  ERROR {symbol}: 獲取數據失敗 - {error_msg}")

                if attempt < max_retries - 1:
                    time.sleep(2)  # 等待後重試
                else:
                    return None

        return None
    
    def calculate_technical_indicators(self, data):
        if data is None or data.empty:
            return None
            
        df = data.copy()
        
        # 移動平均線
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA30'] = df['Close'].rolling(window=30).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 布林通道
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # 成交量指標
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # 價格動量
        df['Price_Momentum'] = df['Close'].pct_change(periods=5)
        
        # KD指標
        low_min = df['Low'].rolling(window=9).min()
        high_max = df['High'].rolling(window=9).max()
        df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
        df['K'] = df['RSV'].ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        
        # SAR指標
        df['SAR'] = self.calculate_sar(df)
        
        # OBV指標
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_MA'] = df['OBV'].rolling(window=10).mean()
        
        # ADX趨勢強度指標
        df['ADX'] = self.calculate_adx(df)
        
        # 均線多頭排列強度
        df['MA_Bullish_Strength'] = self.calculate_ma_bullish_strength(df)
        
        # 價格通道斜率
        df['Price_Channel_Slope'] = self.calculate_price_channel_slope(df)
        
        # 成交量趨勢配合度
        df['Volume_Trend_Alignment'] = self.calculate_volume_trend_alignment(df)
        
        # 動量加速度指標
        df['Momentum_Acceleration'] = self.calculate_momentum_acceleration(df)
        
        # 技術指標斜率變化
        df['RSI_Slope'] = df['RSI'].diff(periods=3)
        df['MACD_Slope'] = df['MACD'].diff(periods=3)
        df['K_Slope'] = df['K'].diff(periods=3)
        
        # 相對強度比較
        df['Relative_Strength'] = self.calculate_relative_strength(df)
        
        # 上漲動能延續性
        df['Uptrend_Continuity'] = self.calculate_uptrend_continuity(df)
        
        # 波動率評估
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['Volatility_Ratio'] = df['Volatility'] / df['Close'].rolling(window=20).mean()
        
        # 動態停損建議
        df['Dynamic_Stop_Loss'] = self.calculate_dynamic_stop_loss(df)
        
        # 支撐位可靠性
        df['Support_Reliability'] = self.calculate_support_reliability(df)
        
        # ===== 新增：短期趨勢反轉識別指標 =====
        
        # 趨勢反轉確認指標
        df['Trend_Reversal_Confirmation'] = self.calculate_trend_reversal_confirmation(df)
        
        # 反轉強度評估
        df['Reversal_Strength'] = self.calculate_reversal_strength(df)
        
        # 反轉可信度驗證
        df['Reversal_Reliability'] = self.calculate_reversal_reliability(df)
        
        # 短期動能轉折點
        df['Short_Term_Momentum_Turn'] = self.calculate_short_term_momentum_turn(df)
        
        # 價格結構反轉
        df['Price_Structure_Reversal'] = self.calculate_price_structure_reversal(df)
        
        return df
    
    def calculate_sar(self, df, af=None, max_af=None):
        """
        智能動態SAR計算
        根據股票波動性自動調整參數
        """
        # 計算股票的歷史波動性
        volatility = df['Close'].pct_change().std() * np.sqrt(252)

        # 根據波動性動態調整參數
        if af is None:
            if volatility > 0.4:  # 高波動股票
                af = 0.015  # 較小的加速因子，避免過於敏感
            elif volatility > 0.25:  # 中等波動
                af = 0.02   # 標準參數
            else:  # 低波動股票
                af = 0.025  # 較大的加速因子，提高敏感度

        if max_af is None:
            if volatility > 0.4:
                max_af = 0.15  # 降低最大加速因子
            elif volatility > 0.25:
                max_af = 0.2   # 標準參數
            else:
                max_af = 0.25  # 提高最大加速因子

        # 改進的初始趨勢判斷
        if len(df) < 5:
            # 對於數據不足的情況，返回一個簡單的SAR序列
            close_price = df['Close'].iloc[0] if not df.empty else 0
            return pd.Series([close_price * 0.98] * len(df), index=df.index)

        # 使用前5天的趨勢來判斷初始方向
        initial_trend = 1 if df['Close'].iloc[4] > df['Close'].iloc[0] else -1

        sar = []
        if initial_trend == 1:
            sar.append(df['Low'].iloc[:5].min())
            ep = df['High'].iloc[:5].max()
        else:
            sar.append(df['High'].iloc[:5].max())
            ep = df['Low'].iloc[:5].min()

        trend = initial_trend
        af_val = af

        for i in range(1, len(df)):
            prev_sar = sar[-1]

            if trend == 1:  # 上升趨勢
                sar_val = prev_sar + af_val * (ep - prev_sar)

                # 防止SAR超過前兩天的最低價
                if i >= 2:
                    sar_val = min(sar_val, df['Low'].iloc[i-1], df['Low'].iloc[i-2])

                if df['Low'].iloc[i] < sar_val:
                    # 趨勢反轉
                    trend = -1
                    sar_val = ep
                    ep = df['Low'].iloc[i]
                    af_val = af
                else:
                    if df['High'].iloc[i] > ep:
                        ep = df['High'].iloc[i]
                        af_val = min(af_val + af, max_af)
            else:  # 下降趨勢
                sar_val = prev_sar + af_val * (ep - prev_sar)

                # 防止SAR低於前兩天的最高價
                if i >= 2:
                    sar_val = max(sar_val, df['High'].iloc[i-1], df['High'].iloc[i-2])

                if df['High'].iloc[i] > sar_val:
                    # 趨勢反轉
                    trend = 1
                    sar_val = ep
                    ep = df['High'].iloc[i]
                    af_val = af
                else:
                    if df['Low'].iloc[i] < ep:
                        ep = df['Low'].iloc[i]
                        af_val = min(af_val + af, max_af)

            sar.append(sar_val)

        # 確保返回的SAR序列沒有無效值
        sar_series = pd.Series(sar, index=df.index)

        # 處理可能的無效值
        sar_series = sar_series.ffill()  # 前向填充
        sar_series = sar_series.fillna(df['Close'])     # 如果還有NaN，用收盤價填充

        # 確保SAR值在合理範圍內（不能是負數或過大）
        sar_series = sar_series.clip(lower=df['Close'].min() * 0.5, upper=df['Close'].max() * 1.5)

        return sar_series

    def analyze_market_sentiment(self):
        """
        分析整體市場情緒 (台股優化版)
        使用 ^TWII (加權指數), 2330 (台積電指標) 等
        """
        try:
            # 獲取市場指標數據
            m_df = yf.download('^TWII', period='30d', progress=False, auto_adjust=True)
            vix_df = yf.download('^VIX', period='30d', progress=False, auto_adjust=True) # 雖然是美股 VIX，但仍具參考價值

            if m_df.empty:
                return {'sentiment': 'neutral', 'score': 50, 'factors': ['數據不足']}
            
            if isinstance(m_df.columns, pd.MultiIndex): m_df.columns = m_df.columns.get_level_values(0)
            if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)

            sentiment_score = 50  # 基準分數
            factors = []

            # 加權指數趨勢分析
            m_ma5 = m_df['Close'].rolling(5).mean().iloc[-1]
            m_ma20 = m_df['Close'].rolling(20).mean().iloc[-1]
            m_current = m_df['Close'].iloc[-1]

            if m_current > m_ma5 > m_ma20:
                sentiment_score += 15
                factors.append("大盤多頭排列")
            elif m_current < m_ma5 < m_ma20:
                sentiment_score -= 15
                factors.append("大盤空頭排列")

            # VIX恐慌指數分析 (逆向指標)
            if not vix_df.empty:
                vix_current = vix_df['Close'].iloc[-1]
                if vix_current < 20:
                    sentiment_score += 10
                    factors.append("市場情緒穩定 (VIX低位)")
                elif vix_current > 30:
                    sentiment_score -= 15
                    factors.append("市場恐慌 (VIX高位)")

            sentiment = 'neutral'
            if sentiment_score >= 65: sentiment = 'bullish'
            elif sentiment_score <= 35: sentiment = 'bearish'

            return {'sentiment': sentiment, 'score': sentiment_score, 'factors': factors}
        except Exception as e:
            return {'sentiment': 'neutral', 'score': 50, 'factors': [f'錯誤: {str(e)}']}

    def calculate_volatility_risk_score(self, df):
        """
        計算波動性風險評分
        高波動性股票需要更嚴格的進場條件
        """
        try:
            # 計算多種波動性指標
            returns = df['Close'].pct_change().dropna()

            # 1. 歷史波動率（年化）
            historical_vol = returns.std() * np.sqrt(252)

            # 2. 近期波動率（最近10天）
            recent_vol = returns.tail(10).std() * np.sqrt(252)

            # 3. 價格跳空分析
            gaps = []
            for i in range(1, len(df)):
                gap = abs(df['Open'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]
                gaps.append(gap)
            avg_gap = np.mean(gaps) if gaps else 0

            # 4. 日內波動率
            intraday_vol = ((df['High'] - df['Low']) / df['Close']).mean()

            # 風險評分計算
            risk_score = 0
            risk_factors = []

            # 歷史波動率評估
            if historical_vol > 0.4:
                risk_score += 30
                risk_factors.append("高歷史波動率")
            elif historical_vol > 0.25:
                risk_score += 15
                risk_factors.append("中等波動率")
            elif historical_vol < 0.15:
                risk_score -= 10
                risk_factors.append("低波動率")

            # 近期波動率變化
            if recent_vol > historical_vol * 1.5:
                risk_score += 20
                risk_factors.append("近期波動率激增")
            elif recent_vol < historical_vol * 0.7:
                risk_score -= 5
                risk_factors.append("近期波動率下降")

            # 跳空風險
            if avg_gap > 0.03:
                risk_score += 15
                risk_factors.append("頻繁跳空")

            # 日內波動
            if intraday_vol > 0.05:
                risk_score += 10
                risk_factors.append("日內波動大")

            # 確定風險等級
            if risk_score >= 50:
                risk_level = 'very_high'
            elif risk_score >= 30:
                risk_level = 'high'
            elif risk_score >= 15:
                risk_level = 'medium'
            elif risk_score >= 0:
                risk_level = 'low'
            else:
                risk_level = 'very_low'

            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'historical_vol': historical_vol,
                'recent_vol': recent_vol,
                'avg_gap': avg_gap,
                'intraday_vol': intraday_vol,
                'risk_factors': risk_factors
            }

        except Exception as e:
            return {
                'risk_score': 25,
                'risk_level': 'medium',
                'historical_vol': 0.2,
                'recent_vol': 0.2,
                'avg_gap': 0.01,
                'intraday_vol': 0.03,
                'risk_factors': ['計算失敗']
            }

    def calculate_entry_timing_score(self, df):
        """
        計算進場時機評分
        分析短期價格行為和動量特徵
        """
        try:
            timing_score = 0
            timing_factors = []

            # 1. 短期價格結構分析
            recent_prices = df['Close'].tail(5)
            recent_lows = df['Low'].tail(5)
            recent_highs = df['High'].tail(5)
            current_price = df['Close'].iloc[-1]

            # 檢查是否在近期低點附近
            recent_low = recent_lows.min()
            if current_price <= recent_low * 1.02:
                timing_score += 20
                timing_factors.append("接近近期低點")

            # 檢查是否形成雙底或多重底部
            if len(recent_lows) >= 3:
                low_points = []
                for i in range(1, len(recent_lows)-1):
                    if recent_lows.iloc[i] <= recent_lows.iloc[i-1] and recent_lows.iloc[i] <= recent_lows.iloc[i+1]:
                        low_points.append(recent_lows.iloc[i])

                if len(low_points) >= 2:
                    # 檢查是否為上升底部
                    if low_points[-1] > low_points[0]:
                        timing_score += 25
                        timing_factors.append("上升底部結構")

            # 2. 短期動量分析
            momentum_1d = df['Close'].pct_change(1).iloc[-1]
            momentum_3d = df['Close'].pct_change(3).iloc[-1]
            momentum_5d = df['Close'].pct_change(5).iloc[-1]

            # 動量轉折檢測
            if momentum_1d > 0 and momentum_3d <= 0:
                timing_score += 15
                timing_factors.append("短期動量轉正")

            if momentum_1d > momentum_3d > momentum_5d:
                timing_score += 10
                timing_factors.append("動量加速向上")

            # 3. 成交量確認
            volume_ratio = df['Volume_Ratio'].iloc[-1]
            volume_trend = df['Volume_Ratio'].tail(3).mean()

            if volume_ratio > 1.2 and momentum_1d > 0:
                timing_score += 15
                timing_factors.append("放量上漲")
            elif volume_ratio > 1.5 and momentum_1d < 0:
                timing_score -= 10
                timing_factors.append("放量下跌")

            # 4. 技術指標同步性
            rsi_current = df['RSI'].iloc[-1]
            rsi_prev = df['RSI'].iloc[-2] if len(df) > 1 else rsi_current
            macd_hist_current = df['MACD_Histogram'].iloc[-1]
            macd_hist_prev = df['MACD_Histogram'].iloc[-2] if len(df) > 1 else macd_hist_current

            # RSI從超賣區反彈
            if rsi_prev < 35 and rsi_current > 35:
                timing_score += 20
                timing_factors.append("RSI超賣反彈")

            # MACD柱狀圖轉正
            if macd_hist_prev <= 0 and macd_hist_current > 0:
                timing_score += 15
                timing_factors.append("MACD柱狀圖轉正")

            # 5. 長期趨勢確認 (200 MA)
            if 'MA200' in df.columns and pd.notna(df['MA200'].iloc[-1]):
                if current_price > df['MA200'].iloc[-1]:
                    timing_score += 15
                    timing_factors.append("股價 > 200MA (長多格局)")

            # 6. 支撐位測試
            support_levels = [
                df['MA20'].iloc[-1],
                df['BB_Lower'].iloc[-1],
                df['SAR'].iloc[-1]
            ]

            for support in support_levels:
                if pd.notna(support) and 0.98 <= current_price / support <= 1.03:
                    timing_score += 10
                    timing_factors.append("測試關鍵支撐位")
                    break

            # 6. 反轉K線形態
            if len(df) >= 3:
                # 檢查錘子線或十字星
                body_size = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])
                total_range = df['High'].iloc[-1] - df['Low'].iloc[-1]
                lower_shadow = min(df['Open'].iloc[-1], df['Close'].iloc[-1]) - df['Low'].iloc[-1]

                if total_range > 0:
                    body_ratio = body_size / total_range
                    shadow_ratio = lower_shadow / total_range

                    # 錘子線：小實體，長下影線
                    if body_ratio < 0.3 and shadow_ratio > 0.6:
                        timing_score += 15
                        timing_factors.append("錘子線形態")

                    # 十字星：極小實體
                    elif body_ratio < 0.1:
                        timing_score += 10
                        timing_factors.append("十字星形態")

            return {
                'timing_score': timing_score,
                'timing_factors': timing_factors,
                'momentum_1d': momentum_1d,
                'momentum_3d': momentum_3d,
                'volume_confirmation': volume_ratio > 1.2 and momentum_1d > 0
            }

        except Exception as e:
            return {
                'timing_score': 0,
                'timing_factors': ['計算失敗'],
                'momentum_1d': 0,
                'momentum_3d': 0,
                'volume_confirmation': False
            }

    def calculate_enhanced_sar_signals(self, df):
        """
        增強的SAR信號計算，包含確認機制
        """
        sar = self.calculate_sar(df)
        df_temp = df.copy()
        df_temp['SAR'] = sar

        signals = []
        confirmations = []

        for i in range(2, len(df_temp)):
            current_price = df_temp['Close'].iloc[i]
            current_sar = df_temp['SAR'].iloc[i]
            prev_price = df_temp['Close'].iloc[i-1]
            prev_sar = df_temp['SAR'].iloc[i-1]

            # 基本SAR信號
            basic_signal = None
            if prev_price >= prev_sar and current_price < current_sar:
                basic_signal = 'SELL'
            elif prev_price <= prev_sar and current_price > current_sar:
                basic_signal = 'BUY'

            # 信號確認機制
            confirmation_score = 0
            confirmation_factors = []

            if basic_signal == 'SELL':
                # 賣出信號確認
                # 1. 成交量確認
                if df_temp['Volume_Ratio'].iloc[i] > 1.2:
                    confirmation_score += 1
                    confirmation_factors.append("放量下跌")

                # 2. RSI確認
                if df_temp['RSI'].iloc[i] > 60:
                    confirmation_score += 1
                    confirmation_factors.append("RSI偏高")

                # 3. MACD確認
                if (df_temp['MACD_Histogram'].iloc[i] < df_temp['MACD_Histogram'].iloc[i-1] and
                    df_temp['MACD_Histogram'].iloc[i] < 0):
                    confirmation_score += 1
                    confirmation_factors.append("MACD轉弱")

                # 4. 均線確認
                if current_price < df_temp['MA5'].iloc[i]:
                    confirmation_score += 1
                    confirmation_factors.append("跌破5日均線")

                # 5. 連續下跌確認
                if (current_price < prev_price and
                    prev_price < df_temp['Close'].iloc[i-2]):
                    confirmation_score += 1
                    confirmation_factors.append("連續下跌")

            elif basic_signal == 'BUY':
                # 買入信號確認（雖然主要用於停損，但可作為參考）
                # 1. 成交量確認
                if df_temp['Volume_Ratio'].iloc[i] > 1.2:
                    confirmation_score += 1
                    confirmation_factors.append("放量上漲")

                # 2. RSI確認
                if 30 < df_temp['RSI'].iloc[i] < 70:
                    confirmation_score += 1
                    confirmation_factors.append("RSI健康")

                # 3. MACD確認
                if (df_temp['MACD_Histogram'].iloc[i] > df_temp['MACD_Histogram'].iloc[i-1] and
                    df_temp['MACD_Histogram'].iloc[i] > 0):
                    confirmation_score += 1
                    confirmation_factors.append("MACD轉強")

            signals.append(basic_signal)
            confirmations.append({
                'score': confirmation_score,
                'factors': confirmation_factors,
                'max_score': 5 if basic_signal == 'SELL' else 3
            })

        # 補齊前面的空值
        signals = [None, None] + signals
        confirmations = [{'score': 0, 'factors': [], 'max_score': 0},
                        {'score': 0, 'factors': [], 'max_score': 0}] + confirmations

        return {
            'sar': sar,
            'signals': signals,
            'confirmations': confirmations
        }
    
    def calculate_adx(self, df, period=14):
        """計算ADX趨勢強度指標"""
        try:
            # 計算+DM和-DM
            high_diff = df['High'].diff()
            low_diff = df['Low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            # 計算TR (True Range)
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift(1))
            tr3 = abs(df['Low'] - df['Close'].shift(1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # 平滑處理
            plus_di = pd.Series(plus_dm).rolling(window=period).mean() / pd.Series(tr).rolling(window=period).mean() * 100
            minus_di = pd.Series(minus_dm).rolling(window=period).mean() / pd.Series(tr).rolling(window=period).mean() * 100
            
            # 計算DX和ADX
            dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
            adx = dx.rolling(window=period).mean()
            
            return adx.fillna(0)
        except:
            return pd.Series(0, index=df.index)
    
    def calculate_ma_bullish_strength(self, df):
        """計算均線多頭排列強度"""
        try:
            # 檢查均線多頭排列
            ma5 = df['MA5']
            ma10 = df['MA10']
            ma20 = df['MA20']
            ma30 = df['MA30']
            ma60 = df['MA60']
            
            # 計算多頭排列強度 (0-100)
            bullish_count = 0
            total_checks = 0
            
            # 檢查各均線排列
            if len(df) > 0:
                if ma5.iloc[-1] > ma10.iloc[-1]:
                    bullish_count += 1
                total_checks += 1
                
                if ma10.iloc[-1] > ma20.iloc[-1]:
                    bullish_count += 1
                total_checks += 1
                
                if ma20.iloc[-1] > ma30.iloc[-1]:
                    bullish_count += 1
                total_checks += 1
                
                if ma30.iloc[-1] > ma60.iloc[-1]:
                    bullish_count += 1
                total_checks += 1
                
                # 計算均線斜率
                ma5_slope = (ma5.iloc[-1] - ma5.iloc[-5]) / ma5.iloc[-5] if len(df) >= 5 else 0
                ma20_slope = (ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5] if len(df) >= 5 else 0
                
                if ma5_slope > 0:
                    bullish_count += 1
                total_checks += 1
                
                if ma20_slope > 0:
                    bullish_count += 1
                total_checks += 1
            
            strength = (bullish_count / total_checks * 100) if total_checks > 0 else 0
            return pd.Series([strength] * len(df), index=df.index)
        except:
            return pd.Series(0, index=df.index)
    
    def calculate_price_channel_slope(self, df, period=20):
        """計算價格通道斜率"""
        try:
            # 計算價格通道
            high_channel = df['High'].rolling(window=period).max()
            low_channel = df['Low'].rolling(window=period).min()
            mid_channel = (high_channel + low_channel) / 2
            
            # 計算斜率
            if len(df) >= period + 5:
                current_mid = mid_channel.iloc[-1]
                prev_mid = mid_channel.iloc[-5]
                slope = (current_mid - prev_mid) / prev_mid * 100
            else:
                slope = 0
            
            return pd.Series([slope] * len(df), index=df.index)
        except:
            return pd.Series(0, index=df.index)
    
    def calculate_volume_trend_alignment(self, df):
        """計算成交量趨勢配合度"""
        try:
            # 計算價格趨勢
            price_trend = df['Close'].pct_change(periods=5)
            
            # 計算成交量趨勢
            volume_trend = df['Volume'].pct_change(periods=5)
            
            # 評估配合度
            alignment_score = 0
            
            if len(df) > 5:
                # 價格上漲時成交量放大
                if price_trend.iloc[-1] > 0 and volume_trend.iloc[-1] > 0:
                    alignment_score += 50
                
                # 價格下跌時成交量萎縮
                if price_trend.iloc[-1] < 0 and volume_trend.iloc[-1] < 0:
                    alignment_score += 30
                
                # 成交量均線向上
                if df['Volume_MA'].iloc[-1] > df['Volume_MA'].iloc[-5]:
                    alignment_score += 20
            
            return pd.Series([alignment_score] * len(df), index=df.index)
        except:
            return pd.Series(0, index=df.index)
    
    def calculate_momentum_acceleration(self, df):
        """計算動量加速度指標"""
        try:
            # 計算價格動量
            momentum_5d = df['Close'].pct_change(periods=5)
            momentum_10d = df['Close'].pct_change(periods=10)
            
            # 計算動量加速度
            if len(df) >= 10:
                acceleration = momentum_5d.iloc[-1] - momentum_10d.iloc[-1]
            else:
                acceleration = 0
            
            return pd.Series([acceleration] * len(df), index=df.index)
        except:
            return pd.Series(0, index=df.index)
    
    def calculate_relative_strength(self, df):
        """計算相對強度比較"""
        try:
            # 簡化的相對強度計算
            # 實際應用中可能需要與大盤或同業比較
            current_price = df['Close'].iloc[-1]
            price_20d_ago = df['Close'].iloc[-20] if len(df) >= 20 else df['Close'].iloc[0]
            
            relative_strength = (current_price / price_20d_ago - 1) * 100
            return pd.Series([relative_strength] * len(df), index=df.index)
        except:
            return pd.Series(0, index=df.index)
    
    def calculate_uptrend_continuity(self, df):
        """計算上漲動能延續性"""
        try:
            continuity_score = 0
            
            # 檢查連續上漲天數
            up_days = 0
            for i in range(min(10, len(df)-1)):
                if df['Close'].iloc[-(i+1)] > df['Close'].iloc[-(i+2)]:
                    up_days += 1
                else:
                    break
            
            continuity_score += up_days * 10
            
            # 檢查技術指標持續向上
            if len(df) >= 5:
                if df['RSI'].iloc[-1] > df['RSI'].iloc[-5]:
                    continuity_score += 20
                if df['MACD'].iloc[-1] > df['MACD'].iloc[-5]:
                    continuity_score += 20
                if df['K'].iloc[-1] > df['K'].iloc[-5]:
                    continuity_score += 20
            
            return pd.Series([continuity_score] * len(df), index=df.index)
        except:
            return pd.Series(0, index=df.index)
    
    def calculate_dynamic_stop_loss(self, df):
        """計算動態停損建議"""
        try:
            # 基於ATR的動態停損
            atr = df['Volatility'].iloc[-1] * 2  # 簡化ATR計算
            current_price = df['Close'].iloc[-1]
            
            # 停損位 = 當前價格 - 2倍ATR
            stop_loss = current_price - (atr * 2)
            
            return pd.Series([stop_loss] * len(df), index=df.index)
        except:
            return pd.Series(0, index=df.index)
    
    def calculate_support_reliability(self, df):
        """計算支撐位可靠性"""
        try:
            reliability_score = 0
            current_price = df['Close'].iloc[-1]
            
            # 檢查多重支撐
            supports = [
                df['MA20'].iloc[-1],
                df['MA30'].iloc[-1],
                df['BB_Lower'].iloc[-1],
                df['SAR'].iloc[-1]
            ]
            
            # 計算支撐位數量
            support_count = sum(1 for s in supports if current_price > s * 0.95 and current_price < s * 1.05)
            reliability_score = support_count * 25
            
            return pd.Series([reliability_score] * len(df), index=df.index)
        except:
            return pd.Series(0, index=df.index)
    
    def calculate_trend_reversal_confirmation(self, df):
        """計算趨勢反轉確認指標"""
        score = 0
        
        # 1. 價格結構反轉確認
        current_price = df['Close'].iloc[-1]
        ma20 = df['MA20'].iloc[-1]
        ma5 = df['MA5'].iloc[-1]
        
        # 價格突破均線
        if current_price > ma20 and current_price > ma5:
            score += 20
        elif current_price > ma20:
            score += 10
        
        # 2. 技術指標反轉確認
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        macd_hist = df['MACD_Histogram'].iloc[-1]
        k = df['K'].iloc[-1]
        d = df['D'].iloc[-1]
        
        # RSI從超賣區反轉
        if rsi > 30 and rsi < 60:  # 避免超買狀態
            score += 15
        elif rsi < 30:
            score += 10
        
        # MACD反轉
        if macd > 0 and macd_hist > 0:
            score += 15
        elif macd > 0:
            score += 10
        
        # KD反轉
        if k > d and k < 40:  # 避免過高K值
            score += 10
        
        # 3. 成交量確認
        volume_ratio = df['Volume_Ratio'].iloc[-1]
        if volume_ratio > 1.2:
            score += 10
        elif volume_ratio > 1.0:
            score += 5
        
        # 4. 動量確認
        momentum = df['Price_Momentum'].iloc[-1]
        if momentum > 0:
            score += 10
        
        # 5. 布林通道位置確認
        bb_upper = df['BB_Upper'].iloc[-1]
        bb_lower = df['BB_Lower'].iloc[-1]
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        if bb_position < 0.5:  # 避免接近上軌
            score += 10
        elif bb_position < 0.7:
            score += 5
        
        return min(100, score)
    
    def calculate_reversal_strength(self, df):
        """計算反轉強度指標"""
        score = 0
        
        # 1. 價格動量強度
        momentum_5d = df['Price_Momentum'].iloc[-1]
        momentum_10d = df['Close'].pct_change(periods=10).iloc[-1]
        
        if momentum_5d > 0.02:  # 強勁動量
            score += 20
        elif momentum_5d > 0:
            score += 10
        
        if momentum_10d > 0.05:  # 中期強勁動量
            score += 15
        elif momentum_10d > 0:
            score += 10
        
        # 2. 技術指標強度
        rsi_slope = df['RSI_Slope'].iloc[-1]
        macd_slope = df['MACD_Slope'].iloc[-1]
        k_slope = df['K_Slope'].iloc[-1]
        
        # 指標斜率強度
        positive_slopes = sum([rsi_slope > 0, macd_slope > 0, k_slope > 0])
        if positive_slopes == 3:
            score += 20
        elif positive_slopes == 2:
            score += 15
        elif positive_slopes == 1:
            score += 10
        
        # 3. 成交量強度
        volume_ratio = df['Volume_Ratio'].iloc[-1]
        if volume_ratio > 1.5:
            score += 15
        elif volume_ratio > 1.2:
            score += 10
        elif volume_ratio > 1.0:
            score += 5
        
        # 4. 均線排列強度
        ma_bullish_strength = df['MA_Bullish_Strength'].iloc[-1]
        if ma_bullish_strength > 80:
            score += 15
        elif ma_bullish_strength > 60:
            score += 10
        
        # 5. 價格通道斜率強度
        price_channel_slope = df['Price_Channel_Slope'].iloc[-1]
        if price_channel_slope > 1:
            score += 10
        elif price_channel_slope > 0:
            score += 5
        
        return min(100, score)
    
    def calculate_reversal_reliability(self, df):
        """計算反轉可信度指標"""
        score = 0
        
        # 1. 多重技術指標一致性
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        macd_hist = df['MACD_Histogram'].iloc[-1]
        k = df['K'].iloc[-1]
        d = df['D'].iloc[-1]
        
        # 指標一致性檢查
        bullish_indicators = 0
        
        # RSI條件
        if 30 < rsi < 70:  # 避免極端值
            bullish_indicators += 1
        
        # MACD條件
        if macd > 0 and macd_hist > 0:
            bullish_indicators += 1
        elif macd > 0:
            bullish_indicators += 0.5
        
        # KD條件
        if k > d and k < 40:
            bullish_indicators += 1
        
        # SAR條件
        current_sar = df['SAR'].iloc[-1]
        if df['Close'].iloc[-1] > current_sar:
            bullish_indicators += 1
        
        # OBV條件
        current_obv = df['OBV'].iloc[-1]
        if current_obv > df['OBV_MA'].iloc[-1]:
            bullish_indicators += 1
        
        # 根據一致性給分
        if bullish_indicators >= 4:
            score += 30
        elif bullish_indicators >= 3:
            score += 20
        elif bullish_indicators >= 2:
            score += 15
        
        # 2. 成交量可靠性
        volume_ratio = df['Volume_Ratio'].iloc[-1]
        if 0.8 <= volume_ratio <= 2.0:  # 合理範圍
            score += 20
        elif volume_ratio > 2.0:
            score += 10  # 過高成交量可能不可靠
        
        # 3. 價格位置可靠性
        bb_upper = df['BB_Upper'].iloc[-1]
        bb_lower = df['BB_Lower'].iloc[-1]
        bb_position = (df['Close'].iloc[-1] - bb_lower) / (bb_upper - bb_lower)
        
        if 0.2 <= bb_position <= 0.7:  # 合理位置
            score += 20
        elif bb_position < 0.2:
            score += 15
        elif bb_position > 0.8:
            score += 5  # 接近上軌不可靠
        
        # 4. 趨勢強度可靠性
        adx = df['ADX'].iloc[-1]
        if 20 <= adx <= 40:  # 適中趨勢強度
            score += 15
        elif adx > 40:
            score += 10
        elif adx < 20:
            score += 5
        
        # 5. 支撐位可靠性
        support_reliability = df['Support_Reliability'].iloc[-1]
        if support_reliability > 60:
            score += 15
        elif support_reliability > 40:
            score += 10
        
        return min(100, score)
    
    def calculate_short_term_momentum_turn(self, df):
        """計算短期動能轉折點"""
        try:
            turn_score = 0
            
            if len(df) >= 7:
                # 檢查動能轉折
                momentum_3d = df['Close'].pct_change(periods=3).iloc[-1]
                momentum_7d = df['Close'].pct_change(periods=7).iloc[-1]
                
                # 短期動能轉正且強於中期
                if momentum_3d > 0 and momentum_3d > momentum_7d:
                    turn_score += 30
                
                # 檢查RSI動能轉折
                rsi_current = df['RSI'].iloc[-1]
                rsi_3d_ago = df['RSI'].iloc[-4]
                if rsi_3d_ago < 40 and rsi_current > 45:
                    turn_score += 25
                
                # 檢查MACD動能轉折
                macd_hist_current = df['MACD_Histogram'].iloc[-1]
                macd_hist_3d_ago = df['MACD_Histogram'].iloc[-4]
                if macd_hist_3d_ago < 0 and macd_hist_current > 0:
                    turn_score += 25
                
                # 檢查價格結構轉折
                if len(df) >= 5:
                    # 檢查是否形成低點抬高的結構
                    low_1 = df['Low'].iloc[-5:-2].min()
                    low_2 = df['Low'].iloc[-2:].min()
                    if low_2 > low_1:
                        turn_score += 20
            
            return pd.Series([turn_score] * len(df), index=df.index)
        except:
            return pd.Series(0, index=df.index)
    
    def calculate_price_structure_reversal(self, df):
        """計算價格結構反轉"""
        try:
            structure_score = 0
            
            if len(df) >= 10:
                # 檢查雙底或W底結構
                lows = df['Low'].iloc[-10:]
                if len(lows) >= 10:
                    # 尋找兩個低點
                    low_points = []
                    for i in range(1, len(lows)-1):
                        if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                            low_points.append((i, lows.iloc[i]))
                    
                    if len(low_points) >= 2:
                        # 檢查第二個低點是否高於第一個
                        if low_points[-1][1] > low_points[-2][1]:
                            structure_score += 40
                
                # 檢查突破頸線
                if len(df) >= 5:
                    recent_highs = df['High'].iloc[-5:]
                    neckline = recent_highs.max()
                    if df['Close'].iloc[-1] > neckline * 0.98:  # 接近突破
                        structure_score += 30
                
                # 檢查價格在支撐位反彈
                current_price = df['Close'].iloc[-1]
                support_levels = [
                    df['MA20'].iloc[-1],
                    df['BB_Lower'].iloc[-1],
                    df['SAR'].iloc[-1]
                ]
                
                for support in support_levels:
                    if 0.98 < current_price / support < 1.02:
                        structure_score += 15
                        break
            
            return pd.Series([structure_score] * len(df), index=df.index)
        except:
            return pd.Series(0, index=df.index)
    
    def detect_bullish_signals(self, df):
        if df is None or df.empty:
            return []
        signals = []
        last_signal_date = None  # 時間過濾：避免短期內重複訊號
        
        for i in range(20, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # 價格過濾：避免明顯下跌趨勢中的假訊號
            if i >= 5:
                recent_trend = df['Close'].iloc[i-5:i].pct_change().mean()
                if recent_trend < -0.02:  # 近5日平均跌幅>2%，跳過
                    continue
            
            # 時間過濾：避免短期內重複訊號
            if last_signal_date is not None:
                days_since_last = (df.index[i] - last_signal_date).days
                if days_since_last < 3:  # 3天內不重複訊號
                    continue
            
            # 修正：加入超買過濾
            if current['RSI'] > 75:  # 嚴重超買狀態跳過
                continue
            
            # 修正：加入布林通道位置過濾
            bb_upper = current['BB_Upper']
            bb_lower = current['BB_Lower']
            bb_position = (current['Close'] - bb_lower) / (bb_upper - bb_lower)
            if bb_position > 0.85:  # 接近上軌跳過
                continue
            
            bullish_conditions = []
            
            # 黃金交叉+放量（改為1.5倍）
            if (current['MA5'] > current['MA20'] and prev['MA5'] <= prev['MA20'] and current['Volume_Ratio'] > 1.5):
                bullish_conditions.append("黃金交叉+放量")
            
            # MACD柱狀圖轉正且RSI未超買
            if (current['MACD_Histogram'] > 0 and prev['MACD_Histogram'] <= 0 and current['RSI'] < 70):
                bullish_conditions.append("MACD柱狀圖轉正+RSI未超買")
            
            # KD低檔交叉（加入K<20且D<25條件）
            if (current['K'] > current['D'] and prev['K'] <= prev['D'] and current['K'] < 20 and current['D'] < 25):
                bullish_conditions.append("KD低檔交叉")
            
            # SAR翻多（加入連續2根K線確認）
            if (i >= 2 and 
                current['Close'] > current['SAR'] and 
                prev['Close'] > prev['SAR'] and 
                df['Close'].iloc[i-2] <= df['SAR'].iloc[i-2]):
                bullish_conditions.append("SAR翻多")
            
            # OBV突破均線
            if (current['OBV'] > current['OBV_MA'] and prev['OBV'] <= prev['OBV_MA']):
                bullish_conditions.append("OBV突破均線")
            
            # RSI超賣反轉
            if (current['RSI'] > 30 and prev['RSI'] <= 30):
                bullish_conditions.append("RSI超賣反轉")
            
            # 價格突破布林下軌
            if (current['Close'] > current['BB_Lower'] and prev['Close'] <= prev['BB_Lower']):
                bullish_conditions.append("突破布林下軌")
            
            # 動量轉正
            if (current['Price_Momentum'] > 0 and prev['Price_Momentum'] <= 0):
                bullish_conditions.append("動量轉正")
            
            # ===== 新增：趨勢持續性指標 =====
            
            # ADX趨勢強度確認
            if current['ADX'] > 25:  # ADX > 25表示趨勢明確
                bullish_conditions.append("ADX趨勢強勁")
            
            # 均線多頭排列強度
            if current['MA_Bullish_Strength'] > 80:  # 多頭排列強度>80%
                bullish_conditions.append("均線多頭排列")
            
            # 價格通道斜率向上
            if current['Price_Channel_Slope'] > 0:
                bullish_conditions.append("價格通道向上")
            
            # 成交量趨勢配合
            if current['Volume_Trend_Alignment'] > 70:  # 成交量配合度>70%
                bullish_conditions.append("成交量配合")
            
            # ===== 新增：動能分析指標 =====
            
            # 動量加速度為正
            if current['Momentum_Acceleration'] > 0:
                bullish_conditions.append("動量加速")
            
            # 技術指標斜率向上
            if (current['RSI_Slope'] > 0 and current['MACD_Slope'] > 0):
                bullish_conditions.append("指標斜率向上")
            
            # 相對強度為正
            if current['Relative_Strength'] > 0:
                bullish_conditions.append("相對強度為正")
            
            # 上漲動能延續性
            if current['Uptrend_Continuity'] > 50:  # 延續性>50分
                bullish_conditions.append("上漲動能延續")
            
            # ===== 新增：短期趨勢反轉識別 =====
            
            # 修正：趨勢反轉確認（統一判斷標準）
            trend_reversal_confirmation = current['Trend_Reversal_Confirmation']
            reversal_strength = current['Reversal_Strength']
            reversal_reliability = current['Reversal_Reliability']
            short_term_momentum_turn = current['Short_Term_Momentum_Turn']
            price_structure_reversal = current['Price_Structure_Reversal']
            
            # 趨勢反轉確認（要求更嚴格）
            if trend_reversal_confirmation > 60:  # 提高門檻
                bullish_conditions.append("趨勢反轉確認")
            
            # 反轉強度
            if reversal_strength > 70:  # 提高門檻
                bullish_conditions.append("反轉強度強勁")
            
            # 反轉可信度
            if reversal_reliability > 70:  # 提高門檻
                bullish_conditions.append("反轉可信度高")
            
            # 短期動能轉折
            if short_term_momentum_turn > 60:  # 提高門檻
                bullish_conditions.append("短期動能轉折")
            
            # 價格結構反轉
            if price_structure_reversal > 50:  # 提高門檻
                bullish_conditions.append("價格結構反轉")
            
            # 修正：更嚴格的多頭訊號條件
            if len(bullish_conditions) >= 5:  # 提高條件數量要求
                # 必須包含至少2個趨勢反轉相關條件
                reversal_conditions = [
                    "趨勢反轉確認", "反轉強度強勁", "反轉可信度高", 
                    "短期動能轉折", "價格結構反轉"
                ]
                reversal_count = sum(1 for cond in bullish_conditions if cond in reversal_conditions)
                
                if reversal_count >= 2:  # 要求至少2個反轉條件
                    # 計算距離當前的天數
                    days_ago = len(df) - 1 - i

                    signals.append({
                        'date': df.index[i],
                        'price': current['Close'],
                        'conditions': bullish_conditions,
                        'signal_types': bullish_conditions,  # 添加signal_types別名
                        'days_ago': days_ago,  # 添加days_ago字段
                        'rsi': current['RSI'],
                        'macd': current['MACD'],
                        'volume_ratio': current['Volume_Ratio'],
                        'k': current['K'],
                        'd': current['D'],
                        'sar': current['SAR'],
                        'obv': current['OBV'],
                        'adx': current['ADX'],
                        'ma_bullish_strength': current['MA_Bullish_Strength'],
                        'momentum_acceleration': current['Momentum_Acceleration'],
                        'uptrend_continuity': current['Uptrend_Continuity'],
                        'trend_reversal_confirmation': current['Trend_Reversal_Confirmation'],
                        'reversal_strength': current['Reversal_Strength'],
                        'reversal_reliability': current['Reversal_Reliability'],
                        'short_term_momentum_turn': current['Short_Term_Momentum_Turn'],
                        'price_structure_reversal': current['Price_Structure_Reversal']
                    })
                    last_signal_date = df.index[i]  # 更新最後訊號日期
        
        return signals
    
    def calculate_long_signal_price(self, df):
        # 1. 近30日最低價
        min_low = df['Low'].min()
        # 2. 30日布林下軌
        ma = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        bb_lower = (ma - 2 * std).iloc[-1]
        # 3. 30日均線
        ma30 = df['MA30'].iloc[-1]
        # 4. 30日內多頭訊號日的收盤價
        signals = []
        for i in range(1, len(df)):
            # 黃金交叉
            if (df['MA5'].iloc[i] > df['MA20'].iloc[i] and 
                df['MA5'].iloc[i-1] <= df['MA20'].iloc[i-1]):
                signals.append(df['Close'].iloc[i])
            # RSI反轉
            if (df['RSI'].iloc[i] > 30 and df['RSI'].iloc[i-1] <= 30):
                signals.append(df['Close'].iloc[i])
            # MACD柱狀圖轉正
            if (df['MACD_Histogram'].iloc[i] > 0 and 
                df['MACD_Histogram'].iloc[i-1] <= 0):
                signals.append(df['Close'].iloc[i])
        if signals:
            signal_price = min(signals)
        else:
            signal_price = min_low
        # 5. 近30日最大成交量K棒的最低價
        max_vol_idx = df['Volume'].idxmax()
        max_vol_low = df.loc[max_vol_idx, 'Low'] if max_vol_idx in df.index else min_low
        # 6. RSI<30時的最低價
        rsi_oversold_lows = df['Low'][df['RSI'] < 30]
        rsi_oversold_low = rsi_oversold_lows.min() if not rsi_oversold_lows.empty else min_low
        
        # 修正：確保抄底價位低於當前價格
        current_price = df['Close'].iloc[-1]
        
        # 綜合多重支撐，取最高值但必須低於當前價格
        candidates = [min_low, bb_lower, ma30, signal_price, max_vol_low, rsi_oversold_low]
        valid_candidates = [v for v in candidates if not pd.isna(v) and v < current_price]
        
        if valid_candidates:
            long_signal_price = max(valid_candidates)
        else:
            # 如果沒有低於當前價格的支撐位，使用最低價的90%作為抄底價位
            long_signal_price = min_low * 0.9
        
        # 信心度：有幾條支撐同時接近long_signal_price
        close_count = sum([abs(long_signal_price-v)/long_signal_price < 0.03 for v in candidates if not pd.isna(v)])
        confidence = min(100, 40 + close_count*15)
        
        return long_signal_price, confidence
    
    def assess_entry_opportunity(self, df):
        """
        增強的進場機會評估
        結合市場情緒、波動性風險和時機分析
        """
        score = 0
        confidence_factors = []

        # 獲取市場情緒
        if self.market_sentiment is None:
            self.market_sentiment = self.analyze_market_sentiment()

        # 獲取波動性風險評估
        volatility_risk = self.calculate_volatility_risk_score(df)

        # 獲取進場時機評分
        timing_analysis = self.calculate_entry_timing_score(df)

        # RSI評估（修正：超買狀態大幅降低分數）
        current_rsi = df['RSI'].iloc[-1]
        if 30 <= current_rsi <= 50:
            score += 2
            confidence_factors.append("RSI從超賣區反轉")
        elif current_rsi < 30:
            score += 3
            confidence_factors.append("RSI嚴重超賣")
        elif current_rsi > 70:
            score -= 5  # 大幅降低超買分數
            confidence_factors.append("RSI超買風險")
        elif current_rsi > 60:
            score -= 2
            confidence_factors.append("RSI偏高")
        
        # MACD評估
        macd_histogram = df['MACD_Histogram'].iloc[-1]
        if df['MACD'].iloc[-1] > 0:
            score += 1
            confidence_factors.append("MACD為正")
        if macd_histogram > 0:
            score += 1
            confidence_factors.append("MACD柱狀圖為正")
        
        # KD評估（改進：K>D且K<30且D<35）
        k = df['K'].iloc[-1]
        d = df['D'].iloc[-1]
        if k > d and k < 30 and d < 35:
            score += 1
            confidence_factors.append("KD低檔交叉")
        
        # SAR評估（加入SAR趨勢向上判斷）
        current_sar = df['SAR'].iloc[-1]
        prev_sar = df['SAR'].iloc[-2] if len(df) > 1 else current_sar
        if df['Close'].iloc[-1] > current_sar and current_sar > prev_sar:
            score += 1
            confidence_factors.append("SAR翻多且趨勢向上")
        elif df['Close'].iloc[-1] > current_sar:
            score += 0.5
            confidence_factors.append("SAR翻多")
        
        # OBV評估（加入OBV斜率為正條件）
        current_obv = df['OBV'].iloc[-1]
        prev_obv = df['OBV'].iloc[-2] if len(df) > 1 else current_obv
        obv_slope = current_obv - prev_obv
        if current_obv > df['OBV_MA'].iloc[-1] and obv_slope > 0:
            score += 1
            confidence_factors.append("OBV突破均線且斜率為正")
        elif current_obv > df['OBV_MA'].iloc[-1]:
            score += 0.5
            confidence_factors.append("OBV突破均線")
        
        # 成交量評估
        if df['Volume_Ratio'].iloc[-1] > 1.5:  # 提高標準
            score += 1
            confidence_factors.append("成交量放大")
        elif df['Volume_Ratio'].iloc[-1] > 1.2:
            score += 0.5
            confidence_factors.append("成交量適中")
        elif df['Volume_Ratio'].iloc[-1] < 0.8:
            score -= 1
            confidence_factors.append("成交量萎縮")
        
        # 價格趨勢評估
        current_price = df['Close'].iloc[-1]
        ma20 = df['MA20'].iloc[-1]
        ma5 = df['MA5'].iloc[-1]
        
        if current_price > ma20:
            score += 1
            confidence_factors.append("價格在20日均線之上")
        if current_price > ma5:
            score += 1
            confidence_factors.append("價格在5日均線之上")
        
        # 布林通道評估（修正：超買位置大幅降低分數）
        bb_upper = df['BB_Upper'].iloc[-1]
        bb_lower = df['BB_Lower'].iloc[-1]
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        if bb_position < 0.3:
            score += 1
            confidence_factors.append("接近布林通道下軌")
        elif bb_position > 0.8:  # 提高超買門檻
            score -= 3  # 大幅降低超買分數
            confidence_factors.append("接近布林通道上軌風險")
        elif bb_position > 0.7:
            score -= 1
            confidence_factors.append("接近布林通道上軌")
        
        # 價格動量評估
        momentum_5d = df['Price_Momentum'].iloc[-1]
        momentum_10d = df['Close'].pct_change(periods=10).iloc[-1]
        
        if momentum_5d > 0 and momentum_10d > 0:
            score += 1
            confidence_factors.append("短期和中期動量均為正")
        elif momentum_5d < 0 and momentum_10d < 0:
            score -= 1
            confidence_factors.append("短期和中期動量均為負")
        
        # ===== 新增：趨勢持續性評估 =====
        
        # ADX趨勢強度評估
        adx = df['ADX'].iloc[-1]
        if adx > 30:
            score += 2
            confidence_factors.append("ADX趨勢強勁")
        elif adx > 20:
            score += 1
            confidence_factors.append("ADX趨勢明確")
        elif adx < 15:
            score -= 1
            confidence_factors.append("ADX趨勢不明")
        
        # 均線多頭排列強度評估
        ma_bullish_strength = df['MA_Bullish_Strength'].iloc[-1]
        if ma_bullish_strength > 90:
            score += 2
            confidence_factors.append("均線完美多頭排列")
        elif ma_bullish_strength > 80:
            score += 1
            confidence_factors.append("均線多頭排列")
        elif ma_bullish_strength < 50:
            score -= 1
            confidence_factors.append("均線排列不佳")
        
        # 價格通道斜率評估
        price_channel_slope = df['Price_Channel_Slope'].iloc[-1]
        if price_channel_slope > 2:
            score += 1
            confidence_factors.append("價格通道強勢向上")
        elif price_channel_slope > 0:
            score += 0.5
            confidence_factors.append("價格通道向上")
        elif price_channel_slope < -2:
            score -= 1
            confidence_factors.append("價格通道向下")
        
        # 成交量趨勢配合度評估
        volume_trend_alignment = df['Volume_Trend_Alignment'].iloc[-1]
        if volume_trend_alignment > 80:
            score += 1
            confidence_factors.append("成交量完美配合")
        elif volume_trend_alignment > 60:
            score += 0.5
            confidence_factors.append("成交量配合良好")
        elif volume_trend_alignment < 30:
            score -= 1
            confidence_factors.append("成交量配合不佳")
        
        # ===== 新增：動能分析評估 =====
        
        # 動量加速度評估
        momentum_acceleration = df['Momentum_Acceleration'].iloc[-1]
        if momentum_acceleration > 0.02:
            score += 1
            confidence_factors.append("動量強勁加速")
        elif momentum_acceleration > 0:
            score += 0.5
            confidence_factors.append("動量加速")
        elif momentum_acceleration < -0.02:
            score -= 1
            confidence_factors.append("動量減速")
        
        # 技術指標斜率評估
        rsi_slope = df['RSI_Slope'].iloc[-1]
        macd_slope = df['MACD_Slope'].iloc[-1]
        k_slope = df['K_Slope'].iloc[-1]
        
        if rsi_slope > 0 and macd_slope > 0 and k_slope > 0:
            score += 1
            confidence_factors.append("所有指標斜率向上")
        elif rsi_slope > 0 and macd_slope > 0:
            score += 0.5
            confidence_factors.append("主要指標斜率向上")
        elif rsi_slope < 0 and macd_slope < 0:
            score -= 1
            confidence_factors.append("指標斜率向下")
        
        # 相對強度評估
        relative_strength = df['Relative_Strength'].iloc[-1]
        if relative_strength > 5:
            score += 1
            confidence_factors.append("相對強度強勁")
        elif relative_strength > 0:
            score += 0.5
            confidence_factors.append("相對強度為正")
        elif relative_strength < -5:
            score -= 1
            confidence_factors.append("相對強度為負")
        
        # 上漲動能延續性評估
        uptrend_continuity = df['Uptrend_Continuity'].iloc[-1]
        if uptrend_continuity > 80:
            score += 2
            confidence_factors.append("上漲動能極強")
        elif uptrend_continuity > 60:
            score += 1
            confidence_factors.append("上漲動能強勁")
        elif uptrend_continuity > 40:
            score += 0.5
            confidence_factors.append("上漲動能良好")
        elif uptrend_continuity < 20:
            score -= 1
            confidence_factors.append("上漲動能不足")
        
        # ===== 新增：風險管理評估 =====
        
        # 波動率評估
        volatility_ratio = df['Volatility_Ratio'].iloc[-1]
        if volatility_ratio < 0.02:
            score += 1
            confidence_factors.append("波動率低")
        elif volatility_ratio > 0.05:
            score -= 1
            confidence_factors.append("波動率過高")
        
        # 支撐位可靠性評估
        support_reliability = df['Support_Reliability'].iloc[-1]
        if support_reliability > 75:
            score += 1
            confidence_factors.append("支撐位可靠")
        elif support_reliability > 50:
            score += 0.5
            confidence_factors.append("支撐位良好")
        elif support_reliability < 25:
            score -= 1
            confidence_factors.append("支撐位薄弱")
        
        # ===== 新增：趨勢反轉評估 =====
        
        # 趨勢反轉確認評估
        trend_reversal_confirmation = df['Trend_Reversal_Confirmation'].iloc[-1]
        if trend_reversal_confirmation > 80:
            score += 3
            confidence_factors.append("趨勢反轉高度確認")
        elif trend_reversal_confirmation > 60:
            score += 2
            confidence_factors.append("趨勢反轉確認")
        elif trend_reversal_confirmation > 40:
            score += 1
            confidence_factors.append("趨勢反轉初步確認")
        elif trend_reversal_confirmation < 20:
            score -= 1
            confidence_factors.append("無趨勢反轉跡象")
        
        # 反轉強度評估
        reversal_strength = df['Reversal_Strength'].iloc[-1]
        if reversal_strength > 80:
            score += 2
            confidence_factors.append("反轉強度極強")
        elif reversal_strength > 60:
            score += 1
            confidence_factors.append("反轉強度強勁")
        elif reversal_strength < 30:
            score -= 1
            confidence_factors.append("反轉強度不足")
        
        # 反轉可信度評估
        reversal_reliability = df['Reversal_Reliability'].iloc[-1]
        if reversal_reliability > 80:
            score += 2
            confidence_factors.append("反轉可信度極高")
        elif reversal_reliability > 60:
            score += 1
            confidence_factors.append("反轉可信度高")
        elif reversal_reliability < 40:
            score -= 1
            confidence_factors.append("反轉可信度低")
        
        # 短期動能轉折評估
        short_term_momentum_turn = df['Short_Term_Momentum_Turn'].iloc[-1]
        if short_term_momentum_turn > 80:
            score += 2
            confidence_factors.append("短期動能強勁轉折")
        elif short_term_momentum_turn > 60:
            score += 1
            confidence_factors.append("短期動能轉折")
        elif short_term_momentum_turn < 30:
            score -= 1
            confidence_factors.append("短期動能未轉折")
        
        # 價格結構反轉評估
        price_structure_reversal = df['Price_Structure_Reversal'].iloc[-1]
        if price_structure_reversal > 70:
            score += 2
            confidence_factors.append("價格結構完美反轉")
        elif price_structure_reversal > 50:
            score += 1
            confidence_factors.append("價格結構反轉")
        elif price_structure_reversal < 20:
            score -= 1
            confidence_factors.append("價格結構未反轉")
        
        # 修正：計算信心度分數（調整計算方式）
        # 基礎分數調整，考慮超買風險
        base_score = score + 15  # 調整基礎分數
        
        # 超買風險調整
        if current_rsi > 70:
            base_score *= 0.6  # 超買狀態大幅降低信心度
        elif current_rsi > 60:
            base_score *= 0.8  # 偏高狀態適度降低信心度
        
        # 布林通道位置調整
        if bb_position > 0.8:
            base_score *= 0.7  # 接近上軌大幅降低信心度
        elif bb_position > 0.7:
            base_score *= 0.9  # 接近上軌適度降低信心度
        
        confidence_score = max(0, min(100, base_score * 3))  # 調整分數計算
        
        # === 新增：市場情緒調整 ===
        market_score = self.market_sentiment['score']
        if market_score >= 70:
            score += 2
            confidence_factors.append("市場情緒樂觀")
        elif market_score >= 55:
            score += 1
            confidence_factors.append("市場情緒正面")
        elif market_score <= 30:
            score -= 3
            confidence_factors.append("市場情緒悲觀")
        elif market_score <= 45:
            score -= 1
            confidence_factors.append("市場情緒負面")

        # === 新增：波動性風險調整 ===
        risk_level = volatility_risk['risk_level']
        if risk_level == 'very_high':
            score -= 4
            confidence_factors.append("極高波動風險")
        elif risk_level == 'high':
            score -= 2
            confidence_factors.append("高波動風險")
        elif risk_level == 'very_low':
            score += 1
            confidence_factors.append("低波動風險")

        # === 新增：時機評分調整 ===
        timing_score = timing_analysis['timing_score']
        if timing_score >= 50:
            score += 3
            confidence_factors.append("絕佳進場時機")
        elif timing_score >= 30:
            score += 2
            confidence_factors.append("良好進場時機")
        elif timing_score >= 15:
            score += 1
            confidence_factors.append("適當進場時機")
        elif timing_score <= 0:
            score -= 2
            confidence_factors.append("進場時機不佳")

        # 添加時機分析的具體因素
        confidence_factors.extend(timing_analysis['timing_factors'])

        # === 新增：強化確認機制 ===
        try:
            enhanced_confirmation = self.confirmation_system.calculate_comprehensive_confirmation(df)
            confirmation_score = 0

            if enhanced_confirmation and isinstance(enhanced_confirmation.get('total_score'), (int, float)):
                confirmation_score = float(enhanced_confirmation['total_score']) / 10  # 轉換為10分制
        except Exception as e:
            print(f"強化確認機制錯誤: {e}")
            enhanced_confirmation = {'total_score': 0, 'all_factors': []}
            confirmation_score = 0

        # === 新增：多時間框架分析 ===
        symbol = getattr(self, 'current_symbol', 'UNKNOWN')
        mtf_analysis = None
        mtf_score = 0
        mtf_factors = []

        try:
            mtf_analysis = self.mtf_analyzer.calculate_multi_timeframe_score(symbol)

            if mtf_analysis and isinstance(mtf_analysis.get('final_score'), (int, float)):
                mtf_score = float(mtf_analysis['final_score']) / 10  # 轉換為10分制
                mtf_factors = mtf_analysis.get('recommendation', [])

                # 根據多時間框架一致性調整評分
                trend_consistency = mtf_analysis.get('trend_consistency', 0)
                if isinstance(trend_consistency, (int, float)):
                    if trend_consistency >= 0.8:
                        mtf_score *= 1.2  # 高一致性加成
                        mtf_factors.append("多時間框架高度一致")
                    elif trend_consistency >= 0.6:
                        mtf_score *= 1.1  # 中等一致性小幅加成
                        mtf_factors.append("多時間框架較為一致")
                    elif trend_consistency < 0.4:
                        mtf_score *= 0.8  # 低一致性減分
                        mtf_factors.append("多時間框架分歧較大")
        except Exception as e:
            print(f"多時間框架分析錯誤: {e}")
            mtf_score = 0
            mtf_factors = ["多時間框架分析失敗"]

        # === 新增：傳統多重確認機制 ===
        traditional_confirmation_score = 0
        traditional_confirmation_factors = []

        # 1. 多頭訊號確認
        bullish_signals = self.detect_bullish_signals(df)
        recent_signals = [s for s in bullish_signals if s['days_ago'] <= 5]
        if len(recent_signals) >= 2:
            traditional_confirmation_score += 2
            traditional_confirmation_factors.append("多重多頭訊號確認")
        elif len(recent_signals) >= 1:
            traditional_confirmation_score += 1
            traditional_confirmation_factors.append("單一多頭訊號")

        # 2. 成交量確認
        if timing_analysis.get('volume_confirmation', False):
            traditional_confirmation_score += 2
            traditional_confirmation_factors.append("成交量確認")

        # 3. 技術指標同步確認
        if current_rsi < 50 and df['MACD'].iloc[-1] > 0:
            traditional_confirmation_score += 1
            traditional_confirmation_factors.append("RSI+MACD同步")

        # 4. 支撐位確認
        current_price = df['Close'].iloc[-1]
        long_signal_result = self.calculate_long_signal_price(df)

        # 處理返回值（可能是元組或單一值）
        if isinstance(long_signal_result, tuple):
            long_signal_price, _ = long_signal_result
        else:
            long_signal_price = long_signal_result

        if long_signal_price and isinstance(long_signal_price, (int, float)) and current_price <= long_signal_price * 1.05:
            traditional_confirmation_score += 2
            traditional_confirmation_factors.append("接近抄底價位")

        # 綜合所有確認評分
        try:
            total_confirmation_score = (float(confirmation_score) * 0.4 +
                                      float(mtf_score) * 0.4 +
                                      float(traditional_confirmation_score) * 0.2)
        except (TypeError, ValueError):
            total_confirmation_score = 0

        # 調整最終評分
        score += total_confirmation_score

        # 合併所有確認因素
        try:
            all_confirmation_factors = []

            # 添加強化確認因素
            if enhanced_confirmation and 'all_factors' in enhanced_confirmation:
                enhanced_factors = enhanced_confirmation['all_factors']
                if isinstance(enhanced_factors, list):
                    all_confirmation_factors.extend(enhanced_factors)

            # 添加多時間框架因素
            if isinstance(mtf_factors, list):
                all_confirmation_factors.extend(mtf_factors)

            # 添加傳統確認因素
            if isinstance(traditional_confirmation_factors, list):
                all_confirmation_factors.extend(traditional_confirmation_factors)

            confidence_factors.extend(all_confirmation_factors)
        except Exception as e:
            print(f"合併確認因素錯誤: {e}")
            confidence_factors.append("確認因素合併失敗")

        # 進場建議（強化版：整合多重確認機制）
        # 基本技術條件
        basic_conditions = current_rsi < 65 and bb_position < 0.7

        # 市場環境條件
        market_conditions = market_score >= 45  # 市場情緒不能太差

        # 波動性條件
        volatility_conditions = risk_level not in ['very_high']  # 避免極高波動

        # 時機條件
        timing_conditions = timing_score >= 10  # 時機評分不能太低

        # 強化確認條件
        enhanced_confirmation_conditions = enhanced_confirmation['total_score'] >= 60  # 強化確認評分

        # 多時間框架條件
        mtf_conditions = True
        if mtf_analysis:
            mtf_conditions = (mtf_analysis['final_score'] >= 20 and
                            mtf_analysis['trend_consistency'] >= 0.5)

        # 傳統確認條件
        traditional_confirmation_conditions = traditional_confirmation_score >= 3

        # 綜合確認條件（至少滿足兩個確認機制）
        confirmation_count = sum([
            enhanced_confirmation_conditions,
            mtf_conditions,
            traditional_confirmation_conditions
        ])

        comprehensive_confirmation = confirmation_count >= 2

        # 分級進場建議
        if (score >= 20 and basic_conditions and market_conditions and
            volatility_conditions and timing_conditions and
            enhanced_confirmation_conditions and mtf_conditions and
            traditional_confirmation_conditions):
            entry_advice = "極強推薦進場"
        elif (score >= 16 and basic_conditions and market_conditions and
              volatility_conditions and timing_conditions and comprehensive_confirmation):
            entry_advice = "強烈推薦進場"
        elif (score >= 12 and current_rsi < 70 and bb_position < 0.8 and
              market_conditions and timing_score >= 5 and confirmation_count >= 1):
            entry_advice = "建議進場"
        elif (score >= 8 and market_score >= 40 and
              (enhanced_confirmation_conditions or traditional_confirmation_conditions)):
            entry_advice = "謹慎觀望"
        else:
            entry_advice = "不建議進場"
        
        # 信心度等級
        if confidence_score >= 80:
            confidence_level = "極高"
        elif confidence_score >= 60:
            confidence_level = "高"
        elif confidence_score >= 40:
            confidence_level = "中等"
        elif confidence_score >= 20:
            confidence_level = "低"
        else:
            confidence_level = "極低"
        
        return entry_advice, confidence_score, confidence_level, confidence_factors
    
    def analyze_stock(self, symbol):
        print(f"分析 {symbol}...")

        # 設置當前分析的股票符號，供其他方法使用
        self.current_symbol = symbol

        # 獲取股票資訊
        stock_info = self.get_stock_info(symbol)
        
        # 獲取股票數據
        data = self.get_stock_data(symbol)
        if data is None:
            return None
        
        # 計算技術指標
        df = self.calculate_technical_indicators(data)
        if df is None or df.empty:
            return None
        
        # 確保 current_price, SAR, confidence_factors 總是存在
        current_price = df['Close'].iloc[-1] if not df.empty else None

        # 改進SAR數據處理
        current_sar = None
        if 'SAR' in df.columns and not df.empty:
            sar_value = df['SAR'].iloc[-1]
            # 檢查SAR值是否有效（不是NaN且不是無窮大）
            if pd.notna(sar_value) and np.isfinite(sar_value):
                current_sar = float(sar_value)

        entry_advice, confidence_score, confidence_level, confidence_factors = self.assess_entry_opportunity(df)

        # 檢測多頭訊號
        signals = self.detect_bullish_signals(df)
        
        # 計算Long Signal Price
        long_signal_price, long_signal_confidence = self.calculate_long_signal_price(df)
        
        base_result = {
            'symbol': symbol,
            'name': stock_info['name'],
            'market': stock_info['market'],
            'current_price': current_price,
            'sar': current_sar, # 總是包含 SAR
            'confidence_factors': confidence_factors, # 總是包含 confidence_factors
            'rsi': df['RSI'].iloc[-1] if 'RSI' in df.columns else None,
            'macd': df['MACD'].iloc[-1] if 'MACD' in df.columns else None,
            'volume_ratio': df['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df.columns else None,
            'long_signal_price': long_signal_price,
            'long_signal_confidence': long_signal_confidence,
            'entry_opportunity': entry_advice,
            'entry_advice': entry_advice,  # 添加entry_advice別名
            'confidence_score': confidence_score,
            'confidence_level': confidence_level,
        }

        if not signals:
            base_result.update({
                'long_days': None,
                'status': '無多頭訊號',
                'distance_to_signal': ((current_price - long_signal_price) / long_signal_price) * 100 if current_price and long_signal_price else None,
            })
            return base_result
        
        # 找到最近的多頭訊號
        latest_signal = signals[-1]
        current_date = df.index[-1]
        days_since_signal = (current_date - latest_signal['date']).days

        base_result.update({
            'long_days': days_since_signal,
            'latest_signal_date': latest_signal['date'].strftime('%Y-%m-%d'),
            'latest_signal_price': latest_signal['price'],
            'signal_conditions': latest_signal['conditions'],
            'current_date': current_date.strftime('%Y-%m-%d'),
            'distance_to_signal': ((current_price - long_signal_price) / long_signal_price) * 100,
            'price_change_since_signal': ((current_price - latest_signal['price']) / latest_signal['price']) * 100,
            # 修正：加入趨勢反轉指標到返回結果
            'trend_reversal_confirmation': latest_signal.get('trend_reversal_confirmation', 0),
            'reversal_strength': latest_signal.get('reversal_strength', 0),
            'reversal_reliability': latest_signal.get('reversal_reliability', 0),
            'short_term_momentum_turn': latest_signal.get('short_term_momentum_turn', 0),
            'price_structure_reversal': latest_signal.get('price_structure_reversal', 0)
        })
        
        # === 新增：BPS 策略優化 ===
        bps_suggestions = self.bps_optimizer.suggest_bps_strikes(symbol)
        earnings_risk = self.bps_optimizer.check_earnings_risk(symbol)
        
        if bps_suggestions:
            base_result.update({
                'bps_short_put': bps_suggestions['short_put_strike'],
                'bps_long_put': bps_suggestions['long_put_strike'],
                'bps_expiry': bps_suggestions['suggested_expiry'], # 新增到期日欄位
                'bps_safety_margin': bps_suggestions['safety_margin_pct'],
                'expected_move_1sd': bps_suggestions['expected_move_1sd']
            })
        
        if earnings_risk:
            base_result.update({
                'earnings_date': earnings_risk.get('earnings_date'),
                'days_to_earnings': earnings_risk.get('days_to_earnings'),
                'earnings_high_risk': earnings_risk.get('high_risk')
            })
            if earnings_risk.get('high_risk'):
                base_result['confidence_factors'].append(f"⚠️ 財報風險：{earnings_risk['days_to_earnings']}天後發布財報")

        return base_result
    
    def analyze_specific_stocks(self, symbols):
        results = []
        print(f"開始對指定的 {len(symbols)} 支股票進行單獨分析...")
        for i, symbol in enumerate(symbols):
            print(f"   正在分析 {symbol} ({i+1}/{len(self.stocks)})...")
            result = self.analyze_stock(symbol)
            if result:
                results.append(result)
            time.sleep(1)
        print("指定股票分析完成")
        return results
    
    def analyze_watchlist(self):
        results = []
        
        print(f"開始分析 {len(self.stocks)} 支股票...")
        
        for i, symbol in enumerate(self.stocks):
            print(f"   正在分析 {symbol} ({i+1}/{len(self.stocks)})...")
            result = self.analyze_stock(symbol)
            if result:
                results.append(result)
            time.sleep(1)
        
        print("正在整合分析結果...")
        
        # 轉換為DataFrame
        df_results = pd.DataFrame(results)
        
        # 計算綜合評分
        if not df_results.empty:
            print("正在計算綜合評分...")
            df_results['composite_score'] = 0.0
            
            # 有訊號的股票
            signal_stocks = df_results[df_results['long_days'].notna()].copy()
            if not signal_stocks.empty:
                # Long Days評分（天數越短分數越高）
                max_days = signal_stocks['long_days'].max()
                signal_stocks['long_days_score'] = (max_days - signal_stocks['long_days']) / max_days * 100
                
                # 距離Long Signal Price評分（距離越近分數越高）
                signal_stocks['distance_score'] = np.maximum(0, 100 - signal_stocks['distance_to_signal'])
                
                # 進場建議評分
                entry_scores = {
                    '強烈推薦進場': 100,
                    '建議進場': 80,
                    '觀望': 50,
                    '不建議進場': 20
                }
                signal_stocks['entry_score'] = signal_stocks['entry_opportunity'].map(entry_scores)
                
                # 綜合評分
                signal_stocks['composite_score'] = (
                    signal_stocks['long_days_score'] * 0.3 +
                    signal_stocks['distance_score'] * 0.3 +
                    signal_stocks['entry_score'] * 0.2 +
                    signal_stocks['confidence_score'] * 0.2
                )
                
                # 更新DataFrame
                for idx in signal_stocks.index:
                    df_results.loc[idx, 'composite_score'] = signal_stocks.loc[idx, 'composite_score']
            
            # 按綜合評分排序
            df_results = df_results.sort_values('composite_score', ascending=False, na_position='last')
        
        print("分析結果整合完成")
        return df_results
    
    def generate_report(self, results_df):
        print("\n" + "="*100)
        print("股票抄底機會分析報告 - 被低估、可抄底、短期趨勢反轉股票")
        print("="*100)
        print(f"分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"觀察池股票數量：{len(self.stocks)}")
        print("="*100)
        
        if results_df.empty:
            print("無法獲取任何股票數據")
            return
        
        # 顯示最佳抄底機會
        # 顯示最佳抄底機會
        print("最佳抄底機會（趨勢反轉確認）：")
        print("-" * 100)
        
        for _, row in results_df.iterrows():
            if pd.notna(row['long_days']):
                print(f"   {row['symbol']} ({row['name']})")
                print(f"   進場建議: {row['entry_opportunity']}")
                print(f"   信心度: {row['confidence_score']:.0f}/100 ({row['confidence_level']})")
                print(f"   當前價格: ${row['current_price']:.2f}")
                print(f"   抄底價位: ${row['long_signal_price']:.2f} (距離: {row['distance_to_signal']:.1f}%)")
                print(f"   RSI: {row['rsi']:.1f} | 成交量比率: {row['volume_ratio']:.2f}")
                
                # 修正：顯示關鍵趨勢反轉指標（確保與實際計算一致）
                reversal_indicators = []
                
                # 檢查是否有趨勢反轉相關欄位
                if 'trend_reversal_confirmation' in row and pd.notna(row['trend_reversal_confirmation']):
                    if row['trend_reversal_confirmation'] > 60:
                        reversal_indicators.append(f"趨勢反轉確認({row['trend_reversal_confirmation']:.0f}分)")
                    elif row['trend_reversal_confirmation'] > 40:
                        reversal_indicators.append(f"趨勢反轉初步({row['trend_reversal_confirmation']:.0f}分)")
                    else:
                        reversal_indicators.append(f"無趨勢反轉({row['trend_reversal_confirmation']:.0f}分)")
                
                if 'reversal_strength' in row and pd.notna(row['reversal_strength']):
                    if row['reversal_strength'] > 70:
                        reversal_indicators.append(f"反轉強勁({row['reversal_strength']:.0f}分)")
                    elif row['reversal_strength'] > 50:
                        reversal_indicators.append(f"反轉中等({row['reversal_strength']:.0f}分)")
                    else:
                        reversal_indicators.append(f"反轉不足({row['reversal_strength']:.0f}分)")
                
                if 'reversal_reliability' in row and pd.notna(row['reversal_reliability']):
                    if row['reversal_reliability'] > 70:
                        reversal_indicators.append(f"反轉可信({row['reversal_reliability']:.0f}分)")
                    elif row['reversal_reliability'] > 50:
                        reversal_indicators.append(f"反轉一般({row['reversal_reliability']:.0f}分)")
                    else:
                        reversal_indicators.append(f"反轉不可信({row['reversal_reliability']:.0f}分)")
                
                if 'short_term_momentum_turn' in row and pd.notna(row['short_term_momentum_turn']):
                    if row['short_term_momentum_turn'] > 60:
                        reversal_indicators.append(f"動能轉折({row['short_term_momentum_turn']:.0f}分)")
                    elif row['short_term_momentum_turn'] > 40:
                        reversal_indicators.append(f"動能初步({row['short_term_momentum_turn']:.0f}分)")
                    else:
                        reversal_indicators.append(f"動能未轉({row['short_term_momentum_turn']:.0f}分)")
                
                if 'price_structure_reversal' in row and pd.notna(row['price_structure_reversal']):
                    if row['price_structure_reversal'] > 50:
                        reversal_indicators.append(f"結構反轉({row['price_structure_reversal']:.0f}分)")
                    elif row['price_structure_reversal'] > 30:
                        reversal_indicators.append(f"結構初步({row['price_structure_reversal']:.0f}分)")
                    else:
                        reversal_indicators.append(f"結構未轉({row['price_structure_reversal']:.0f}分)")
                
                if reversal_indicators:
                    print(f"    趨勢反轉: {', '.join(reversal_indicators[:3])}")  # 只顯示前3個
                
                # 顯示信心度因素（重點顯示趨勢反轉相關）
                confidence_factors = row.get('confidence_factors', [])
                if confidence_factors:
                    # 優先顯示趨勢反轉相關因素
                    reversal_factors = [f for f in confidence_factors if "反轉" in f or "轉折" in f or "確認" in f]
                    risk_factors = [f for f in confidence_factors if "風險" in f or "超買" in f or "偏高" in f]
                    other_factors = [f for f in confidence_factors if f not in reversal_factors and f not in risk_factors]
                    
                    if reversal_factors:
                        print(f"    反轉確認: {', '.join(reversal_factors[:2])}")
                    if risk_factors:
                        print(f"    風險提醒: {', '.join(risk_factors[:2])}")
                    if other_factors:
                        print(f"    技術優勢: {', '.join(other_factors[:2])}")
                
                print()
        
        # 顯示無訊號但可能被低估的股票
        no_signals = results_df[results_df['long_days'].isna()]
        if not no_signals.empty:
            print("\n無多頭訊號但可能被低估的股票：")
            print("-" * 60)
            for _, row in no_signals.iterrows():
                print(f"   {row['symbol']} ({row['name']})")
                print(f"     當前價格: ${row['current_price']:.2f}")
                print(f"     抄底價位: ${row['long_signal_price']:.2f} (距離: {row['distance_to_signal']:.1f}%)")
                print(f"     RSI: {row['rsi']:.1f}")
                print()
        
        print("="*100)
    
    def save_csv(self, df, filename='integrated_stock_analysis.csv'):
        if not df.empty:
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f" 分析結果已儲存至 {filename}")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main():
    print("🚀 啟動綜合股票分析器...")
    
    # 自動更新股票清單 (Auto Update Watchlist)
    print("\n📦 檢查並更新股票監控清單...")
    try:
        import sys
        import os
        # 確保可以導入同目錄下的 update_watchlist.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
            
        import update_watchlist
        print("   正在執行 update_watchlist.py 以獲取最新熱門股票...")
        success = update_watchlist.run_update()
        if success:
            print("   ✅ 股票清單更新成功，將使用最新數據進行分析。")
        else:
            print("   ⚠️ 股票清單更新失敗 (可能因網絡或來源問題)，將使用現有清單繼續執行。")
    except Exception as e:
        print(f"   ⚠️ 執行自動更新時發生錯誤: {e}")
        print("   將跳過更新步驟，使用現有清單。")
    
    # 設定分析參數
    # 如果是週六或週日，查看下週數據
    analyzer = IntegratedStockAnalyzer()
    
    print("開始股票分析...")
    results = analyzer.analyze_watchlist()
    
    print("正在生成分析報告...")
    analyzer.generate_report(results)
    
    print("正在儲存分析結果...")
    analyzer.save_csv(results)
    
    analysis_result = {}
    # 新增：輸出 analysis_result.json 供前端使用
    if not results.empty:
        tz = pytz.timezone('Asia/Taipei')
        now = datetime.now(tz)
        analysis_result = {
            "timestamp": now.isoformat(),
            "analysis_date": now.strftime('%Y-%m-%d %H:%M:%S'),
            "total_stocks": len(analyzer.stocks),
            "analyzed_stocks": len(results),
            "result": results.to_dict('records')
        }
        
        # 使用統一的路徑管理器保存分析結果
        try:
            from backend.path_manager import get_analysis_path
            analysis_path = get_analysis_path()
            print(f"使用統一分析結果文件路徑: {analysis_path}")
        except ImportError:
            # 回退到根目錄
            if Path("/app").exists():
                analysis_path = Path("/app/analysis_result.json")
            else:
                analysis_path = Path("analysis_result.json")
            print(f"使用回退分析結果文件路徑: {analysis_path}")

        # 確保目錄存在
        try:
            analysis_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print(f"無法創建目錄 {analysis_path.parent}，使用現有路徑")

        # 保存分析結果
        try:
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False, cls=NpEncoder)
            print(f"✅ 分析結果已儲存至統一路徑: {analysis_path}")
        except (PermissionError, OSError) as e:
            print(f"❌ 無法寫入分析結果文件: {e}")
            # 嘗試寫入到當前目錄作為回退
            fallback_path = Path("analysis_result.json")
            try:
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result, f, indent=2, ensure_ascii=False, cls=NpEncoder)
                print(f"✅ 分析結果已儲存至回退路徑: {fallback_path}")
            except Exception as fallback_error:
                print(f"❌ 回退保存也失敗: {fallback_error}")

    # 新增：更新股票監控清單
    print("\n開始更新股票監控清單...")

    # 使用統一的路徑管理器
    try:
        from backend.path_manager import get_monitored_stocks_path
        MONITORED_STOCKS_PATH = get_monitored_stocks_path()
        print(f"使用統一監控股票文件路徑: {MONITORED_STOCKS_PATH}")
    except ImportError:
        # 回退到根目錄
        if Path("/app").exists():
            MONITORED_STOCKS_PATH = Path("/app/monitored_stocks.json")
        else:
            MONITORED_STOCKS_PATH = Path("monitored_stocks.json")
        print(f"使用回退監控股票文件路徑: {MONITORED_STOCKS_PATH}")
    
    try:
        with open(MONITORED_STOCKS_PATH, 'r', encoding='utf-8') as f:
            monitored_stocks = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        monitored_stocks = []
        
    monitored_symbols = {stock['symbol'] for stock in monitored_stocks}
    print(f"目前監控清單中有 {len(monitored_symbols)} 支股票。")

    stocks_to_add = []
    if analysis_result and 'result' in analysis_result:
        for stock in analysis_result['result']:
            composite_score = stock.get('composite_score')
            confidence_score = stock.get('confidence_score')
            symbol = stock.get('symbol')

            if composite_score is None or confidence_score is None or symbol is None:
                continue

            if composite_score >= 90 and confidence_score >= 80:
                if symbol not in monitored_symbols:
                    print(f"新增股票到監控清單: {symbol} (綜合評分: {composite_score:.2f}, 信心度: {confidence_score:.2f})")
                    
                    new_entry = {
                        "symbol": symbol,
                        "name": stock.get('name'),
                        "market": stock.get('market'),
                        "entry_date": datetime.now(pytz.timezone('Asia/Taipei')).strftime('%Y-%m-%d'),
                        "entry_price": stock.get('current_price'),
                        "entry_composite_score": composite_score,
                        "entry_confidence_score": confidence_score,
                        "entry_signal_conditions": stock.get('signal_conditions'),
                        "long_signal_price_at_entry": stock.get('long_signal_price'),
                        "quantity": 1 # 新增：預設數量為1
                    }
                    stocks_to_add.append(new_entry)
                    monitored_symbols.add(symbol)

    if stocks_to_add:
        monitored_stocks.extend(stocks_to_add)
        try:
            with open(MONITORED_STOCKS_PATH, 'w', encoding='utf-8') as f:
                json.dump(monitored_stocks, f, indent=2, ensure_ascii=False)
            print(f"✅ 成功新增 {len(stocks_to_add)} 支股票到監控清單。檔案已更新: {MONITORED_STOCKS_PATH}")
        except (PermissionError, OSError) as e:
            print(f"❌ 無法寫入監控股票文件: {e}")
            print(f"   文件路徑: {MONITORED_STOCKS_PATH}")
            print(f"   請檢查文件權限或磁盤空間")
        except Exception as e:
            print(f"❌ 寫入監控股票文件時發生未知錯誤: {e}")
    else:
        print("本次分析無新增符合條件的股票進入監控清單。")
    
    print("\n 整合分析完成！")

if __name__ == "__main__":
    main() 