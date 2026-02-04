import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TWAnalyzer:
    def __init__(self):
        # 移植自美股的核心權重
        self.confirmation_weights = {
            'technical_sync': 0.40,
            'price_action': 0.30,
            'volume_confirmation': 0.30
        }

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def analyze_indicators(self, df):
        """核心指標計算"""
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['prev_rsi'] = df['RSI'].shift(1) # 新增：計算前一根 RSI
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Volume Ratio (對標 5 日均量)
        df['Vol_Avg'] = df['Volume'].rolling(20).mean() # 假設分時線
        df['Vol_Ratio'] = df['Volume'] / df['Vol_Avg']
        
        return df

    def get_confidence_score(self, df):
        """計算台股信心評分 (0-100)"""
        if len(df) < 60: return 0, []
        
        df = self.analyze_indicators(df)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        score = 0
        factors = []
        
        # 1. 趨勢確認 (MA 系統)
        if last['Close'] > last['MA200']:
            score += 20
            factors.append("長期趨勢看多 (>200MA)")
        if last['MA5'] > last['MA20']:
            score += 15
            factors.append("短均線黃金交叉")
            
        # 2. 動能與反轉 (RSI/MACD)
        if last['RSI'] < 35:
            score += 25
            factors.append("RSI 超賣區 (抄底訊號)")
        elif last['RSI'] > last['prev_rsi'] if 'prev_rsi' in last else True and last['RSI'] < 50:
            score += 10
            factors.append("RSI 低位翻揚")
            
        if last['MACD_Hist'] > 0 and prev['MACD_Hist'] <= 0:
            score += 20
            factors.append("MACD 柱狀圖轉正")
            
        # 3. 量能確認
        if last['Vol_Ratio'] > 1.5:
            score += 10
            factors.append("爆量確認 (主力進場)")
            
        return min(score, 100), factors

if __name__ == "__main__":
    # 測試代碼略
    pass
