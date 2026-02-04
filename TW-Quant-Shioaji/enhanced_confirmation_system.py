import pandas as pd
import numpy as np

class EnhancedConfirmationSystem:
    """
    100% 靈魂版增強確認系統
    整合多指標、成交量分佈與趨勢強度
    """
    def __init__(self):
        pass

    def calculate_comprehensive_confirmation(self, df_subset):
        """
        計算綜合確認分數 (0-100)
        """
        if len(df_subset) < 2:
            return {"total_score": 0, "details": {}}

        row = df_subset.iloc[-1]
        prev_row = df_subset.iloc[-2]
        
        score = 0
        details = {}

        # 1. 趨勢強度 (30分)
        trend_score = 0
        if 'MA20' in row and row['Close'] > row['MA20']: trend_score += 10
        if 'MA20' in row and 'MA60' in row and row['MA20'] > row['MA60']: trend_score += 10
        if 'MA60' in row and 'MA200' in row and not pd.isna(row['MA200']):
            if row['MA60'] > row['MA200']: trend_score += 10
        score += trend_score
        details['trend'] = trend_score

        # 2. 動能確認 (30分)
        momentum_score = 0
        # RSI 轉折或低位回升
        if row['RSI'] < 30: 
            momentum_score += 15 # 超賣反彈預期
        elif row['RSI'] > prev_row['RSI'] and row['RSI'] < 60:
            momentum_score += 10 # 強勢上攻
            
        # MACD 柱狀體翻正或縮減
        if row['MACD_Histogram'] > 0 and prev_row['MACD_Histogram'] <= 0:
            momentum_score += 15 # 金叉
        elif row['MACD_Histogram'] > prev_row['MACD_Histogram']:
            momentum_score += 5  # 動能增強
        
        score += momentum_score
        details['momentum'] = momentum_score

        # 3. 成交量確認 (20分)
        vol_score = 0
        if row['Volume_Ratio'] > 2.0:
            vol_score += 20 # 爆量突破
        elif row['Volume_Ratio'] > 1.2:
            vol_score += 10 # 溫和放量
        score += vol_score
        details['volume'] = vol_score

        # 4. 價格結構 (20分)
        struct_score = 0
        # 靠近布林下軌支撐
        bb_width = row['BB_Upper'] - row['BB_Lower']
        if bb_width > 0:
            bb_pos = (row['Close'] - row['BB_Lower']) / bb_width
            if bb_pos < 0.2:
                struct_score += 15 # 下軌支撐
            elif bb_pos < 0.5:
                struct_score += 5  # 中軌以下
        
        # 突破前高 (簡單代理)
        if row['Close'] > df_subset['Close'].iloc[-20:].max() * 0.98:
            struct_score += 5
            
        score += struct_score
        details['structure'] = struct_score

        return {
            "total_score": score,
            "details": details
        }
