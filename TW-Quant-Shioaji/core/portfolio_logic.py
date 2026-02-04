import numpy as np
from datetime import datetime

def evaluate_smart_sar_exit(trade, current_analysis, current_dt=None):
    """
    智能SAR停損評估 (台股現貨版)
    結合多重確認機制，避免假突破
    """
    if current_dt is None:
        current_dt = datetime.now()
    current_price = current_analysis.get('current_price')
    current_sar = current_analysis.get('sar')
    entry_price = trade.get('entry_price')
    entry_date = trade.get('entry_date')

    if current_price is None or current_sar is None:
        return {'should_exit': False, 'reason': '數據不完整'}

    # 基本SAR信號
    basic_sar_triggered = current_price < current_sar

    if not basic_sar_triggered:
        return {'should_exit': False, 'reason': 'SAR未觸發'}

    # 計算持倉天數
    try:
        if isinstance(entry_date, str):
            entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
        else:
            entry_dt = entry_date
        holding_days = (current_dt - entry_dt).days
    except:
        holding_days = 0

    # 計算獲利幅度
    profit_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price else 0

    # 確認機制評分
    confirmation_score = 0
    required_confirmation = 3 # 基礎要求

    # 1. 獲利保護
    if profit_pct > 15: required_confirmation = 6
    elif profit_pct > 5: required_confirmation = 4
    else: required_confirmation = 2

    # 2. 技術指標確認
    current_rsi = current_analysis.get('rsi', 50)
    current_macd = current_analysis.get('macd', 0)
    
    if current_rsi > 70: confirmation_score += 2
    if current_macd < 0: confirmation_score += 2
    
    # 3. 跌破深度
    sar_penetration = ((current_sar - current_price) / current_price) * 100
    if sar_penetration > 1.5: confirmation_score += 2
    
    # Day 0/1 保護
    if holding_days <= 1 and profit_pct > -2.0:
        required_confirmation += 5

    should_exit = confirmation_score >= required_confirmation
    return {
        'should_exit': should_exit,
        'reason': f"SAR停損 (Score: {confirmation_score}/{required_confirmation})"
    }

def evaluate_exit_confidence(trade, latest_analysis):
    """
    評估單一持倉的出場信心度 (台股現貨版)
    """
    initial_snapshot = trade.get('initial_analysis_snapshot', {})
    entry_reasons = set(initial_snapshot.get('confidence_factors', []))
    current_factors = set(latest_analysis.get('confidence_factors', []))

    if not entry_reasons:
        return {"exit_confidence": 0.0, "erosion_score": 0.0}

    # 1. 理由侵蝕分數 (加權版)
    # 簡單起見，這裡先用全攤平，若要優化可引入時間框架權重
    disappeared_reasons = [r for r in entry_reasons if r not in current_factors]
    erosion_score = len(disappeared_reasons) / len(entry_reasons)
    
    # 2. 風險懲罰 (MACD死叉, 價格跌破均線等)
    penalty_score = 0
    if "MACD死叉" in current_factors: penalty_score += 0.5
    if "價格跌破20日均線" in current_factors: penalty_score += 0.4
    
    exit_confidence = (erosion_score * 0.7) + (min(penalty_score, 1.0) * 0.3)
    return {
        "exit_confidence": round(exit_confidence, 2),
        "erosion_score": round(erosion_score, 2)
    }
