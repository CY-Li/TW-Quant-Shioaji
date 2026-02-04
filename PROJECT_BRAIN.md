# BullPS-v3 專案邏輯核心大腦 (Pi 專屬記憶檔案)

這是為了確保 AI (Pi) 在每次對談中都能瞬間恢復對專案深度熟悉度的「外掛記憶」。包含核心算法、變數定義與交易哲學。

---

## 1. 核心分析引擎 (`integrated_stock_analyzer.py`)
- **進場分數 (Entry Score)**: 
    - 基礎 15 分，加減分制（RSI < 30 +3, RSI > 70 -5, ADX > 30 +2）。
    - 信心度計算：基於「多重支撐交集」，每多一條支撐線 (MA30, BB_Lower, 6M_Low) 距離 < 3%，加 15 分。
- **反轉引擎 (Trend Reversal)**:
    - `Trend_Reversal_Confirmation`: 綜合價格突破、RSI 反轉、MACD 轉強、成交量。
    - `Reversal_Reliability`: 五大指標一致性檢查（RSI, MACD, KD, SAR, OBV）。
- **SAR 調整**: 根據波動率動態調整 AF (高波動 0.015 / 低波動 0.025)。

## 2. 策略優化器 (`bps_optimizer.py`)
- **安全邊際 (1SD)**: 公式為 `Price * Vol * sqrt(Days/365)`。
- **平滑邏輯**: `floor_to_half` 確保履約價符合市場成交慣例。
- **財報過濾**: 14 天內財報列為 `high_risk`。

## 3. 持倉管理 (`portfolio_manager.py`)
- **理由侵蝕模型 (Erosion Model)**: 
    - 關鍵數據：`initial_analysis_snapshot` (進場理由快照)。
    - 邏輯：比對當前與進場理由，消失比例即為 Erosion Score。
- **懲罰機制 (Penalty Score)**:
    - MACD 死叉 (0.6), 跌破 MA20 (0.5), 動量減速 (0.5)。
    - 信心度 < 0.2 (20%) 觸發強制賣出建議。
- **智能 SAR 停損**: 
    - Day 0 保護：第一天虧損 < 2% 不得因 SAR 波動出場。
    - 獲利保護：獲利 > 20% 時，放寬 SAR 確認標準。

## 4. 監控與停利 (`profit_tracker.py`)
- **停利目標**: `PROFIT_TARGET_PCT = 0.50` (50% 權利金回收)。
- **插值估價**: `get_interpolated_price` 用於處理流動性不佳的期權報價。

---

## 5. 操作原則 (Rules for AI)
1. **不准瞎掰**: 任何關於標的的分析，必須先呼叫 `analyzer` 獲取真實分數。
2. **優先查表**: 針對持倉的建議，必須先讀取 `monitored_stocks.json` 與 `portfolio_manager` 的信心度。
3. **流動性警示**: 推薦 BPS 時，必須檢查 Bid/Ask Spread，如果 Spread > 權利金的 20%，必須發出警示。
