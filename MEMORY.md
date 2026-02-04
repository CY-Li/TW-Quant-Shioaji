# MEMORY.md - Long Term Memory

## Lessons Learned

### 2026-02-02: Automation & Error Handling
- **Lesson**: When chaining scripts (e.g., a manager script calling a worker script via `subprocess`), **ALWAYS validate the worker script's output format** manually first.
- **Context**: `backtest_aggressive.py` printed debug text alongside JSON, causing `loop.py` to crash on `json.loads()`.
- **Action**: Ensure all worker scripts use strict output formatting (e.g., `json.dumps` only) and silence `print` statements unless debug mode is on. Implement `try-except` blocks around `json.loads` to catch and log parsing errors.

### 2026-02-03: Notification Optimization
- **LINE Quota Management**: Modified all cron jobs and monitoring scripts to stop sending proactive LINE messages. 
- **Logging Redirection**: 
    - `Pre-market Snapshot Analysis` now logs to `memory/YYYY-MM-DD.md` (isolated turn, no delivery).
    - `BullPS Profit Watchdog` now runs in an isolated session and logs to memory.
    - `intraday_monitor.py` now writes alerts to `memory/intraday_alerts.log` instead of sending LINE messages.
- **Action**: User must manually check memory files or logs for status updates. Critical alerts are still recorded but not pushed.

### 2026-02-03: TW Stock Automation & Backtesting
- **Backtest Optimization**: Fixed `MA200` KeyError and pathing issues in `portfolio_simulator_v7.py`. Lowered entry threshold to 65 for higher sensitivity.
- **Daily Scanner**: Created `tw_daily_scanner.py` using `yfinance` to bypass Shioaji production data restrictions. This allows intraday/daily signal generation for TW market.
- **Connection Verified**: Shioaji API connection successfully tested in simulation environment.
- **Immediate Focus**: Transitioning from pure backtesting to "Live Signal & Paper Trading" for Taiwan stocks.

### 2026-02-03: BPS Recommendations (Live Scan)
- **Scanned**: 30 Core Watchlist stocks.
- **Top Picks**:
    1. **JPM** (Price $314.53): BPS 290/285, Exp 2026-03-13. Credit ~$0.79. (Score 70: RSI Bounce, Support).
    2. **V** (Price $334.95): BPS 310/305, Exp 2026-03-13. Credit ~$0.64. (Score 50: MACD Turn).
    3. **MSFT** (Price $415.13): BPS 380/375, Exp 2026-03-13. Credit ~$0.31. (Score 50: Low Volatility Play).
- **Note**: NFLX also flagged ($82) but excluded for clarity (possible split/data anomaly).

### 2026-02-04: System Continuity & Brain Implementation
- **Issue**: Agent consistency in project logic understanding.
- **Action**: Created `PROJECT_BRAIN.md` at the workspace root. This file serves as a "context anchor" that maps out the core logic of `integrated_stock_analyzer.py`, `portfolio_manager.py`, and `bps_optimizer.py`.
- **Constraint**: Added a mandatory instruction to read `PROJECT_BRAIN.md` at the start of any deep analysis session.
- **Knowledge Base**: Recorded the "Reasoning Erosion Model" (0.8 threshold), "1SD Expected Move" logic, and "50% Profit Target" as immutable project truths.

### 2026-02-04: TW-Quant V9 Sector Mission
- **Task**: Run V9 strategy backtest on three specific sectors (Semiconductors, Electronic Components, and Machinery/Biotech).
- **V9 Logic**: Entry requires TWII RSI >= 40, Volume > 1.3x MA20, and Analyzer Score >= 65.
- **Priority**: Maintain consistency between the simulator and the daily scanner.
- **Status**: Mission data initialized in `memory/2026-02-04.md`.

### 2026-02-01: Intraday Monitor
- **Lesson**: `yfinance` data can be delayed or missing strikes.
- **Action**: Implement "Nearest Strike" logic when exact strikes are unavailable. Always handle `data.empty` checks for market holidays/closed hours.
