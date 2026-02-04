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
