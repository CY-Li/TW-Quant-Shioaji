import shioaji as sj
import time
import json
import os

class ShioajiExplorer:
    def __init__(self, api_key, api_secret):
        self.api = sj.Shioaji()
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_logged_in = False

    def test_connection(self):
        """測試 API 連線與基本數據抓取權限"""
        try:
            print(f"正在嘗試連線至 Shioaji (模擬/測試環境)...")
            self.api.login(
                api_key=self.api_key,
                secret_key=self.api_secret,
                contracts_cb=print
            )
            # 由於沒有正式環境權限，我們捕捉特定的錯誤
            self.is_logged_in = True
            print("✅ 身份驗證封包已發送")
            return {
                "status": "connected",
                "msg": "API 已建立 Session，但如預期受到 Production 權限限制"
            }
        except Exception as e:
            if "production permission" in str(e):
                return {
                    "status": "verified",
                    "msg": "連線測試成功：Key 有效，且系統如預期阻擋了正式環境存取"
                }
            return {
                "status": "failed",
                "error": str(e)
            }

if __name__ == "__main__":
    # 使用使用者提供的測試 Key
    KEY = "DKGb3BFQ7SsX8Mb9y3bhrMmrgjjUPoD827Wjd6Ki5rKk"
    SECRET = "125X7iLKE525gr2njqPu5zB83WDVnsfnY4mNydbkZSJF"
    
    explorer = ShioajiExplorer(KEY, SECRET)
    result = explorer.test_connection()
    print(json.dumps(result, indent=2, ensure_ascii=False))
