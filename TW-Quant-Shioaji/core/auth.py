import shioaji as sj
import time

class ShioajiClient:
    def __init__(self, api_key=None, api_secret=None, cert_path=None, cert_pass=None):
        self.api = sj.Shioaji()
        self.api_key = api_key
        self.api_secret = api_secret
        self.cert_path = cert_path
        self.cert_pass = cert_pass
        self.is_logged_in = False

    def login(self):
        """執行 API 登入與憑證載入"""
        try:
            print(f"正在登入永豐金 API (Key: {self.api_key[:5]}***)...")
            self.api.login(
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            
            if self.cert_path and self.cert_pass:
                print("正在載入憑證...")
                self.api.activate_ca(
                    ca_path=self.cert_path,
                    ca_passwd=self.cert_pass,
                    person_id=self.api_key # 通常 api_key 就是身份證字號
                )
            
            self.is_logged_in = True
            print("✅ Shioaji 登入成功！")
            return True
        except Exception as e:
            print(f"❌ 登入失敗: {e}")
            return False

    def get_account_status(self):
        if self.is_logged_in:
            return self.api.list_accounts()
        return None

if __name__ == "__main__":
    # 此部分僅供測試，正式使用需從環境變數或加密文件讀取
    pass
