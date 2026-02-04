import shioaji as sj
import json
import os

# 配置資訊
API_KEY = "BVhtFhFdjJirnkxbey2PYMUxTBz3QNFnagjiViMCMV11"
SECRET_KEY = "fjPPAFN9FuYXzsgATtfbaoY52Y3nrx3tHpVMSak4U6g"
CERT_PATH = "/home/jimmy161688/.openclaw/workspace/TW-Quant-Shioaji/cert/Sinopac.pfx"
PERSON_ID = "F128319357"
CERT_PASS = "F128319357"

def final_login_test():
    api = sj.Shioaji()
    try:
        print(f"正在執行最終連線測試 (PersonID: {PERSON_ID})...")
        
        # 1. 登入
        api.login(api_key=API_KEY, secret_key=SECRET_KEY)
        print("✅ 第一階段：API 登入成功")
        
        # 2. 啟動憑證 (正式環境關鍵步驟)
        api.activate_ca(
            ca_path=CERT_PATH,
            ca_passwd=CERT_PASS,
            person_id=PERSON_ID
        )
        print("✅ 第二階段：憑證載入成功")
        
        # 3. 獲取帳號列表以驗證正式權限
        accounts = api.list_accounts()
        
        return {
            "status": "success",
            "accounts": [str(acc) for acc in accounts],
            "msg": "正式環境連線完全暢通！"
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }
    finally:
        # api.logout() # 保持登入以便後續操作，或測試完先登出
        pass

if __name__ == "__main__":
    result = final_login_test()
    print(json.dumps(result, indent=2, ensure_ascii=False))
