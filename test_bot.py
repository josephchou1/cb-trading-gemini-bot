import os
import time
import json
import re
import threading
import io
from pathlib import Path
from datetime import date, datetime, time as dtime
from contextlib import contextmanager
from http.server import HTTPServer, BaseHTTPRequestHandler
import psycopg2
from psycopg2 import pool
import requests
import twstock
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DB_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FUGLE_TOKEN = os.getenv("FUGLE_TOKEN")

required_env = {
    "TELEGRAM_BOT_TOKEN": BOT_TOKEN,
    "TELEGRAM_CHAT_ID": CHAT_ID,
    "DATABASE_URL": DB_URL,
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "FUGLE_TOKEN": FUGLE_TOKEN,
}
missing_env = [name for name, value in required_env.items() if not value]
if missing_env:
    raise RuntimeError("缺少必要環境變數：" + ", ".join(missing_env))

client = genai.Client(api_key=GEMINI_API_KEY)

db_pool = None
try:
    db_pool = psycopg2.pool.ThreadedConnectionPool(1, 10, DB_URL, sslmode="require")
    print("✅ 資料庫連線池 (ThreadedConnectionPool) 初始化成功")
except Exception as e:
    raise RuntimeError("資料庫連線池初始化失敗，請檢查 DATABASE_URL 與雲端資料庫網路設定。") from e

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = db_pool.getconn()
        yield conn
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn and db_pool:
            db_pool.putconn(conn)

DEFAULT_STOP_LOSS_PERCENT = 3.0
DEFAULT_TAKE_PROFIT_PERCENT = 11.0
DEFAULT_TRAILING_STOP_PERCENT = 6.0
DEFAULT_TRAILING_ACTIVATION_PERCENT = 3.0
DEFAULT_WARNING_BUFFER_PERCENT = 1.0

def get_stock_info(identifier: str):
    if not identifier:
        return None, None
    identifier = str(identifier).strip()
    
    if identifier in twstock.codes:
        return identifier, twstock.codes[identifier].name
    
    if len(identifier) == 5 and identifier.isdigit():
        base_code = identifier[:4]
        cb_suffix = identifier[4]
        suffix_map = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九', '0': '十'}
        name_suffix = suffix_map.get(cb_suffix, 'CB')
        if base_code in twstock.codes:
            cb_name = f"{twstock.codes[base_code].name}{name_suffix}"
            return identifier, cb_name
        return identifier, f"可轉債{identifier}"

    for code, info in twstock.codes.items():
        if identifier in info.name or info.name in identifier:
            return code, info.name
            
    return identifier, identifier

def is_market_open() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    market_start = dtime(9, 0, 0)
    market_end = dtime(13, 45, 0)
    return market_start <= now.time() <= market_end

def is_market_closing_time() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    return dtime(13, 40, 0) <= now.time() <= dtime(13, 45, 0)

def init_db_schema():
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                ALTER TABLE position_lots 
                ADD COLUMN IF NOT EXISTS warning_buffer_percent NUMERIC DEFAULT 1.0,
                ADD COLUMN IF NOT EXISTS warning_sl_price NUMERIC,
                ADD COLUMN IF NOT EXISTS warning_tp_price NUMERIC,
                ADD COLUMN IF NOT EXISTS ts_activation_percent NUMERIC DEFAULT 3.0;
                
                CREATE TABLE IF NOT EXISTS daily_reports (
                    report_date DATE PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS trade_history (
                    history_id SERIAL PRIMARY KEY,
                    lot_id INT,
                    stock_code VARCHAR(20),
                    stock_name VARCHAR(50),
                    action_type VARCHAR(20),
                    quantity INT,
                    price NUMERIC,
                    realized_pnl NUMERIC,
                    realized_pnl_pct NUMERIC,
                    exit_reason VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            cur.close()
    except Exception as e:
        print(f"⚠️ 初始化 schema 提示：{e}")

def split_telegram_text(text: str, max_chars: int = 3000):
    """Split long plain-text messages into Telegram-safe chunks at line boundaries."""
    text = str(text)
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = ""
    for line in text.splitlines(keepends=True):
        while len(line) > max_chars:
            if current:
                chunks.append(current.rstrip())
                current = ""
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        if len(current) + len(line) > max_chars:
            if current:
                chunks.append(current.rstrip())
            current = line
        else:
            current += line
    if current:
        chunks.append(current.rstrip())
    return chunks or [""]

def send_telegram(text: str, silent: bool = False, reply_markup=None, chat_id=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    headers = {"User-Agent": "Mozilla/5.0", "Connection": "close"}
    chunks = split_telegram_text(text)
    last_data = None
    for index, chunk in enumerate(chunks):
        payload = {
            "chat_id": chat_id or CHAT_ID,
            "text": chunk,
            "disable_notification": silent
        }
        if reply_markup and index == len(chunks) - 1:
            payload["reply_markup"] = reply_markup

        delivered = False
        for _ in range(3):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=8)
                data = resp.json()
                if resp.status_code == 200 and data.get("ok"):
                    last_data = data
                    delivered = True
                    break
                description = data.get("description", "未知錯誤")
                print(f"❌ Telegram sendMessage HTTP {resp.status_code}: {description}", flush=True)
            except Exception as e:
                safe_error = str(e).replace(BOT_TOKEN, "<redacted>")
                print(f"❌ Telegram sendMessage 連線錯誤：{type(e).__name__}: {safe_error}", flush=True)
                time.sleep(0.5)
        if not delivered:
            return None
    return last_data

def get_market_quote(code: str):
    code, name = get_stock_info(code)
    if not code:
        return None, None, None, None

    if len(code) == 5 and code.isdigit():
        try:
            url_fugle_cb = f"https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/{code}"
            headers = {"X-API-KEY": FUGLE_TOKEN, "Connection": "close"}
            resp_f = requests.get(url_fugle_cb, headers=headers, timeout=5)
            if resp_f.status_code == 200:
                d = resp_f.json()
                price = d.get("closePrice") or d.get("lastUpdatedPrice") or (d.get("trade", {}).get("price") if isinstance(d.get("trade"), dict) else None)
                if price:
                    return name, float(price), float(d.get("highPrice", price)), float(d.get("lowPrice", price))
        except Exception:
            pass

        for suffix in [".TWO", ".TW"]:
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}{suffix}"
                headers = {"User-Agent": "Mozilla/5.0", "Connection": "close"}
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    result = data.get("chart", {}).get("result")
                    if result:
                        meta = result[0]["meta"]
                        price = meta.get("regularMarketPrice")
                        if price:
                            return name, float(price), float(meta.get("regularMarketDayHigh", price)), float(meta.get("regularMarketDayLow", price))
            except Exception:
                pass

    if is_market_open() and len(code) == 4:
        try:
            url = f"https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/{code}"
            headers = {"X-API-KEY": FUGLE_TOKEN, "Connection": "close"}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                d = resp.json()
                price = (
                    d.get("closePrice") 
                    or d.get("lastUpdatedPrice") 
                    or d.get("avgPrice") 
                    or (d.get("trade", {}).get("price") if isinstance(d.get("trade"), dict) else None)
                )
                high = d.get("highPrice")
                low = d.get("lowPrice")
                if price:
                    return name, float(price), float(high) if high else float(price), float(low) if low else float(price)
        except Exception:
            pass

    for suffix in [".TW", ".TWO"]:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}{suffix}"
            headers = {"User-Agent": "Mozilla/5.0", "Connection": "close"}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                result = data.get("chart", {}).get("result")
                if result:
                    meta = result[0]["meta"]
                    price = meta.get("regularMarketPrice")
                    high = meta.get("regularMarketDayHigh")
                    low = meta.get("regularMarketDayLow")
                    if price:
                        return name, float(price), float(high) if high else float(price), float(low) if low else float(price)
        except Exception:
            pass

    return name, None, None, None

def get_portfolio_summary_text(filter_keyword: str = None, sort_by_profit: bool = False) -> str:
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            today = date.today()

            cur.execute("""
                SELECT lot_id, stock_code, stock_name, buy_price, quantity, 
                       stop_loss_price, take_profit_price, stop_loss_percent, take_profit_percent, 
                       trailing_stop_percent, warning_buffer_percent, warning_sl_price, warning_tp_price,
                       ts_activation_percent
                FROM position_lots
                WHERE monitoring_status = 'MONITORING'
                ORDER BY lot_id ASC;
            """)
            lots = cur.fetchall()

            if not lots:
                cur.close()
                return "📋 【最新持倉狀況】\n目前已無任何監控中的持倉批次。"

            cur.execute("""
                SELECT lot_id, event_type 
                FROM notification_records 
                WHERE trade_date = %s;
            """, (today,))
            notif_rows = cur.fetchall()
            cur.close()

        triggered_map = {}
        for r_lot_id, r_event in notif_rows:
            if r_lot_id not in triggered_map or triggered_map[r_lot_id].startswith("APPROACHING"):
                triggered_map[r_lot_id] = r_event

        parsed_items = []
        for r in lots:
            lot_id = r[0]
            code = r[1]
            _, name = get_stock_info(code)
            buy_price = float(r[3])
            qty = r[4]
            sl_price = float(r[5])
            tp_price = float(r[6])
            sl_pct = abs(float(r[7]))
            tp_pct = abs(float(r[8]))
            ts_p = float(r[9]) if r[9] else DEFAULT_TRAILING_STOP_PERCENT
            wb_p = float(r[10]) if r[10] is not None else DEFAULT_WARNING_BUFFER_PERCENT
            w_sl_p = float(r[11]) if r[11] is not None else None
            w_tp_p = float(r[12]) if r[12] is not None else None
            ts_act_p = float(r[13]) if r[13] is not None else DEFAULT_TRAILING_ACTIVATION_PERCENT

            if filter_keyword:
                kw = filter_keyword.strip().lower()
                matched_kw = (kw in code.lower() or kw in name.lower() or kw == str(lot_id))
                if not matched_kw:
                    continue

            _, cur_price, _, _ = get_market_quote(code)
            if cur_price is not None:
                cur_price = float(cur_price)
                diff_val = cur_price - buy_price
                diff_pct = (diff_val / buy_price) * 100
                sign = "+" if diff_val >= 0 else ""
                price_line = f"  最新成交：{cur_price:.1f} 元 ({sign}{diff_pct:.2f}%)"
            else:
                cur_price = buy_price
                diff_pct = 0.0
                price_line = "  最新成交：查無即時行情"

            event_type = triggered_map.get(lot_id)
            icon = "🔹"
            status_line = ""

            if event_type == 'STOP_LOSS':
                icon = "🔴"
                status_line = "\n\n  今日狀態：🚨 盤中已觸發停損防線，請儘速處理！"
            elif event_type == 'TAKE_PROFIT':
                icon = "🟡"
                status_line = "\n\n  今日狀態：🎉 盤中已觸發停利目標，可分批入袋！"
            elif event_type == 'TRAILING_STOP':
                icon = "🟠"
                status_line = "\n\n  今日狀態：⚠️ 盤中已觸發移動停利，請獲利入袋！"
            elif event_type == 'APPROACHING_STOP_LOSS':
                icon = "🔸"
                status_line = "\n\n  今日狀態：⚠️ 盤中逼近停損防線"
            elif event_type == 'APPROACHING_TAKE_PROFIT':
                icon = "🔸"
                status_line = "\n\n  今日狀態：🎯 盤中逼近停利目標"

            warn_info = []
            if w_sl_p:
                warn_info.append(f"跌至{w_sl_p:.1f}元")
            if w_tp_p:
                warn_info.append(f"漲至{w_tp_p:.1f}元")
            if not warn_info:
                warn_line = f"  預警通知：{wb_p:.1f}%"
            else:
                warn_line = f"  預警通知：{' / '.join(warn_info)}"

            item_text = (
                f"{icon} 【第 {lot_id} 筆】| {name} ({code})\n"
                f"  買入成本：{buy_price:.1f} 元 ({qty}張)\n"
                f"{price_line}\n"
                f"  停損防線：{sl_price:.1f} 元 (-{sl_pct:.1f}%)\n"
                f"  停利目標：{tp_price:.1f} 元 (+{tp_pct:.1f}%)\n"
                f"  移動停利：{ts_p:.1f}%（獲利達 +{ts_act_p:.1f}% 啟動）\n"
                f"{warn_line}"
                f"{status_line}"
            )
            parsed_items.append({"text": item_text, "profit_pct": diff_pct})

        if not parsed_items:
            return f"📋 【最新持倉監控列表】\n\n找不到符合「{filter_keyword}」的監控中持倉。"

        if sort_by_profit:
            parsed_items.sort(key=lambda x: x["profit_pct"], reverse=True)

        final_texts = [item["text"] for item in parsed_items]
        header = f"📋 【最新持倉監控列表】{'（已依獲利排序）' if sort_by_profit else ''}"
        return header + "\n\n" + "\n----------------------------\n".join(final_texts)

    except Exception as e:
        return f"❌ 讀取最新庫存失敗：{e}"

def get_show_portfolio_markup():
    return {
        "inline_keyboard": [
            [{"text": "📋 顯示全部庫存清單", "callback_data": "SHOW_ALL_PORTFOLIO"}]
        ]
    }

def clean_input_text(text: str) -> str:
    t = text.strip()
    t = re.sub(r'^[地弟低]\s*(\d+|[一二兩三四五六七八九十]+)', r'第\1', t)
    t = re.sub(r'(?<=\s)[地弟低]\s*(\d+|[一二兩三四五六七八九十]+)', r'第\1', t)

    for mis in ['玉井', '玉景', '預井', '預鏡', '玉鏡', '預緊', '於警', '預景', '魚警']:
        t = t.replace(mis, '預警')

    for mis in ['移動挺立', '移動停地', '一動停利', '移動停力', '一動挺立']:
        t = t.replace(mis, '移動停利')
    for mis in ['挺立', '挺利', '廷立', '停力', '停一', '挺力', '停例', '聽力', '停立']:
        t = t.replace(mis, '停利')
    for mis in ['停筍', '聽損', '廷損', '停准', '停省']:
        t = t.replace(mis, '停損')

    t = re.sub(r'[—–－\-]+', ' ', t)
    t = t.replace('，', ' ').replace('。', ' ')
    return re.sub(r'\s+', ' ', t)

def extract_trade_intent(user_text: str):
    clean_text = clean_input_text(user_text)
    
    prompt = (
        "你是一個台灣股市 AI 交易管家助理。請分析使用者的這句話，並嚴格以純 JSON 格式回傳結果（絕對不要包含 ```json 或任何 markdown 標記，只要輸出大括號 {} 內部的 JSON）。\n\n"
        "支援的 action 類型：\n"
        "- \"ADD_LOT\": 買進建倉。需包含欄位: \"stock_code\" (4碼股票代號或5碼可轉債代號，請根據股票中文名稱自動查出正確代號，例如 臺企銀=2834, 台積電=2330, 聯發科=2454 等), \"stock_name\" (股票名稱), \"buy_price\" (買進價格，浮點數), \"quantity\" (張數，整數，若未寫預設為 1)\n"
        "- \"SELL_LOT\": 賣出/平倉。需包含欄位: \"stock_code\", \"sell_price\", \"quantity\"\n"
        "- \"QUERY_PORTFOLIO_FILTERED\": 查詢庫存。需包含欄位: \"filter_keyword\" (過濾關鍵字), \"sort_by_profit\" (布林值，是否依獲利排序)\n"
        "- \"GET_PRICE\": 查現價。需包含欄位: \"stock_code\"\n"
        "- \"QUERY_PORTFOLIO\": 查詢全部監控中的庫存。\n"
        "- \"SET_WARNING_PRICE\": 設定持倉預警價格。需包含 lot_id 或 stock_code，以及 warning_sl_price、warning_tp_price 或 general_price。\n"
        "- \"SET_WARNING_BUFFER\": 設定預警緩衝百分比。需包含 buffer_percent，可選 lot_id、stock_code、is_global。\n"
        "- \"DELETE_LOT\": 作廢持倉。需包含 lot_ids (整數列表)。\n"
        "- \"SELL_MULTI_LOTS\": 按持倉編號賣出。需包含 lot_ids (整數列表)、quantity，可選 sell_price。\n"
        "- \"SELL_BY_LOT_ID\": 按單一持倉編號賣出。需包含 lot_id、quantity，可選 sell_price。\n"
        "- \"UPDATE_SETTINGS\": 修改指定持倉風控。需包含 lot_id 或 stock_code，以及 new_tp_val/new_tp_is_pct、new_sl_val/new_sl_is_pct 或 new_ts_percent。\n"
        "- \"UPDATE_GLOBAL_SETTINGS_EXT\": 修改全域風控。需包含 new_sl、new_tp、new_ts，及 new_tp_is_pct (如適用) 和 raw_text。\n"
        "- \"UNKNOWN\": 無法辨識。\n\n"
        f"請解析這句使用者訊息：\"{clean_text}\""
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        result = json.loads(response.text.strip())
        
        if result.get("action") == "ADD_LOT":
            raw_code = result.get("stock_code")
            c_code, c_name = get_stock_info(raw_code)
            result["stock_code"] = c_code
            result["stock_name"] = c_name
            
        return result
    except Exception as e:
        print(f"❌ Gemini 智慧意圖解析錯誤: {e}")
        return {"action": "UNKNOWN"}

def handle_screenshot_image(photo_file_id: str):
    try:
        file_info_url = f"https://api.telegram.org/bot{BOT_TOKEN}/getFile"
        resp = requests.get(file_info_url, params={"file_id": photo_file_id}, timeout=5).json()
        if not resp.get("ok"):
            send_telegram("❌ 無法取得圖片檔案資訊。")
            return

        file_path = resp["result"]["file_path"]
        download_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
        
        img_resp = requests.get(download_url, timeout=10)
        image = Image.open(io.BytesIO(img_resp.content))

        prompt = """
        這是一張台灣股市或可轉債的券商 App 庫存明細截圖。
        請幫我找出畫面中所有的持股資料，並嚴格以 JSON 格式回傳一個列表（List），不要包含額外文字。
        每個物件包含以下欄位：
        - "stock_id": 股票或可轉債的 4 碼或 5 碼代號（例如 "2330", "12331"）
        - "shares": 持有股數（整數，系統預設以 1 張為 1 單位，若無法判定張數請回傳 1）
        - "cost": 平均成本價（浮點數，若無則填 0.0）
        格式範例：
        [
          {"stock_id": "2330", "shares": 5, "cost": 600.0},
          {"stock_id": "2881", "shares": 2, "cost": 50.5}
        ]
        """

        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=[image, prompt]
        )
        
        text_response = response.text.strip()
        json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
        if json_match:
            holdings = json.loads(json_match.group(0))
            for item in holdings:
                s_id = str(item.get("stock_id"))
                qty = int(item.get("shares", 1))
                cost = float(item.get("cost", 0.0))
                
                if cost > 0:
                    add_data = {
                        "stock_code": s_id,
                        "buy_price": cost,
                        "quantity": qty
                    }
                    handle_add_lot(add_data)
                else:
                    _, cur_p, _, _ = get_market_quote(s_id)
                    add_data = {
                        "stock_code": s_id,
                        "buy_price": cur_p or 100.0,
                        "quantity": qty
                    }
                    handle_add_lot(add_data)
        else:
            send_telegram(f"⚠️ 無法從截圖中解析出庫存明細格式。\nAI 回應：{text_response}")

    except Exception as e:
        send_telegram(f"❌ 圖片辨識入庫失敗：{e}")

def handle_update_global_settings_ext(data: dict):
    global DEFAULT_STOP_LOSS_PERCENT, DEFAULT_TAKE_PROFIT_PERCENT, DEFAULT_TRAILING_STOP_PERCENT
    new_sl = data.get("new_sl")
    new_tp = data.get("new_tp")
    new_tp_is_pct = data.get("new_tp_is_pct", False)
    new_ts = data.get("new_ts")

    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            if new_sl is not None and "停損" in data.get("raw_text", ""):
                DEFAULT_STOP_LOSS_PERCENT = new_sl
                cur.execute("""
                    UPDATE position_lots 
                    SET stop_loss_percent = %s, 
                        stop_loss_price = ROUND(buy_price * (1 - %s / 100.0), 2)
                    WHERE monitoring_status = 'MONITORING';
                """, (new_sl, new_sl))
                row_count = cur.rowcount
                conn.commit()
                cur.close()
                send_telegram(
                    f"⚙️ 【全域停損防線更新成功】\n\n• 新全域預設停損：-{new_sl:.1f}%\n• 已同步套用：共 {row_count} 筆現有庫存",
                    reply_markup=get_show_portfolio_markup()
                )
                return

            elif new_tp is not None and "停利" in data.get("raw_text", ""):
                if new_tp_is_pct:
                    DEFAULT_TAKE_PROFIT_PERCENT = new_tp
                    cur.execute("""
                        UPDATE position_lots 
                        SET take_profit_percent = %s, 
                            take_profit_price = ROUND(buy_price * (1 + %s / 100.0), 2)
                        WHERE monitoring_status = 'MONITORING';
                    """, (new_tp, new_tp))
                else:
                    cur.execute("""
                        UPDATE position_lots 
                        SET take_profit_price = %s,
                            take_profit_percent = ROUND(((%s - buy_price) / buy_price) * 100, 2)
                        WHERE monitoring_status = 'MONITORING';
                    """, (new_tp, new_tp))
                row_count = cur.rowcount
                conn.commit()
                cur.close()
                send_telegram(
                    f"⚙️ 【全域停利目標更新成功】\n\n• 新全域預設停利：+{new_tp:.1f}%\n• 已同步套用：共 {row_count} 筆現有庫存",
                    reply_markup=get_show_portfolio_markup()
                )
                return

            elif new_ts is not None and "移動停利" in data.get("raw_text", ""):
                DEFAULT_TRAILING_STOP_PERCENT = new_ts
                cur.execute("""
                    UPDATE position_lots 
                    SET trailing_stop_percent = %s 
                    WHERE monitoring_status = 'MONITORING';
                """, (new_ts,))
                row_count = cur.rowcount
                conn.commit()
                cur.close()
                send_telegram(
                    f"⚙️ 【全域移動停利更新成功】\n\n• 新全域移動停利：{new_ts:.1f}%\n• 已同步套用：共 {row_count} 筆現有庫存",
                    reply_markup=get_show_portfolio_markup()
                )
                return

        send_telegram("⚠️ 未能明確辨識要更新的全域設定數值。")
    except Exception as e:
        send_telegram(f"❌ 更新全域設定失敗：{e}")

def handle_set_warning_price(data: dict):
    lot_id = data.get("lot_id")
    code = data.get("stock_code")
    w_sl = data.get("warning_sl_price")
    w_tp = data.get("warning_tp_price")
    gen_p = data.get("general_price")

    try:
        with get_db_connection() as conn:
            cur = conn.cursor()

            if lot_id:
                cur.execute("SELECT lot_id, stock_code, stock_name, buy_price FROM position_lots WHERE lot_id = %s AND monitoring_status = 'MONITORING';", (lot_id,))
            elif code:
                cur.execute("SELECT lot_id, stock_code, stock_name, buy_price FROM position_lots WHERE stock_code = %s AND monitoring_status = 'MONITORING' ORDER BY lot_id DESC LIMIT 1;", (code,))
            else:
                send_telegram("⚠️ 請指明要設定預警價格的持倉編號或股票。")
                cur.close()
                return

            row = cur.fetchone()
            if not row:
                send_telegram("⚠️ 找不到監控中的對應持倉資料。")
                cur.close()
                return

            t_lot_id, t_code, _, buy_p = row
            _, t_name = get_stock_info(t_code)
            buy_val = float(buy_p)

            if gen_p is not None and w_sl is None and w_tp is None:
                if float(gen_p) < buy_val:
                    w_sl = float(gen_p)
                else:
                    w_tp = float(gen_p)

            update_fields = []
            params = []
            summary_txt = []

            if w_sl is not None:
                update_fields.append("warning_sl_price = %s")
                params.append(w_sl)
                summary_txt.append(f"• 停損預警價格：跌至 {w_sl:.1f} 元提醒")

            if w_tp is not None:
                update_fields.append("warning_tp_price = %s")
                params.append(w_tp)
                summary_txt.append(f"• 停利預警價格：漲至 {w_tp:.1f} 元提醒")

            if not update_fields:
                send_telegram("⚠️ 未能判斷具體的預警價格，請重新輸入。")
                cur.close()
                return

            params.append(t_lot_id)
            cur.execute(f"UPDATE position_lots SET {', '.join(update_fields)} WHERE lot_id = %s;", tuple(params))
            conn.commit()
            cur.close()

        msg = f"🔔 【指定預警價格設定成功】\n\n• 庫存編號：【第 {t_lot_id} 筆】\n• 標的：{t_name} ({t_code})\n" + "\n".join(summary_txt)
        send_telegram(msg, reply_markup=get_show_portfolio_markup())

    except Exception as e:
        send_telegram(f"❌ 設定預警價格失敗：{e}")

def handle_set_warning_buffer(data: dict):
    global DEFAULT_WARNING_BUFFER_PERCENT
    val = data.get("buffer_percent")
    is_global = data.get("is_global", False)
    lot_id = data.get("lot_id")
    code = data.get("stock_code")

    if val is None or float(val) <= 0:
        send_telegram("⚠️ 預警數值不正確，請重新輸入。")
        return

    new_wb = float(val)
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()

            if is_global or (not lot_id and not code):
                DEFAULT_WARNING_BUFFER_PERCENT = new_wb
                cur.execute("""
                    UPDATE position_lots 
                    SET warning_buffer_percent = %s 
                    WHERE monitoring_status = 'MONITORING';
                """, (DEFAULT_WARNING_BUFFER_PERCENT,))
                row_count = cur.rowcount
                conn.commit()
                cur.close()

                send_telegram(
                    f"🔔 【全域預警緩衝區間更新成功】\n\n• 全域預警門檻：距離防線／目標 {DEFAULT_WARNING_BUFFER_PERCENT:.1f}%\n• 已同步套用：共 {row_count} 筆監控中庫存",
                    reply_markup=get_show_portfolio_markup()
                )
                return

            if lot_id:
                cur.execute("SELECT lot_id, stock_code, stock_name FROM position_lots WHERE lot_id = %s AND monitoring_status = 'MONITORING';", (lot_id,))
            else:
                cur.execute("SELECT lot_id, stock_code, stock_name FROM position_lots WHERE stock_code = %s AND monitoring_status = 'MONITORING' ORDER BY lot_id DESC LIMIT 1;", (code,))

            row = cur.fetchone()
            if not row:
                send_telegram("⚠️ 找不到監控中的對應持倉資料。")
                cur.close()
                return

            t_lot_id, t_code, _ = row
            _, t_name = get_stock_info(t_code)
            cur.execute("UPDATE position_lots SET warning_buffer_percent = %s, warning_sl_price = NULL, warning_tp_price = NULL WHERE lot_id = %s;", (new_wb, t_lot_id))
            conn.commit()
            cur.close()

        send_telegram(
            f"🔔 【持倉預警設定更新成功】\n\n• 庫存編號：【第 {t_lot_id} 筆】\n• 標的：{t_name} ({t_code})\n• 個別預警門檻：距離防線／目標 {new_wb:.1f}%",
            reply_markup=get_show_portfolio_markup()
        )

    except Exception as e:
        send_telegram(f"❌ 更新預警門檻失敗：{e}")

def handle_delete_lot(data: dict):
    lot_ids = data.get("lot_ids", [])
    if not lot_ids:
        send_telegram("⚠️ 未指明要刪除哪一筆。")
        return

    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            deleted_items = []
            for lot_id in lot_ids:
                cur.execute("SELECT stock_code, stock_name, quantity, buy_price FROM position_lots WHERE lot_id = %s;", (lot_id,))
                row = cur.fetchone()
                if not row:
                    deleted_items.append(f"• 【第 {lot_id} 筆】⚠️ 找不到此批次資料")
                    continue

                code, _, cur_qty, buy_p = row
                _, name = get_stock_info(code)
                cur.execute("UPDATE position_lots SET monitoring_status = 'DELETED', quantity = 0 WHERE lot_id = %s;", (lot_id,))
                
                cur.execute("""
                    INSERT INTO trade_history (lot_id, stock_code, stock_name, action_type, quantity, price, realized_pnl, realized_pnl_pct, exit_reason)
                    VALUES (%s, %s, %s, 'DELETED', %s, %s, 0, 0, '作廢移除');
                """, (lot_id, code, name, cur_qty, float(buy_p)))

                deleted_items.append(f"• 【第 {lot_id} 筆】{name} ({code}) {cur_qty}張（已作廢移除）")

            conn.commit()
            cur.close()

        msg = f"🗑 【持倉資料作廢／刪除成功】\n\n" + "\n".join(deleted_items)
        send_telegram(msg, reply_markup=get_show_portfolio_markup())

    except Exception as e:
        send_telegram(f"❌ 刪除持倉失敗：{e}")

def handle_close_all_positions():
    send_telegram("📤 正在查詢全部持倉行情並準備平倉，請稍候。")
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT lot_id, stock_code, stock_name, buy_price, quantity
                FROM position_lots
                WHERE monitoring_status = 'MONITORING'
                ORDER BY lot_id ASC;
            """)
            rows = cur.fetchall()
            if not rows:
                cur.close()
                send_telegram("📋 目前沒有監控中的持倉可平倉。", reply_markup=get_show_portfolio_markup())
                return
            closed_lots = 0
            closed_quantity = 0
            total_pnl = 0.0
            skipped = []
            for lot_id, code, stock_name, buy_price, quantity in rows:
                _, market_price, _, _ = get_market_quote(code)
                if market_price is None:
                    skipped.append(f"{stock_name or code} ({code})")
                    continue
                buy_price = float(buy_price)
                market_price = float(market_price)
                quantity = int(quantity)
                pnl = (market_price - buy_price) * quantity * 1000
                pnl_pct = ((market_price - buy_price) / buy_price) * 100 if buy_price else 0
                cur.execute("""
                    UPDATE position_lots SET monitoring_status = 'CLOSED', quantity = 0
                    WHERE lot_id = %s AND monitoring_status = 'MONITORING';
                """, (lot_id,))
                if cur.rowcount != 1:
                    continue
                cur.execute("""
                    INSERT INTO trade_history
                    (lot_id, stock_code, stock_name, action_type, quantity, price, realized_pnl, realized_pnl_pct, exit_reason)
                    VALUES (%s, %s, %s, 'FULL_SELL', %s, %s, %s, %s, '全部平倉');
                """, (lot_id, code, stock_name, quantity, market_price, pnl, pnl_pct))
                closed_lots += 1
                closed_quantity += quantity
                total_pnl += pnl
            conn.commit()
            cur.close()
        sign = "+" if total_pnl >= 0 else ""
        msg = (f"📤【全部平倉處理完成】\n\n已平倉 {closed_lots} 筆，共 {closed_quantity} 張。"
               f"\n已實現損益：{sign}{total_pnl:,.0f} 元。")
        if skipped:
            msg += "\n\n以下標的無法取得行情，仍保留持倉：\n" + "\n".join(skipped)
        send_telegram(msg, reply_markup=get_show_portfolio_markup())
    except Exception as e:
        send_telegram(f"❌ 全部平倉處理失敗：{e}")

def handle_delete_all_positions():
    send_telegram("🗑 正在作廢目前全部監控中的持倉，請稍候。")
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT lot_id, stock_code, stock_name, buy_price, quantity
                FROM position_lots
                WHERE monitoring_status = 'MONITORING'
                ORDER BY lot_id ASC;
            """)
            rows = cur.fetchall()
            if not rows:
                cur.close()
                send_telegram("📋 目前沒有監控中的持倉可刪除。", reply_markup=get_show_portfolio_markup())
                return
            deleted_quantity = 0
            for lot_id, code, stock_name, buy_price, quantity in rows:
                cur.execute("""
                    UPDATE position_lots SET monitoring_status = 'DELETED', quantity = 0
                    WHERE lot_id = %s AND monitoring_status = 'MONITORING';
                """, (lot_id,))
                if cur.rowcount != 1:
                    continue
                cur.execute("""
                    INSERT INTO trade_history
                    (lot_id, stock_code, stock_name, action_type, quantity, price, realized_pnl, realized_pnl_pct, exit_reason)
                    VALUES (%s, %s, %s, 'DELETED', %s, %s, 0, 0, '全部刪除作廢');
                """, (lot_id, code, stock_name, quantity, float(buy_price)))
                deleted_quantity += int(quantity)
            conn.commit()
            cur.close()
        send_telegram(
            f"🗑【全部庫存已作廢】\n\n共作廢 {len(rows)} 筆、{deleted_quantity} 張。交易紀錄已新增；庫存流水號會繼續累計。",
            reply_markup=get_show_portfolio_markup(),
        )
    except Exception as e:
        send_telegram(f"❌ 全部刪除處理失敗：{e}")

def handle_update_settings(data: dict):
    global DEFAULT_TRAILING_STOP_PERCENT
    is_global = data.get("is_global", False)
    lot_id = data.get("lot_id")
    code = data.get("stock_code")
    new_tp_val = data.get("new_tp_val")
    new_tp_is_pct = data.get("new_tp_is_pct", False)
    new_sl_val = data.get("new_sl_val")
    new_sl_is_pct = data.get("new_sl_is_pct", False)
    new_ts = data.get("new_ts_percent")

    try:
        with get_db_connection() as conn:
            cur = conn.cursor()

            if is_global:
                if new_ts is not None:
                    DEFAULT_TRAILING_STOP_PERCENT = float(new_ts)
                    cur.execute("UPDATE position_lots SET trailing_stop_percent = %s WHERE monitoring_status = 'MONITORING';", (DEFAULT_TRAILING_STOP_PERCENT,))
                    row_count = cur.rowcount
                    conn.commit()
                    cur.close()
                    send_telegram(
                        f"⚙️ 【全域移動停利設定更新成功】\n\n• 全域移動停利：{DEFAULT_TRAILING_STOP_PERCENT:.1f}%\n• 已同步套用：共 {row_count} 筆監控中庫存",
                        reply_markup=get_show_portfolio_markup()
                    )
                    return
                else:
                    send_telegram("⚠️ 請指明要修改的全域參數。")
                    cur.close()
                    return

            if lot_id:
                cur.execute("""
                    SELECT lot_id, stock_code, stock_name, buy_price, stop_loss_price, take_profit_price, trailing_stop_percent, quantity 
                    FROM position_lots 
                    WHERE lot_id = %s AND monitoring_status = 'MONITORING';
                """, (lot_id,))
            elif code:
                cur.execute("""
                    SELECT lot_id, stock_code, stock_name, buy_price, stop_loss_price, take_profit_price, trailing_stop_percent, quantity 
                    FROM position_lots 
                    WHERE stock_code = %s AND monitoring_status = 'MONITORING'
                    ORDER BY lot_id DESC LIMIT 1;
                """, (code,))
            else:
                send_telegram("⚠️ 請指明要修改哪一筆或哪檔標的。")
                return

            row = cur.fetchone()
            if not row:
                send_telegram("⚠️ 找不到監控中的對應持倉資料。")
                cur.close()
                return

            t_lot_id, t_code, _, buy_p, cur_sl, cur_tp, cur_ts, cur_qty = row
            _, t_name = get_stock_info(t_code)
            buy_val = float(buy_p)
            final_sl = float(cur_sl)
            final_tp = float(cur_tp)
            final_ts = float(cur_ts) if cur_ts else DEFAULT_TRAILING_STOP_PERCENT

            sl_tag = ""
            tp_tag = ""
            ts_tag = ""

            update_sql_parts = []
            update_params = []

            if new_tp_val is not None:
                if new_tp_is_pct:
                    tp_pct = new_tp_val
                    final_tp = round(buy_val * (1 + tp_pct / 100.0), 2)
                else:
                    final_tp = float(new_tp_val)
                    tp_pct = abs(((final_tp - buy_val) / buy_val) * 100)
                update_sql_parts.append("take_profit_price = %s, take_profit_percent = %s")
                update_params.extend([final_tp, tp_pct])
                tp_tag = " （本次更新）"

            if new_sl_val is not None:
                if new_sl_is_pct:
                    sl_pct = new_sl_val
                    final_sl = round(buy_val * (1 - sl_pct / 100.0), 2)
                else:
                    final_sl = float(new_sl_val)
                    sl_pct = abs(((buy_val - final_sl) / buy_val) * 100)
                update_sql_parts.append("stop_loss_price = %s, stop_loss_percent = %s")
                update_params.extend([final_sl, sl_pct])
                sl_tag = " （本次更新）"

            if new_ts is not None:
                final_ts = float(new_ts)
                update_sql_parts.append("trailing_stop_percent = %s")
                update_params.append(final_ts)
                ts_tag = " （本次更新）"

            if update_sql_parts:
                update_params.append(t_lot_id)
                cur.execute(f"UPDATE position_lots SET {', '.join(update_sql_parts)} WHERE lot_id = %s;", tuple(update_params))
                conn.commit()

            cur.close()

        cur_sl_pct = abs(((buy_val - final_sl) / buy_val) * 100)
        cur_tp_pct = abs(((final_tp - buy_val) / buy_val) * 100)

        msg = (
            f"⚙️ 【持倉風控設定更新成功】\n\n"
            f"• 庫存編號：【第 {t_lot_id} 筆】\n"
            f"• 標的：{t_name} ({t_code})\n"
            f"• 買入成本：{buy_val:.1f} 元 ({cur_qty} 張)\n"
            f"• 停損防線：{final_sl:.1f} 元 (-{cur_sl_pct:.1f}%){sl_tag}\n"
            f"• 停利目標：{final_tp:.1f} 元 (+{cur_tp_pct:.1f}%){tp_tag}\n"
            f"• 移動停利：{final_ts:.1f}%{ts_tag}"
        )
        send_telegram(msg, reply_markup=get_show_portfolio_markup())

    except Exception as e:
        send_telegram(f"❌ 更新設定失敗：{e}")

def handle_sell_multi_lots(data: dict):
    lot_ids = data.get("lot_ids", [])
    custom_sell_price = data.get("sell_price")
    needed_qty = int(data.get("quantity") or 1)

    if not lot_ids:
        send_telegram("⚠️ 未指定任何持倉批次編號。")
        return

    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            sold_details = []
            for lot_id in lot_ids:
                cur.execute("SELECT stock_code, stock_name, buy_price, quantity, monitoring_status FROM position_lots WHERE lot_id = %s;", (lot_id,))
                row = cur.fetchone()
                if not row:
                    sold_details.append(f"• 【第 {lot_id} 筆】⚠️ 找不到此批次資料")
                    continue

                code, _, buy_price, cur_qty, status = row
                _, name = get_stock_info(code)
                if status == 'CLOSED':
                    sold_details.append(f"• 【第 {lot_id} 筆】{name}：已平倉結案，無法重複賣出")
                    continue

                sell_price = custom_sell_price
                if not sell_price:
                    _, market_p, _, _ = get_market_quote(code)
                    sell_price = market_p

                if not sell_price:
                    sold_details.append(f"• 【第 {lot_id} 筆】{name} ({code})：無法取得行情，請手動指定賣出價格")
                    continue

                sell_price = float(sell_price)
                buy_p = float(buy_price)

                if cur_qty <= needed_qty:
                    actual_sold = cur_qty
                    rem_qty = 0
                    action_type = 'FULL_SELL'
                    cur.execute("UPDATE position_lots SET monitoring_status = 'CLOSED', quantity = 0 WHERE lot_id = %s;", (lot_id,))
                else:
                    actual_sold = needed_qty
                    rem_qty = cur_qty - needed_qty
                    action_type = 'PARTIAL_SELL'
                    cur.execute("UPDATE position_lots SET quantity = %s WHERE lot_id = %s;", (rem_qty, lot_id))

                diff_per_share = sell_price - buy_p
                pct = (diff_per_share / buy_p) * 100
                total_pnl = diff_per_share * actual_sold * 1000
                sign = "+" if total_pnl >= 0 else ""

                cur.execute("""
                    INSERT INTO trade_history (lot_id, stock_code, stock_name, action_type, quantity, price, realized_pnl, realized_pnl_pct, exit_reason)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, '手動平倉');
                """, (lot_id, code, name, action_type, actual_sold, sell_price, total_pnl, pct))

                rem_status_txt = f"剩餘：{rem_qty} 張" if rem_qty > 0 else "已全數出場"
                sold_details.append(
                    f"• 【第 {lot_id} 筆】{name} ({code})\n"
                    f"  交易動作：{'部分平倉' if action_type == 'PARTIAL_SELL' else '完全平倉'} (賣出 {actual_sold} 張)\n"
                    f"  剩餘庫存：{rem_status_txt}監控中\n"
                    f"  買入成本：{buy_p:.1f} 元 ➔ 出場價格：{sell_price:.1f} 元\n"
                    f"  實現損益：{sign}{total_pnl:,.0f} 元 ({sign}{pct:.2f}%)"
                )

            conn.commit()
            cur.close()

        msg = f"📤 【持倉平倉結算完成】\n\n" + "\n\n".join(sold_details)
        send_telegram(msg, reply_markup=get_show_portfolio_markup())

    except Exception as e:
        send_telegram(f"❌ 指定平倉失敗：{e}")

def handle_sell_lot(data: dict):
    code = data.get("stock_code")
    sell_price = data.get("sell_price")
    needed_qty = int(data.get("quantity") or 1)

    if not code:
        send_telegram("⚠️ 未能判斷賣出標的。")
        return

    if not sell_price:
        _, market_p, _, _ = get_market_quote(code)
        sell_price = market_p

    if not sell_price:
        send_telegram(f"⚠️ 無法取得 {code} 行情，請主動輸入賣出金額。")
        return

    sell_price = float(sell_price)
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT lot_id, stock_name, buy_price, quantity FROM position_lots WHERE stock_code = %s AND monitoring_status = 'MONITORING' ORDER BY lot_id ASC;", (code,))
            rows = cur.fetchall()

            if not rows:
                send_telegram(f"⚠️ 庫存中無監控中的 {code} 持倉可平倉。")
                cur.close()
                return

            total_available = sum(r[3] for r in rows)
            if total_available < needed_qty:
                send_telegram(f"⚠️ 庫存不足！{code} 目前僅剩 {total_available} 張。")
                cur.close()
                return

            remaining_to_sell = needed_qty
            sold_summary = []

            for r in rows:
                if remaining_to_sell <= 0:
                    break
                lot_id, _, buy_price, cur_qty = r
                _, name = get_stock_info(code)
                buy_p = float(buy_price)

                if cur_qty <= remaining_to_sell:
                    deduct = cur_qty
                    rem_qty = 0
                    action_type = 'FULL_SELL'
                    cur.execute("UPDATE position_lots SET monitoring_status = 'CLOSED', quantity = 0 WHERE lot_id = %s;", (lot_id,))
                    remaining_to_sell -= deduct
                else:
                    deduct = remaining_to_sell
                    rem_qty = cur_qty - deduct
                    action_type = 'PARTIAL_SELL'
                    cur.execute("UPDATE position_lots SET quantity = %s WHERE lot_id = %s;", (rem_qty, lot_id))
                    remaining_to_sell = 0

                diff_per_share = sell_price - buy_p
                pct = (diff_per_share / buy_p) * 100
                total_pnl = diff_per_share * deduct * 1000
                sign = "+" if total_pnl >= 0 else ""

                cur.execute("""
                    INSERT INTO trade_history (lot_id, stock_code, stock_name, action_type, quantity, price, realized_pnl, realized_pnl_pct, exit_reason)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, '手動平倉');
                """, (lot_id, code, name, action_type, deduct, sell_price, total_pnl, pct))

                rem_status_txt = f"剩餘：{rem_qty} 張" if rem_qty > 0 else "已全數出場"
                sold_summary.append(
                    f"• 【第 {lot_id} 筆】{name}：賣出 {deduct} 張 ({rem_status_txt})\n"
                    f"  買入成本：{buy_p:.1f} 元 ➔ 出場價格：{sell_price:.1f} 元\n"
                    f"  實現損益：{sign}{total_pnl:,.0f} 元 ({sign}{pct:.2f}%)"
                )

            conn.commit()
            cur.close()

        msg = f"📤 【持倉平倉結算完成（共賣出 {needed_qty} 張）】\n\n" + "\n\n".join(sold_summary)
        send_telegram(msg, reply_markup=get_show_portfolio_markup())

    except Exception as e:
        send_telegram(f"❌ 沖銷失敗：{e}")

def handle_add_lot(data: dict):
    code = data.get("stock_code")
    _, name = get_stock_info(code)
    raw_price = data.get("buy_price")
    
    if not code or raw_price is None:
        send_telegram("⚠️ 未能確認有效的代號或買進價格，請重新輸入。")
        return

    buy_price = float(raw_price)
    qty = int(data.get("quantity") or 1)
    
    sl_pct = float(DEFAULT_STOP_LOSS_PERCENT)
    tp_pct = float(DEFAULT_TAKE_PROFIT_PERCENT)
    ts_pct = float(DEFAULT_TRAILING_STOP_PERCENT)
    wb_pct = float(DEFAULT_WARNING_BUFFER_PERCENT)
    ts_act_pct = DEFAULT_TRAILING_ACTIVATION_PERCENT

    sl_price = round(buy_price * (1 - sl_pct / 100), 2)
    tp_price = round(buy_price * (1 + tp_pct / 100), 2)

    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO position_lots 
                (stock_code, stock_name, buy_date, buy_price, quantity, 
                 take_profit_percent, stop_loss_percent, trailing_stop_percent, 
                 take_profit_price, stop_loss_price, highest_price, monitoring_status, warning_buffer_percent,
                 ts_activation_percent)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'MONITORING', %s, %s)
                RETURNING lot_id;
            """, (code, name, date.today(), buy_price, qty, tp_pct, sl_pct, ts_pct, tp_price, sl_price, buy_price, wb_pct, ts_act_pct))
            lot_id = cur.fetchone()[0]

            cur.execute("""
                INSERT INTO trade_history (lot_id, stock_code, stock_name, action_type, quantity, price, realized_pnl, realized_pnl_pct, exit_reason)
                VALUES (%s, %s, %s, 'BUY', %s, %s, 0, 0, '建倉買進');
            """, (lot_id, code, name, qty, buy_price))

            conn.commit()
            cur.close()

        msg = (
            f"✅ 【持倉建立成功】\n\n"
            f"• 庫存編號：【第 {lot_id} 筆】\n"
            f"• 標的：{name} ({code})\n"
            f"• 買入成本：{buy_price:.1f} 元 ({qty} 張)\n"
            f"• 停損防線：{sl_price:.1f} 元 (-{sl_pct:.1f}%)\n"
            f"• 停利目標：{tp_price:.1f} 元 (+{tp_pct:.1f}%)\n"
            f"• 移動停利：{ts_pct:.1f}%（獲利達 +{ts_act_pct:.1f}% 啟動）\n"
            f"• 預警通知：{wb_pct:.1f}%"
        )
        send_telegram(msg, reply_markup=get_show_portfolio_markup())
    except Exception as e:
        send_telegram(f"❌ 寫入失敗：{e}")

def handle_query(filter_keyword: str = None, sort_by_profit: bool = False, chat_id=None):
    send_telegram("📋 正在讀取目前的全部持倉，請稍候。", chat_id=chat_id)
    summary = get_portfolio_summary_text(filter_keyword=filter_keyword, sort_by_profit=sort_by_profit)
    send_telegram(summary, chat_id=chat_id, reply_markup=get_show_portfolio_markup())

def handle_get_price(data: dict):
    raw_code = data.get("stock_code")
    code, name = get_stock_info(raw_code)

    if not code:
        send_telegram("⚠️ 未能判斷查詢標的。")
        return

    _, price, _, _ = get_market_quote(code)
    if price is None:
        send_telegram(f"❌ 市場查無 {name} ({code}) 行情。")
    else:
        send_telegram(f"📊 【市場即時行情】\n\n• 標的：{name} ({code})\n• 最新成交價：{price:.2f} 元")

def send_daily_market_report():
    today = date.today()
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM daily_reports WHERE report_date = %s;", (today,))
            if cur.fetchone():
                cur.close()
                return

            cur.execute("""
                SELECT event_type, COUNT(*) 
                FROM notification_records 
                WHERE trade_date = %s 
                GROUP BY event_type;
            """, (today,))
            ev_counts = dict(cur.fetchall())

            sl_cnt = ev_counts.get('STOP_LOSS', 0)
            tp_cnt = ev_counts.get('TAKE_PROFIT', 0)
            ts_cnt = ev_counts.get('TRAILING_STOP', 0)
            warn_cnt = ev_counts.get('APPROACHING_STOP_LOSS', 0) + ev_counts.get('APPROACHING_TAKE_PROFIT', 0)

            cur.execute("INSERT INTO daily_reports (report_date) VALUES (%s);", (today,))
            conn.commit()
            cur.close()

        report_msg = (
            f"📊 【{today.strftime('%Y-%m-%d')} 今日風控觸發統計】\n\n"
            f"• 停損觸發：{sl_cnt} 次\n"
            f"• 停利達標：{tp_cnt} 次\n"
            f"• 移動停利：{ts_cnt} 次\n"
            f"• 接近預警：{warn_cnt} 次\n\n"
            f"🛡 今日盯盤結束，祝您投資順心！"
        )
        send_telegram(report_msg)
        print("📢 今日風控統計日報已成功發送！")

    except Exception as e:
        print(f"❌ 統計發送失敗：{e}")

def background_monitor():
    while True:
        if is_market_closing_time():
            send_daily_market_report()

        if not is_market_open():
            time.sleep(120)
            continue

        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT lot_id, stock_code, stock_name, buy_price, stop_loss_price, 
                           take_profit_price, quantity, stop_loss_percent, take_profit_percent, 
                           highest_price, trailing_stop_percent, warning_buffer_percent, 
                           warning_sl_price, warning_tp_price, ts_activation_percent
                    FROM position_lots 
                    WHERE monitoring_status = 'MONITORING';
                """)
                lots = cur.fetchall()

                for lot in lots:
                    lot_id, code, _, buy_p, sl_p, tp_p, qty, sl_pct, tp_pct, high_p, ts_pct, wb_pct, w_sl_p, w_tp_p, ts_act_p = lot
                    _, name = get_stock_info(code)
                    _, cur_price, day_high, day_low = get_market_quote(code)
                    if not cur_price:
                        continue

                    today = date.today()
                    buy_val = float(buy_p)
                    sl_val = float(sl_p)
                    tp_val = float(tp_p)
                    high_val = float(high_p) if high_p else buy_val
                    ts_val = float(ts_pct) if ts_pct else DEFAULT_TRAILING_STOP_PERCENT
                    wb_val = float(wb_pct) if wb_pct is not None else DEFAULT_WARNING_BUFFER_PERCENT
                    w_sl_val = float(w_sl_p) if w_sl_p is not None else None
                    w_tp_val = float(w_tp_p) if w_tp_p is not None else None
                    ts_act_val = float(ts_act_p) if ts_act_p is not None else DEFAULT_TRAILING_ACTIVATION_PERCENT
                    diff_pct = ((cur_price - buy_val) / buy_val) * 100

                    if cur_price > high_val:
                        high_val = cur_price
                        cur.execute("UPDATE position_lots SET highest_price = %s WHERE lot_id = %s;", (high_val, lot_id))
                        conn.commit()

                    trailing_stop_price = round(high_val * (1 - ts_val / 100), 2)

                    is_hit_sl = (cur_price <= sl_val)
                    if is_hit_sl:
                        cur.execute("SELECT 1 FROM notification_records WHERE lot_id = %s AND trade_date = %s AND event_type = 'STOP_LOSS';", (lot_id, today))
                        if not cur.fetchone():
                            actual_trigger_p = cur_price
                            trigger_diff_pct = ((actual_trigger_p - buy_val) / buy_val) * 100
                            sl_alert_msg = (
                                f"🚨 【停損出場警報！】\n\n"
                                f"• 庫存編號：【第 {lot_id} 筆】\n"
                                f"• 標的：{name} ({code}) ({qty} 張)\n"
                                f"• 買入成本：{buy_val:.1f} 元\n"
                                f"• 停損防線：{sl_val:.1f} 元 (-{abs(float(sl_pct)):.1f}%)\n"
                                f"• 目前現價：{cur_price:.2f} 元 (觸發價: {actual_trigger_p:.2f}元, {trigger_diff_pct:.2f}%)\n\n"
                                f"⚠️ 已跌破停損防線！請嚴守紀律果斷出場！"
                            )
                            send_telegram(sl_alert_msg, silent=False)
                            cur.execute("INSERT INTO notification_records (lot_id, trade_date, event_type, trigger_price) VALUES (%s, %s, 'STOP_LOSS', %s);", (lot_id, today, actual_trigger_p))
                            conn.commit()
                        continue

                    is_hit_tp = (cur_price >= tp_val)
                    if is_hit_tp:
                        cur.execute("SELECT 1 FROM notification_records WHERE lot_id = %s AND trade_date = %s AND event_type = 'TAKE_PROFIT';", (lot_id, today))
                        if not cur.fetchone():
                            actual_trigger_p = cur_price
                            trigger_diff_pct = ((actual_trigger_p - buy_val) / buy_val) * 100
                            tp_alert_msg = (
                                f"🎉 【停利達標警報！】\n\n"
                                f"• 庫存編號：【第 {lot_id} 筆】\n"
                                f"• 標的：{name} ({code}) ({qty} 張)\n"
                                f"• 買入成本：{buy_val:.1f} 元\n"
                                f"• 停利目標：{tp_val:.1f} 元 (+{abs(float(tp_pct)):.1f}%)\n"
                                f"• 目前現價：{cur_price:.2f} 元 (達標價: {actual_trigger_p:.2f}元, +{trigger_diff_pct:.2f}%)\n\n"
                                f"💰 獲利達標！可考慮分批獲利入袋！"
                            )
                            send_telegram(tp_alert_msg, silent=False)
                            cur.execute("INSERT INTO notification_records (lot_id, trade_date, event_type, trigger_price) VALUES (%s, %s, 'TAKE_PROFIT', %s);", (lot_id, today, actual_trigger_p))
                            conn.commit()
                        continue

                    activation_price = buy_val * (1 + ts_act_val / 100)
                    has_reached_activation = high_val >= activation_price

                    if has_reached_activation and cur_price <= trailing_stop_price:
                        cur.execute("SELECT 1 FROM notification_records WHERE lot_id = %s AND trade_date = %s AND event_type = 'TRAILING_STOP';", (lot_id, today))
                        if not cur.fetchone():
                            ts_alert_msg = (
                                f"🚨 【移動停利出場警報！】\n\n"
                                f"• 庫存編號：【第 {lot_id} 筆】\n"
                                f"• 標的：{name} ({code}) ({qty} 張)\n"
                                f"• 買入成本：{buy_val:.1f} 元\n"
                                f"• 啟動條件：曾達獲利 +{ts_act_val:.1f}% 門檻（最高 {high_val:.1f} 元）\n"
                                f"• 移動防線：{trailing_stop_price:.1f} 元 (自高點回檔 -{ts_val:.1f}%)\n"
                                f"• 目前現價：{cur_price:.2f} 元 ({diff_pct:.2f}%)\n\n"
                                f"⚠️ 自最高點回檔觸發移動停利！請獲利入袋！"
                            )
                            send_telegram(ts_alert_msg, silent=False)
                            cur.execute("INSERT INTO notification_records (lot_id, trade_date, event_type, trigger_price) VALUES (%s, %s, 'TRAILING_STOP', %s);", (lot_id, today, cur_price))
                            conn.commit()
                        continue

                    is_warn_sl = False
                    if w_sl_val is not None:
                        is_warn_sl = (cur_price <= w_sl_val and cur_price > sl_val)
                    else:
                        dist_to_sl_pct = ((cur_price - sl_val) / buy_val) * 100
                        is_warn_sl = (0 < dist_to_sl_pct <= wb_val)

                    is_warn_tp = False
                    if w_tp_val is not None:
                        is_warn_tp = (cur_price >= w_tp_val and cur_price < tp_val)
                    else:
                        dist_to_tp_pct = ((tp_val - cur_price) / buy_val) * 100
                        is_warn_tp = (0 < dist_to_tp_pct <= wb_val)

                    if is_warn_sl:
                        cur.execute("SELECT 1 FROM notification_records WHERE lot_id = %s AND trade_date = %s AND event_type = 'APPROACHING_STOP_LOSS';", (lot_id, today))
                        if not cur.fetchone():
                            warn_sl_msg = (
                                f"⚠️ 【接近停損防線預警】\n\n"
                                f"• 庫存編號：【第 {lot_id} 筆】\n"
                                f"• 標的：{name} ({code}) ({qty} 張)\n"
                                f"• 目前現價：{cur_price:.2f} 元 ({diff_pct:.2f}%)\n"
                                f"• 停損防線：{sl_val:.1f} 元 (-{abs(float(sl_pct)):.1f}%)\n"
                                f"• 距離防線：僅剩 {cur_price - sl_val:.2f} 元\n\n"
                                f"⏳ 股價逼近停損，請做好出場準備。（即時預警通知）"
                            )
                            send_telegram(warn_sl_msg, silent=False)
                            cur.execute("INSERT INTO notification_records (lot_id, trade_date, event_type, trigger_price) VALUES (%s, %s, 'APPROACHING_STOP_LOSS', %s);", (lot_id, today, cur_price))
                            conn.commit()

                    elif is_warn_tp:
                        cur.execute("SELECT 1 FROM notification_records WHERE lot_id = %s AND trade_date = %s AND event_type = 'APPROACHING_TAKE_PROFIT';", (lot_id, today))
                        if not cur.fetchone():
                            warn_tp_msg = (
                                f"🎯 【接近停利目標預警】\n\n"
                                f"• 庫存編號：【第 {lot_id} 筆】\n"
                                f"• 標的：{name} ({code}) ({qty} 張)\n"
                                f"• 目前現價：{cur_price:.2f} 元 (+{diff_pct:.2f}%)\n"
                                f"• 停利目標：{tp_val:.1f} 元 (+{abs(float(tp_pct)):.1f}%)\n"
                                f"• 距離目標：_{tp_val - cur_price:.2f} 元\n\n"
                                f"⏳ 股價即將達標，可留意獲利賣單。（即時預警通知）"
                            )
                            send_telegram(warn_tp_msg, silent=False)
                            cur.execute("INSERT INTO notification_records (lot_id, trade_date, event_type, trigger_price) VALUES (%s, %s, 'APPROACHING_TAKE_PROFIT', %s);", (lot_id, today, cur_price))
                            conn.commit()

                cur.close()
        except Exception as e:
            print(f"❌ 巡邏例外錯誤：{e}")

        time.sleep(60)

class HealthCheckHandler(BaseHTTPRequestHandler):
    def _send_health_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()

    def do_HEAD(self):
        self._send_health_headers()

    def do_GET(self):
        self._send_health_headers()
        self.wfile.write(b"AI Trading Bot is running 24/7!")

    def log_message(self, format, *args):
        return

def run_web_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    print(f"Web server started on port {port}")
    server.serve_forever()

def run_bot():
    print("🤖 周大 AI 交易管家已上線，正在檢查資料庫結構並監聽訊息與圖片...")
    api_url = f"https://api.telegram.org/bot{BOT_TOKEN}"
    try:
        webhook_resp = requests.get(f"{api_url}/getWebhookInfo", timeout=10)
        webhook_data = webhook_resp.json()
        if not webhook_data.get("ok"):
            raise RuntimeError(f"Telegram API 錯誤：{webhook_data.get('description', 'getWebhookInfo 失敗')}")
        if webhook_data.get("result", {}).get("url"):
            delete_resp = requests.post(
                f"{api_url}/deleteWebhook",
                data={"drop_pending_updates": "false"},
                timeout=10,
            )
            delete_data = delete_resp.json()
            if not delete_data.get("ok"):
                raise RuntimeError(f"無法關閉既有 webhook：{delete_data.get('description', '未知錯誤')}")
            print("ℹ️ 已移除既有 Telegram webhook，改用 getUpdates 輪詢；保留尚未處理的訊息。")
    except Exception as e:
        safe_error = str(e).replace(BOT_TOKEN, "<redacted>")
        raise RuntimeError(f"Telegram 啟動檢查失敗：{type(e).__name__}: {safe_error}") from e

    init_db_schema()
    
    threading.Thread(target=background_monitor, daemon=True).start()
    threading.Thread(target=run_web_server, daemon=True).start()

    headers = {"User-Agent": "Mozilla/5.0"}
    last_update_id = 0

    while True:
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
            params = {"offset": last_update_id + 1, "timeout": 0}
            
            resp = requests.get(url, params=params, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if "result" in data and len(data["result"]) > 0:
                    for item in data["result"]:
                        last_update_id = item["update_id"]
                        
                        if "callback_query" in item:
                            cb = item["callback_query"]
                            cb_id = cb["id"]
                            cb_data = cb.get("data")
                            callback_chat_id = (cb.get("message") or {}).get("chat", {}).get("id")
                            
                            try:
                                requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery", json={"callback_query_id": cb_id}, timeout=5)
                            except Exception as e:
                                print(f"⚠️ Telegram 按鈕確認失敗：{type(e).__name__}: {e}", flush=True)
                            
                            if str(callback_chat_id) != str(CHAT_ID):
                                print("⚠️ 忽略未授權聊天的按鈕事件", flush=True)
                                continue

                            if cb_data == "SHOW_ALL_PORTFOLIO":
                                print("📋 收到顯示全部庫存按鈕點擊", flush=True)
                                threading.Thread(target=handle_query, kwargs={"chat_id": callback_chat_id}, daemon=True).start()
                            continue

                        msg = item.get("message", {})
                        if str(msg.get('chat', {}).get('id')) != str(CHAT_ID):
                            print('⚠️ 忽略未授權聊天的訊息', flush=True)
                            continue
                        
                        if "photo" in msg:
                            print("📷 收到使用者上傳的庫存截圖...")
                            photo_file_id = msg["photo"][-1]["file_id"]
                            send_telegram("🤖 收到您的庫存截圖，AI 正在透過 gemini-3.5-flash-lite 全力辨識中，請稍候...")
                            threading.Thread(target=handle_screenshot_image, args=(photo_file_id,), daemon=True).start()
                            continue

                        text = msg.get("text", "").strip()
                        if not text:
                            continue

                        print(f"📩 收到訊息: {text}", flush=True)
                        normalized_text = re.sub(r"[\s，。！？、,.!?]+", "", text).lower()
                        inventory_phrases = {
                            "庫存", "查詢", "查詢庫存", "顯示庫存", "顯示全部庫存", "顯示所有庫存",
                            "顯示目前庫存", "目前庫存", "目前持股", "目前的持股狀況", "顯示目前持股狀況", "目前持股狀況", "持股狀況", "顯示持股", "查看庫存", "查看持股", "持倉清單",
                            "持股", "清單", "庫存清單", "全部庫存", "所有庫存",
                            "库存", "查询", "查询库存", "显示库存", "显示全部库存", "显示所有库存",
                            "当前库存", "当前持股", "当前的持股情况", "持股情况", "清单", "库存清单",
                        }
                        close_all_phrases = {"全部平倉", "全數平倉", "平倉全部", "全部出場", "全部賣出", "全部平仓"}
                        delete_all_phrases = {"全部刪除", "全部删除", "刪除全部庫存", "刪除全部持倉", "刪除全部持仓", "清空庫存", "清空持倉"}
                        if normalized_text in inventory_phrases:
                            threading.Thread(target=handle_query, daemon=True).start()
                            continue
                        if normalized_text in close_all_phrases:
                            threading.Thread(target=handle_close_all_positions, daemon=True).start()
                            continue
                        if normalized_text in delete_all_phrases:
                            threading.Thread(target=handle_delete_all_positions, daemon=True).start()
                            continue

                        if text in ["/report", "日報", "風控統計"]:
                            send_daily_market_report()
                        elif text in ["/start", "你好", "哈囉"]:
                            send_telegram("👋 周大您好！AI 管家已就位。可直接輸入「買進 61041 147元 1張」、傳送**庫存截圖**或輸入「查詢」。")
                        else:
                            parsed = extract_trade_intent(text)
                            act = parsed.get("action")
                            print(f"👉 動作判斷: {act}", flush=True)

                            if act == "QUERY_PORTFOLIO_FILTERED":
                                threading.Thread(target=handle_query, kwargs={
                                    "filter_keyword": parsed.get("filter_keyword"),
                                    "sort_by_profit": parsed.get("sort_by_profit", False),
                                }, daemon=True).start()
                            elif act == "SET_WARNING_PRICE":
                                handle_set_warning_price(parsed)
                            elif act == "SET_WARNING_BUFFER":
                                handle_set_warning_buffer(parsed)
                            elif act == "DELETE_LOT":
                                handle_delete_lot(parsed)
                            elif act == "UPDATE_SETTINGS":
                                handle_update_settings(parsed)
                            elif act == "UPDATE_GLOBAL_SETTINGS_EXT":
                                handle_update_global_settings_ext(parsed)
                            elif act in ["SELL_MULTI_LOTS", "SELL_BY_LOT_ID"]:
                                if "lot_ids" not in parsed and "lot_id" in parsed:
                                    parsed["lot_ids"] = [parsed["lot_id"]]
                                handle_sell_multi_lots(parsed)
                            elif act == "GET_PRICE":
                                handle_get_price(parsed)
                            elif act == "ADD_LOT":
                                handle_add_lot(parsed)
                            elif act == "SELL_LOT":
                                handle_sell_lot(parsed)
                            elif act == "QUERY_PORTFOLIO":
                                threading.Thread(target=handle_query, daemon=True).start()
                            else:
                                send_telegram("🤖 收到訊息，如需記帳請指明標的與價格（例如：買進 61041 147元 1張），或直接傳送庫存截圖。")
            else:
                try:
                    telegram_error = resp.json().get("description", "")
                except ValueError:
                    telegram_error = ""
                print(f"❌ Telegram getUpdates HTTP {resp.status_code}: {telegram_error}")

        except Exception as e:
            safe_error = str(e).replace(BOT_TOKEN, "<redacted>")
            print(f"❌ Telegram 主迴圈錯誤：{type(e).__name__}: {safe_error}")

        time.sleep(1)

if __name__ == "__main__":
    run_bot()
