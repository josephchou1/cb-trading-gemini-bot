import os
import time
import json
import re
import threading
import io
from pathlib import Path
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
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
TAIPEI_TZ = ZoneInfo("Asia/Taipei")

def taipei_now():
    return datetime.now(TAIPEI_TZ)

def get_stock_info(identifier: str):
    if not identifier:
        return None, None
    identifier = str(identifier).strip()
    
    if identifier in twstock.codes:
        return identifier, twstock.codes[identifier].name
    
    if len(identifier) in (5, 6) and identifier.isdigit():
        base_code = identifier[:4]
        cb_suffix = identifier[4:] if len(identifier) == 6 else identifier[4]
        suffix_map = {
            '1': '一', '01': '一', '2': '二', '02': '二', '3': '三', '03': '三',
            '4': '四', '04': '四', '5': '五', '05': '五', '6': '六', '06': '六',
            '7': '七', '07': '七', '8': '八', '08': '八', '9': '九', '09': '九',
            '0': '十', '10': '十',
        }
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
    now = taipei_now()
    if now.weekday() >= 5:
        return False
    market_start = dtime(9, 0, 0)
    market_end = dtime(13, 45, 0)
    return market_start <= now.time() <= market_end

def is_market_closing_time() -> bool:
    now = taipei_now()
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

    if len(code) in (5, 6) and code.isdigit():
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
            today = taipei_now().date()

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
            triggered_map.setdefault(r_lot_id, set()).add(r_event)

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

            event_types = triggered_map.get(lot_id, set())
            icon = "🔹"
            event_labels = {
                'STOP_LOSS': ("🔴", "停損已觸發"),
                'TAKE_PROFIT': ("🟡", "停利已觸發"),
                'TRAILING_STOP': ("🟠", "移動停利已觸發"),
                'APPROACHING_STOP_LOSS': ("🔸", "停損預警已觸發"),
                'APPROACHING_TAKE_PROFIT': ("🔸", "停利預警已觸發"),
            }
            display_order = (
                'STOP_LOSS', 