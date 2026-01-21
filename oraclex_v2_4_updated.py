#!/usr/bin/env python3
"""
ORACLEX V2.5+ - FIXED DATA HANDLING
- Logs incoming data from Relay
- Processes market data correctly
- Returns analysis for all symbols
"""

import os
import json
import asyncio
import sqlite3
from aiohttp import web
from datetime import datetime, timezone
from typing import Dict
from collections import deque

import pandas as pd
import numpy as np
from scipy import stats

try:
    import google.generativeai as genai
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        GEMINI_ENABLED = True
    else:
        GEMINI_ENABLED = False
except:
    GEMINI_ENABLED = False

# Data storage
market_data = {
    "XAUUSD": {"candles": deque(maxlen=200), "current": None, "timestamp": None},
    "BTCUSD": {"candles": deque(maxlen=200), "current": None, "timestamp": None},
    "ETHUSD": {"candles": deque(maxlen=200), "current": None, "timestamp": None},
    "EURUSD": {"candles": deque(maxlen=200), "current": None, "timestamp": None},
    "GBPUSD": {"candles": deque(maxlen=200), "current": None, "timestamp": None},
    "AUDUSD": {"candles": deque(maxlen=200), "current": None, "timestamp": None},
    "NZDUSD": {"candles": deque(maxlen=200), "current": None, "timestamp": None},
}

def init_db():
    conn = sqlite3.connect("oraclex_analysis.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS analysis (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        symbol TEXT,
        confluence REAL,
        confidence REAL,
        interpretation TEXT
    )""")
    conn.commit()
    conn.close()

def save_analysis(symbol, confluence, confidence, interpretation):
    try:
        conn = sqlite3.connect("oraclex_analysis.db")
        c = conn.cursor()
        c.execute("INSERT INTO analysis VALUES (NULL, ?, ?, ?, ?, ?)",
                 (datetime.now(timezone.utc).isoformat(), symbol, confluence, confidence, interpretation))
        conn.commit()
        conn.close()
    except:
        pass

# Analysis functions
def calculate_market_regime(df: pd.DataFrame) -> Dict:
    if len(df) < 20:
        return {"trend": "Unknown", "volatility": "Normal", "structure": "Unknown"}
    
    ema_50 = df["close"].ewm(span=50).mean()
    ema_200 = df["close"].ewm(span=200).mean()
    
    ratio = ema_50.iloc[-1] / ema_200.iloc[-1]
    if ratio > 1.01:
        trend = "Strong"
    elif ratio < 0.99:
        trend = "Weak"
    else:
        trend = "Ranging"
    
    bb_std = df["close"].std()
    bb_width = 2 * bb_std / df["close"].mean()
    historical_widths = [2 * df["close"].iloc[max(0, i-20):i].std() / df["close"].iloc[i] 
                         for i in range(20, len(df), 10)]
    if historical_widths:
        median = np.median(historical_widths)
        if bb_width > median * 1.5:
            volatility = "Expanding"
        elif bb_width < median * 0.5:
            volatility = "Contracting"
        else:
            volatility = "Normal"
    else:
        volatility = "Normal"
    
    highs = df["high"].tail(20).values
    swings = len(np.where(np.diff(highs) > 0)[0])
    structure = "Clean" if swings < 5 else "Choppy"
    
    return {"trend": trend, "volatility": volatility, "structure": structure}

def calculate_bias_stability(df: pd.DataFrame) -> Dict:
    if len(df) < 5:
        return {"bias": "NEUTRAL", "active_since_minutes": 0, "last_flip_minutes_ago": None}
    
    ema_50 = df["close"].ewm(span=50).mean()
    current_bias = "BULLISH" if df["close"].iloc[-1] > ema_50.iloc[-1] else "BEARISH"
    
    bias_started = 0
    for i in range(len(df) - 1, 0, -1):
        bias = "BULLISH" if df["close"].iloc[i] > ema_50.iloc[i] else "BEARISH"
        if bias != current_bias:
            bias_started = len(df) - i
            break
    
    return {"bias": current_bias, "active_since_minutes": bias_started, "last_flip_minutes_ago": bias_started if bias_started > 0 else None}

def calculate_confluence_breakdown(df: pd.DataFrame) -> Dict:
    ema_50 = df["close"].ewm(span=50).mean()
    ema_trend = 1 if df["close"].iloc[-1] > ema_50.iloc[-1] else 0
    
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    momentum = 1 if rsi.iloc[-1] > 50 else 0
    
    bb_std = df["close"].std()
    bb_upper = df["close"].mean() + (2 * bb_std)
    volatility = 1 if df["close"].iloc[-1] > bb_upper * 0.95 else 0
    
    volume = 1 if df["volume"].iloc[-1] > df["volume"].mean() else 0
    
    agreement = (ema_trend + momentum + volatility + volume) / 4
    
    return {
        "EMA_Trend": {"weight": 0.35 + (0.05 * agreement), "active": ema_trend},
        "Momentum": {"weight": 0.30 + (0.05 * agreement), "active": momentum},
        "Volatility": {"weight": 0.20, "active": volatility},
        "Volume": {"weight": 0.15 - (0.05 * agreement), "active": volume}
    }

def calculate_state_statistics(df: pd.DataFrame) -> Dict:
    if len(df) < 100:
        return {"continuation": 45, "reversal": 35, "consolidation": 20}
    
    returns = df["close"].pct_change()
    cont = (returns.iloc[-5:] > 0).sum() / 5 * 100
    rev = (returns.iloc[-5:] < 0).sum() / 5 * 100
    
    return {
        "continuation": round(cont, 0),
        "reversal": round(rev, 0),
        "consolidation": round(100 - cont - rev, 0)
    }

def get_current_session() -> str:
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 16:
        return "Europe"
    elif 16 <= hour < 24:
        return "US"
    return "Overlap"

def calculate_dynamic_confluence(df: pd.DataFrame) -> float:
    if len(df) < 10:
        return 50.0
    
    scores = []
    
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_hist = rsi.dropna().tolist()
    if rsi_hist and len(rsi_hist) > 5:
        rsi_pct = stats.percentileofscore(rsi_hist[-100:], rsi.iloc[-1])
        scores.append(1 if rsi_pct > 50 else 0)
    
    exp1 = df["close"].ewm(span=12).mean()
    exp2 = df["close"].ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    scores.append(1 if hist.iloc[-1] > 0 else 0)
    
    min_p = df["close"].rolling(14).min()
    max_p = df["close"].rolling(14).max()
    k = 100 * (df["close"] - min_p) / (max_p - min_p)
    d = k.rolling(3).mean()
    scores.append(1 if k.iloc[-1] > d.iloc[-1] else 0)
    
    atr = (df["high"] - df["low"]).rolling(14).mean()
    atr_hist = atr.dropna().tolist()
    if atr_hist and len(atr_hist) > 5:
        atr_pct = stats.percentileofscore(atr_hist[-100:], atr.iloc[-1])
        scores.append(1 if atr_pct > 25 else 0)
    
    bb_std = df["close"].std()
    bb_upper = df["close"].mean() + (2 * bb_std)
    bb_lower = df["close"].mean() - (2 * bb_std)
    bb_pos = (df["close"].iloc[-1] - bb_lower) / (bb_upper - bb_lower)
    scores.append(1 if bb_pos > 0.5 else 0)
    
    ema20 = df["close"].ewm(span=20).mean()
    scores.append(1 if df["close"].iloc[-1] > ema20.iloc[-1] else 0)
    
    scores.append(1 if len(df) > 20 else 0)
    
    confluence = (sum(scores) / len(scores) * 100) if scores else 50.0
    return round(confluence, 1)

def calculate_dynamic_confidence(confluence: float, volatility: str, spread: float) -> float:
    conf_pt = (confluence / 100) * 35
    vol_pts = {"Expanding": 22, "Normal": 25, "Contracting": 15}.get(volatility, 20)
    
    if spread < 0.01:
        spread_pt = 20
    elif spread < 0.05:
        spread_pt = 15
    elif spread < 0.1:
        spread_pt = 10
    else:
        spread_pt = 3
    
    cons_pt = 15 if confluence > 60 else 8
    total = conf_pt + vol_pts + spread_pt + cons_pt
    return round(min(100, max(0, total)), 1)

async def generate_interpretation(symbol: str, confluence: float, confidence: float, bias: str) -> str:
    if not GEMINI_ENABLED:
        return f"{symbol}: {confidence}% clarity, {bias}"
    
    try:
        prompt = f"""Explain {symbol} in 2 sentences. Confluence: {confluence}%. Confidence: {confidence}%. Bias: {bias}. 
NO trading signals. Educational."""
        model = genai.GenerativeModel("gemini-pro")
        resp = model.generate_content(prompt, timeout=5)
        return resp.text
    except:
        return f"{symbol}: {confidence}% clarity"

# API ENDPOINTS
async def health(request):
    return web.json_response({"status": "ok", "version": "2.5+", "gemini": GEMINI_ENABLED})

async def receive_data(request):
    try:
        payload = await request.json()
        print(f"📥 Received data from Relay")
        print(f"   Payload keys: {list(payload.keys())}")
        
        market_data_list = payload.get("market_data", [])
        print(f"   Market data items: {len(market_data_list)}")
        
        for sym_data in market_data_list:
            sym = sym_data.get("symbol")
            print(f"   → Processing: {sym}")
            
            if sym not in market_data:
                print(f"      ⚠ Symbol {sym} not in our list, skipping")
                continue
            
            market_data[sym]["current"] = sym_data
            market_data[sym]["timestamp"] = datetime.now(timezone.utc)
            
            # Process timeframes
            for tf in sym_data.get("timeframes", []):
                for c in tf.get("candles", []):
                    market_data[sym]["candles"].append({
                        "time": c.get("t"),
                        "o": c.get("o"),
                        "h": c.get("h"),
                        "l": c.get("l"),
                        "c": c.get("c"),
                        "v": c.get("v")
                    })
            
            print(f"      ✓ {sym} stored, candles: {len(market_data[sym]['candles'])}")
        
        print(f"   ✓ Processing complete\n")
        return web.json_response({"status": "ok", "symbols": len(market_data_list)})
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        return web.json_response({"error": str(e)}, status=400)

async def analyze(sym: str) -> Dict:
    """Full analysis"""
    sym_data = market_data[sym]
    
    default = {
        "symbol": sym,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price": sym_data["current"].get("price", 0) if sym_data["current"] else 0,
        "market_regime": {"trend": "Unknown", "volatility": "Normal", "structure": "Unknown"},
        "bias": "NEUTRAL",
        "bias_stability": {"bias": "NEUTRAL", "active_since_minutes": 0, "last_flip_minutes_ago": None},
        "confluence_breakdown": {},
        "state_statistics": {"continuation": 45, "reversal": 35, "consolidation": 20},
        "current_session": get_current_session(),
        "confluence": 50.0,
        "confidence": 50.0,
        "interpretation": f"{sym}: Waiting for market data...",
    }
    
    if len(sym_data["candles"]) >= 5:
        try:
            df = pd.DataFrame(list(sym_data["candles"]))
            df["close"] = df["c"]
            df["high"] = df["h"]
            df["low"] = df["l"]
            df["volume"] = df["v"]
            df = df.sort_values("time").reset_index(drop=True)
            
            regime = calculate_market_regime(df)
            bias = calculate_bias_stability(df)
            confluence = calculate_dynamic_confluence(df)
            
            bid = sym_data["current"].get("bid", 0) if sym_data["current"] else 0
            ask = sym_data["current"].get("ask", 0) if sym_data["current"] else 0
            spread = ((ask - bid) / bid * 100) if bid else 0.05
            confidence = calculate_dynamic_confidence(confluence, regime["volatility"], spread)
            
            interp = await generate_interpretation(sym, confluence, confidence, bias["bias"])
            save_analysis(sym, confluence, confidence, interp)
            
            return {
                "symbol": sym,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "price": sym_data["current"].get("price", 0) if sym_data["current"] else 0,
                "market_regime": regime,
                "bias": bias,
                "bias_stability": bias,
                "confluence_breakdown": calculate_confluence_breakdown(df),
                "state_statistics": calculate_state_statistics(df),
                "current_session": get_current_session(),
                "confluence": confluence,
                "confidence": confidence,
                "interpretation": interp,
            }
        except Exception as e:
            print(f"   Analysis error for {sym}: {e}")
            return default
    
    return default

async def all_analysis(request):
    analyses = []
    for sym in market_data.keys():
        analysis = await analyze(sym)
        analyses.append(analysis)
    return web.json_response({"analyses": analyses})

async def symbol_analysis(request):
    sym = request.match_info.get("symbol", "").upper()
    if sym not in market_data:
        return web.json_response({"error": "Invalid symbol"}, status=404)
    return web.json_response(await analyze(sym))

# START
async def main():
    init_db()
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_post("/market-data-v1.6", receive_data)
    app.router.add_get("/latest-analysis", all_analysis)
    app.router.add_get("/analysis/{symbol}", symbol_analysis)
    
    port = int(os.getenv("PORT", 8080))
    
    print("\n" + "="*80)
    print("✨ ORACLEX V2.5+ - PYTHON BACKEND")
    print("="*80)
    print(f"✅ Server starting on port {port}")
    print("="*80 + "\n")
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    
    print(f"✅ Listening on port {port}\n")
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
