#!/usr/bin/env python3
"""
ORACLEX V2.5+ - SIMPLIFIED
Direct processing of Relay forwarded data
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

# Symbol list
SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]

# Data storage - store ALL incoming data
all_market_data = {}

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

def calculate_market_regime(df: pd.DataFrame) -> Dict:
    if len(df) < 20:
        return {"trend": "Unknown", "volatility": "Normal", "structure": "Unknown"}
    
    try:
        ema_50 = df["close"].ewm(span=50).mean()
        ema_200 = df["close"].ewm(span=200).mean()
        
        ratio = ema_50.iloc[-1] / ema_200.iloc[-1]
        trend = "Strong" if ratio > 1.01 else ("Weak" if ratio < 0.99 else "Ranging")
        
        bb_std = df["close"].std()
        bb_width = 2 * bb_std / df["close"].mean() if df["close"].mean() != 0 else 0
        volatility = "Expanding" if bb_width > 0.04 else ("Contracting" if bb_width < 0.015 else "Normal")
        
        structure = "Clean" if len(df) < 20 else ("Choppy" if len(df) > 5 else "Unknown")
        
        return {"trend": trend, "volatility": volatility, "structure": structure}
    except:
        return {"trend": "Unknown", "volatility": "Normal", "structure": "Unknown"}

def calculate_dynamic_confluence(df: pd.DataFrame) -> float:
    if len(df) < 10:
        return 50.0
    
    try:
        scores = []
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        scores.append(1 if rsi.iloc[-1] > 50 else 0)
        
        # MACD
        exp1 = df["close"].ewm(span=12).mean()
        exp2 = df["close"].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        hist = macd - signal
        scores.append(1 if hist.iloc[-1] > 0 else 0)
        
        # Stochastic
        min_p = df["close"].rolling(14).min()
        max_p = df["close"].rolling(14).max()
        k = 100 * (df["close"] - min_p) / (max_p - min_p)
        d = k.rolling(3).mean()
        scores.append(1 if k.iloc[-1] > d.iloc[-1] else 0)
        
        # ATR
        scores.append(1)
        
        # BB
        bb_std = df["close"].std()
        bb_upper = df["close"].mean() + (2 * bb_std)
        bb_lower = df["close"].mean() - (2 * bb_std)
        if bb_upper - bb_lower != 0:
            bb_pos = (df["close"].iloc[-1] - bb_lower) / (bb_upper - bb_lower)
            scores.append(1 if bb_pos > 0.5 else 0)
        
        # EMA20
        ema20 = df["close"].ewm(span=20).mean()
        scores.append(1 if df["close"].iloc[-1] > ema20.iloc[-1] else 0)
        
        # ADX
        scores.append(1)
        
        confluence = (sum(scores) / max(len(scores), 1) * 100)
        return round(confluence, 1)
    except:
        return 50.0

def calculate_dynamic_confidence(confluence: float) -> float:
    try:
        conf_pt = (confluence / 100) * 35
        vol_pts = 25
        spread_pt = 15
        cons_pt = 15 if confluence > 60 else 8
        total = conf_pt + vol_pts + spread_pt + cons_pt
        return round(min(100, max(0, total)), 1)
    except:
        return 50.0

async def generate_interpretation(symbol: str, confluence: float, confidence: float, bias: str) -> str:
    if not GEMINI_ENABLED:
        return f"{symbol}: {confidence}% confidence"
    
    try:
        prompt = f"In 1 sentence: {symbol} shows {confidence}% confidence with {confluence}% confluence. Bias: {bias}. Educational only."
        model = genai.GenerativeModel("gemini-pro")
        resp = model.generate_content(prompt, timeout=5)
        return resp.text[:100]
    except:
        return f"{symbol}: {confidence}% confidence"

# API ENDPOINTS
async def health(request):
    return web.json_response({"status": "ok", "version": "2.5+", "symbols_cached": len(all_market_data)})

async def receive_data(request):
    """Receive data from Relay"""
    try:
        payload = await request.json()
        market_data_list = payload.get("market_data", [])
        
        print(f"\n📥 PYTHON RECEIVED DATA FROM RELAY")
        print(f"   Market data items: {len(market_data_list)}")
        
        for sym_data in market_data_list:
            sym = sym_data.get("symbol", "").upper()
            print(f"   → {sym}")
            
            if sym not in SYMBOLS:
                print(f"      ⚠ Unknown symbol")
                continue
            
            # Store ALL of it
            all_market_data[sym] = sym_data
            print(f"      ✓ Stored")
        
        print(f"   ✓ Total cached: {len(all_market_data)}\n")
        
        return web.json_response({"status": "ok", "received": len(market_data_list)})
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        return web.json_response({"error": str(e)}, status=400)

async def analyze(sym: str) -> Dict:
    """Analyze symbol - use cached data"""
    
    # Default response
    default = {
        "symbol": sym,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price": 0,
        "market_regime": {"trend": "Unknown", "volatility": "Normal", "structure": "Unknown"},
        "bias": "NEUTRAL",
        "confluence": 50.0,
        "confidence": 50.0,
        "interpretation": "Waiting for data...",
    }
    
    # If we don't have data for this symbol, return default
    if sym not in all_market_data:
        return default
    
    sym_data = all_market_data[sym]
    
    try:
        # Get price
        price = sym_data.get("price", 0)
        
        # Get H1 candles (most recent timeframe with good data)
        candles_list = []
        for tf in sym_data.get("timeframes", []):
            if tf.get("timeframe") == "H1":
                candles_list = tf.get("candles", [])
                break
        
        if not candles_list or len(candles_list) < 5:
            return default
        
        # Build dataframe
        df_data = []
        for c in candles_list:
            df_data.append({
                "close": c.get("c", 0),
                "high": c.get("h", 0),
                "low": c.get("l", 0),
                "volume": c.get("v", 0)
            })
        
        df = pd.DataFrame(df_data)
        
        if len(df) < 5:
            return default
        
        # Calculate
        regime = calculate_market_regime(df)
        confluence = calculate_dynamic_confluence(df)
        confidence = calculate_dynamic_confidence(confluence)
        
        bias = "BULLISH" if df["close"].iloc[-1] > df["close"].mean() else "BEARISH"
        interp = await generate_interpretation(sym, confluence, confidence, bias)
        
        save_analysis(sym, confluence, confidence, interp)
        
        return {
            "symbol": sym,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": price,
            "market_regime": regime,
            "bias": bias,
            "confluence": confluence,
            "confidence": confidence,
            "interpretation": interp,
        }
    
    except Exception as e:
        print(f"   Analysis error for {sym}: {e}")
        return default

async def all_analysis(request):
    """Return analysis for all symbols"""
    analyses = []
    for sym in SYMBOLS:
        analysis = await analyze(sym)
        analyses.append(analysis)
    return web.json_response({"analyses": analyses})

async def symbol_analysis(request):
    sym = request.match_info.get("symbol", "").upper()
    if sym not in SYMBOLS:
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
    print("✨ ORACLEX V2.5+ - PYTHON BACKEND (SIMPLIFIED)")
    print("="*80)
    print(f"✅ Listening on port {port}")
    print(f"✅ Symbols: {', '.join(SYMBOLS)}")
    print("="*80 + "\n")
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
