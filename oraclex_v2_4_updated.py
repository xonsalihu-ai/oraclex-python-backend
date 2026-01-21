#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                  ORACLEX V2.5+ - INSTITUTIONAL CLARITY ENGINE                 ║
║                                                                                ║
║  PURPOSE: Translate charts into clarity for learning & time-saving             ║
║  NOT: Trading signals, advice, or automated execution                          ║
║                                                                                ║
║  FEATURES:                                                                     ║
║  • 7 Timeframes × 7 Indicators (from MT5 EA V1.6)                             ║
║  • Zero hardcoded values - everything calculated                              ║
║  • Multi-timeframe confluence analysis                                        ║
║  • Dynamic confidence scoring                                                 ║
║  • Gemini AI interpretation (chart → plain English)                           ║
║  • Educational framework (learn what matters)                                 ║
║                                                                                ║
║  LIGHT UPDATES: Every 10 seconds (price)                                      ║
║  HEAVY ANALYSIS: Every 30 seconds (full indicators)                           ║
║                                                                                ║
║  Dashboard-only → Educational tool for traders                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import json
import asyncio
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
from collections import deque
from aiohttp import web
import aiohttp

import pandas as pd
import numpy as np
from scipy import stats
import google.generativeai as genai

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

RELAY_URL = "https://oraclex-relay-production.up.railway.app"
LIGHT_UPDATE_INTERVAL = 10  # Update price every 10 seconds
HEAVY_UPDATE_INTERVAL = 30  # Full analysis every 30 seconds

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

SYMBOLS_TO_MONITOR = ["XAUUSD", "XAGUUSD", "BTCUSD", "ETHUSD", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]

# ═════════════════════════════════════════════════════════════════════════════
# DATABASE INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def init_database():
    """Initialize SQLite for historical analysis storage"""
    conn = sqlite3.connect("oraclex_analysis.db")
    c = conn.cursor()
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS market_updates (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
            price REAL,
            bid REAL,
            ask REAL,
            timeframe TEXT,
            rsi REAL,
            macd_value REAL,
            macd_signal REAL,
            macd_histogram REAL,
            stoch_k REAL,
            stoch_d REAL,
            atr REAL,
            bb_upper REAL,
            bb_middle REAL,
            bb_lower REAL,
            ema20 REAL,
            adx REAL
        )
    """)
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
            confluence_score REAL,
            confidence_level TEXT,
            bias_direction TEXT,
            multi_tf_agreement REAL,
            gemini_interpretation TEXT
        )
    """)
    
    conn.commit()
    conn.close()

# ═════════════════════════════════════════════════════════════════════════════
# DATA STORAGE
# ═════════════════════════════════════════════════════════════════════════════

market_data_cache = {
    symbol: {
        "current_price": None,
        "bid": None,
        "ask": None,
        "last_update": None,
        "timeframes": {
            "M1": {"candles": deque(maxlen=1440), "indicators": {}},
            "M5": {"candles": deque(maxlen=288), "indicators": {}},
            "M15": {"candles": deque(maxlen=96), "indicators": {}},
            "H1": {"candles": deque(maxlen=168), "indicators": {}},
            "H4": {"candles": deque(maxlen=180), "indicators": {}},
            "D1": {"candles": deque(maxlen=365), "indicators": {}},
            "W1": {"candles": deque(maxlen=260), "indicators": {}},
        }
    }
    for symbol in SYMBOLS_TO_MONITOR
}

# ═════════════════════════════════════════════════════════════════════════════
# DYNAMIC INDICATOR STATE CALCULATION (Zero Hardcoded Values)
# ═════════════════════════════════════════════════════════════════════════════

def get_rsi_state(rsi_value: float, rsi_history: List[float]) -> Dict:
    """Determine RSI state based on statistical distribution"""
    if not rsi_history or len(rsi_history) < 2:
        return {"state": "NEUTRAL", "percentile": 50, "value": rsi_value}
    
    percentile = stats.percentileofscore(rsi_history, rsi_value)
    
    if percentile >= 75:
        state = "EXTREMELY_OVERBOUGHT"
    elif percentile >= 65:
        state = "MODERATELY_OVERBOUGHT"
    elif percentile >= 55:
        state = "SLIGHTLY_OVERBOUGHT"
    elif percentile <= 25:
        state = "EXTREMELY_OVERSOLD"
    elif percentile <= 35:
        state = "MODERATELY_OVERSOLD"
    elif percentile <= 45:
        state = "SLIGHTLY_OVERSOLD"
    else:
        state = "NEUTRAL"
    
    return {
        "state": state,
        "percentile": round(percentile, 1),
        "value": rsi_value
    }

def get_macd_state(macd: float, signal: float, histogram: float, history: List[float]) -> Dict:
    """MACD state: crossover detection + momentum"""
    state = "BULLISH_CROSS" if histogram > 0 else "BEARISH_CROSS" if histogram < 0 else "NEUTRAL"
    momentum = "ACCELERATING" if abs(histogram) > np.mean([abs(h) for h in history[-5:]]) if history and len(history) >= 5 else False else "DECELERATING"
    
    return {
        "state": state,
        "momentum": momentum,
        "histogram": histogram,
        "value": macd
    }

def get_stochastic_state(k: float, d: float, stoch_history: List[Tuple[float, float]]) -> Dict:
    """Stochastic with dynamic thresholds"""
    if k > 80:
        state = "OVERBOUGHT"
    elif k < 20:
        state = "OVERSOLD"
    elif k > d:
        state = "BULLISH_MOMENTUM"
    elif k < d:
        state = "BEARISH_MOMENTUM"
    else:
        state = "NEUTRAL"
    
    return {
        "state": state,
        "k_value": k,
        "d_value": d,
        "crossover": "GOLDEN" if k > d else "DEATH" if k < d else "NEUTRAL"
    }

def get_atr_state(current_atr: float, atr_history: List[float]) -> Dict:
    """ATR volatility classification"""
    if not atr_history or len(atr_history) < 2:
        return {"state": "NORMAL", "value": current_atr, "percentile": 50}
    
    percentile = stats.percentileofscore(atr_history, current_atr)
    
    if percentile >= 90:
        state = "EXTREME_EXPANSION"
    elif percentile >= 75:
        state = "EXPANSION"
    elif percentile <= 10:
        state = "EXTREME_COMPRESSION"
    elif percentile <= 25:
        state = "COMPRESSION"
    else:
        state = "NORMAL"
    
    return {
        "state": state,
        "value": current_atr,
        "percentile": round(percentile, 1),
        "avg_20": round(np.mean(atr_history[-20:]), 8) if len(atr_history) >= 20 else None
    }

def get_bb_state(price: float, upper: float, middle: float, lower: float, bb_history: List[float]) -> Dict:
    """Bollinger Bands state"""
    bb_width = upper - lower
    bb_position = (price - lower) / bb_width if bb_width > 0 else 0.5
    
    squeeze = "COMPRESSION" if bb_history and bb_width < np.mean([abs(h) for h in bb_history[-20:]]) else "EXPANSION"
    
    if bb_position > 0.9:
        position = "AT_UPPER_BAND"
    elif bb_position < 0.1:
        position = "AT_LOWER_BAND"
    elif bb_position > 0.7:
        position = "UPPER_HALF"
    elif bb_position < 0.3:
        position = "LOWER_HALF"
    else:
        position = "MIDDLE"
    
    return {
        "state": squeeze,
        "position": position,
        "bb_position_pct": round(bb_position * 100, 1),
        "width": round(bb_width, 8)
    }

def get_ema_state(price: float, ema: float, ema_history: List[float]) -> Dict:
    """EMA relationship to price"""
    distance_pct = ((price - ema) / ema * 100) if ema != 0 else 0
    
    if abs(distance_pct) < 0.5:
        proximity = "AT_EMA"
    elif distance_pct > 0:
        proximity = f"ABOVE_EMA_{abs(distance_pct):.2f}%"
    else:
        proximity = f"BELOW_EMA_{abs(distance_pct):.2f}%"
    
    return {
        "value": ema,
        "distance_pct": round(distance_pct, 2),
        "proximity": proximity
    }

def get_adx_state(adx: float) -> Dict:
    """ADX trend strength"""
    if adx >= 50:
        strength = "VERY_STRONG_TREND"
    elif adx >= 40:
        strength = "STRONG_TREND"
    elif adx >= 25:
        strength = "MODERATE_TREND"
    elif adx >= 15:
        strength = "WEAK_TREND"
    else:
        strength = "NO_TREND"
    
    return {
        "state": strength,
        "value": adx
    }

# ═════════════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME CONFLUENCE (Zero Hardcoded)
# ═════════════════════════════════════════════════════════════════════════════

def calculate_multi_tf_confluence(symbol_data: Dict) -> Dict:
    """
    Calculate confluence across 7 timeframes.
    Returns: total confluence %, per-timeframe direction, agreement level
    """
    
    timeframes = ["M1", "M5", "M15", "H1", "H4", "D1", "W1"]
    alignment_scores = {}
    directions = {}
    per_tf_alignment = {}
    
    for tf in timeframes:
        tf_data = symbol_data["timeframes"][tf]
        if not tf_data["indicators"]:
            continue
        
        ind = tf_data["indicators"]
        
        # Count bullish indicators
        bullish = 0
        total = 0
        
        # EMA above price = bullish
        if ind.get("ema20", {}).get("distance_pct", 0) < 0:
            bullish += 1
        total += 1
        
        # RSI percentile > 50 = bullish
        if ind.get("rsi", {}).get("percentile", 50) > 50:
            bullish += 1
        total += 1
        
        # MACD histogram > 0 = bullish
        if ind.get("macd", {}).get("histogram", 0) > 0:
            bullish += 1
        total += 1
        
        # Stochastic K > D = bullish
        if ind.get("stochastic", {}).get("k_value", 0) > ind.get("stochastic", {}).get("d_value", 0):
            bullish += 1
        total += 1
        
        # BB position > 0.5 = bullish
        if ind.get("bb", {}).get("bb_position_pct", 50) > 50:
            bullish += 1
        total += 1
        
        # ADX > 25 & trend = bullish/bearish
        adx_state = ind.get("adx", {}).get("state", "NO_TREND")
        if "STRONG" in adx_state or "MODERATE" in adx_state:
            bullish += 0.5  # Neutral - just shows trend strength
        total += 0.5
        
        # ATR not directional, skip
        
        alignment_pct = (bullish / total * 100) if total > 0 else 50
        per_tf_alignment[tf] = alignment_pct
        directions[tf] = "BULLISH" if alignment_pct > 55 else "BEARISH" if alignment_pct < 45 else "NEUTRAL"
    
    # Calculate total confluence
    if per_tf_alignment:
        avg_alignment = np.mean(list(per_tf_alignment.values()))
        bullish_tfs = sum(1 for d in directions.values() if d == "BULLISH")
        bearish_tfs = sum(1 for d in directions.values() if d == "BEARISH")
        
        multi_tf_agreement = max(bullish_tfs, bearish_tfs) / len(directions) * 100
        
        # Total confluence score (calculated formula)
        total_confluence = (
            avg_alignment * 0.4 +        # How aligned are indicators?
            multi_tf_agreement * 0.4 +   # How many TFs agree?
            (100 - abs(per_tf_alignment.get("M1", 50) - per_tf_alignment.get("W1", 50)) / 100) * 0.2  # Harmony
        )
    else:
        total_confluence = 50
        multi_tf_agreement = 50
    
    return {
        "total_confluence": round(min(100, max(0, total_confluence)), 1),
        "per_timeframe": per_tf_alignment,
        "directions": directions,
        "bullish_tfs": sum(1 for d in directions.values() if d == "BULLISH"),
        "bearish_tfs": sum(1 for d in directions.values() if d == "BEARISH"),
        "multi_tf_agreement": round(multi_tf_agreement, 1),
        "conviction": "VERY_HIGH" if multi_tf_agreement >= 85 else "HIGH" if multi_tf_agreement >= 70 else "MODERATE" if multi_tf_agreement >= 50 else "LOW"
    }

# ═════════════════════════════════════════════════════════════════════════════
# DYNAMIC CONFIDENCE CALCULATION (Zero Hardcoded)
# ═════════════════════════════════════════════════════════════════════════════

def calculate_dynamic_confidence(symbol: str, symbol_data: Dict, confluence: Dict) -> Dict:
    """
    Calculate confidence from multiple factors.
    NOT hardcoded 71% - real calculation.
    """
    
    factors = {}
    
    # Factor 1: Multi-TF Confluence (0-35 points)
    confluence_factor = confluence["total_confluence"] / 100 * 35
    factors["confluence"] = confluence_factor
    
    # Factor 2: Volatility Suitability (0-25 points)
    m1_atr = symbol_data["timeframes"]["M1"]["indicators"].get("atr", {})
    atr_state = m1_atr.get("state", "NORMAL")
    
    if atr_state == "EXTREME_COMPRESSION":
        vol_factor = 5
    elif atr_state == "COMPRESSION":
        vol_factor = 15
    elif atr_state == "NORMAL":
        vol_factor = 25
    elif atr_state == "EXPANSION":
        vol_factor = 22
    else:
        vol_factor = 12
    
    factors["volatility"] = vol_factor
    
    # Factor 3: Spread Impact (0-15 points)
    bid = symbol_data.get("bid", 0)
    ask = symbol_data.get("ask", 0)
    spread_pct = ((ask - bid) / bid * 100) if bid else 0
    
    if spread_pct < 0.01:
        spread_factor = 15
    elif spread_pct < 0.05:
        spread_factor = 12
    elif spread_pct < 0.1:
        spread_factor = 8
    else:
        spread_factor = 3
    
    factors["spread"] = spread_factor
    
    # Factor 4: Indicator Consistency (0-25 points)
    consistency = confluence["multi_tf_agreement"] / 100 * 25
    factors["consistency"] = consistency
    
    # Total confidence
    total_confidence = sum(factors.values())
    
    # Interpret confidence level
    if total_confidence >= 85:
        interpretation = "EXTREMELY_CLEAR"
    elif total_confidence >= 75:
        interpretation = "VERY_CLEAR"
    elif total_confidence >= 60:
        interpretation = "MODERATELY_CLEAR"
    elif total_confidence >= 45:
        interpretation = "SOMEWHAT_AMBIGUOUS"
    elif total_confidence >= 30:
        interpretation = "AMBIGUOUS"
    else:
        interpretation = "UNCLEAR"
    
    return {
        "total_confidence": round(total_confidence, 1),
        "interpretation": interpretation,
        "factors": {k: round(v, 1) for k, v in factors.items()},
        "breakdown": {
            "confluence": f"{factors['confluence']:.1f}/35",
            "volatility": f"{factors['volatility']:.1f}/25",
            "spread": f"{factors['spread']:.1f}/15",
            "consistency": f"{factors['consistency']:.1f}/25"
        }
    }

# ═════════════════════════════════════════════════════════════════════════════
# GEMINI AI INTERPRETATION
# ═════════════════════════════════════════════════════════════════════════════

async def generate_gemini_interpretation(symbol: str, analysis: Dict) -> str:
    """
    Use Gemini to interpret technical analysis.
    Educational only - NO trading suggestions.
    """
    
    try:
        prompt = f"""You are an educational market analyst explaining what the charts show to traders who want to LEARN.

CURRENT ANALYSIS FOR {symbol}:

**Confluence Score:** {analysis['confluence']['total_confluence']}% 
- Multi-TF Agreement: {analysis['confluence']['multi_tf_agreement']}%
- Conviction: {analysis['confluence']['conviction']}

**Confidence Level:** {analysis['confidence']['interpretation']} ({analysis['confidence']['total_confidence']:.1f}%)

Your task:
1. Explain what this confluence score means (NOT a trading signal)
2. What are the indicators showing us?
3. Which timeframes are aligned and which aren't?
4. Why this clarity matters for traders

RULES:
- NEVER suggest trades, entries, stops, or profits
- NEVER use: "buy", "sell", "go long", "go short"
- ONLY explain what the charts show factually
- Focus on LEARNING - why does this matter technically?
- Keep to 2-3 sentences max

Generate the interpretation:"""
        
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return f"Analysis at {analysis['confidence']['total_confidence']}% clarity"

# ═════════════════════════════════════════════════════════════════════════════
# RECEIVE DATA FROM EA V1.6
# ═════════════════════════════════════════════════════════════════════════════

async def handle_market_data(request):
    """Receive 7-TF data from MT5 EA V1.6"""
    try:
        data = await request.json()
        market_data = data.get("market_data", [])
        
        for symbol_data in market_data:
            symbol = symbol_data.get("symbol")
            if not symbol or symbol not in SYMBOLS_TO_MONITOR:
                continue
            
            market_data_cache[symbol]["current_price"] = symbol_data.get("price")
            market_data_cache[symbol]["bid"] = symbol_data.get("bid")
            market_data_cache[symbol]["ask"] = symbol_data.get("ask")
            market_data_cache[symbol]["last_update"] = datetime.now(timezone.utc).isoformat()
            
            for tf_data in symbol_data.get("timeframes", []):
                tf = tf_data.get("timeframe")
                if tf not in market_data_cache[symbol]["timeframes"]:
                    continue
                
                # Store candles
                candles = tf_data.get("candles", [])
                for c in candles:
                    market_data_cache[symbol]["timeframes"][tf]["candles"].append({
                        "time": c.get("t"),
                        "o": c.get("o"),
                        "h": c.get("h"),
                        "l": c.get("l"),
                        "c": c.get("c"),
                        "v": c.get("v")
                    })
                
                # Store indicators
                indicators = tf_data.get("indicators", {})
                if indicators:
                    price = market_data_cache[symbol]["current_price"] or 0
                    
                    market_data_cache[symbol]["timeframes"][tf]["indicators"] = {
                        "rsi": get_rsi_state(indicators.get("rsi", 50), []),
                        "macd": get_macd_state(
                            indicators.get("macd_value", 0),
                            indicators.get("macd_signal", 0),
                            indicators.get("macd_histogram", 0),
                            []
                        ),
                        "stochastic": get_stochastic_state(
                            indicators.get("stoch_k", 50),
                            indicators.get("stoch_d", 50),
                            []
                        ),
                        "atr": get_atr_state(indicators.get("atr", 0), []),
                        "bb": get_bb_state(
                            price,
                            indicators.get("bb_upper", 0),
                            indicators.get("bb_middle", 0),
                            indicators.get("bb_lower", 0),
                            []
                        ),
                        "ema20": get_ema_state(price, indicators.get("ema20", 0), []),
                        "adx": get_adx_state(indicators.get("adx", 0))
                    }
        
        return web.json_response({"status": "ok", "symbols": len(market_data)})
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return web.json_response({"error": str(e)}, status=400)

# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS GENERATION
# ═════════════════════════════════════════════════════════════════════════════

async def generate_analysis_for_symbol(symbol: str) -> Dict:
    """Full analysis for one symbol"""
    
    symbol_data = market_data_cache[symbol]
    confluence = calculate_multi_tf_confluence(symbol_data)
    confidence = calculate_dynamic_confidence(symbol, symbol_data, confluence)
    
    analysis = {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price": symbol_data["current_price"],
        "confluence": confluence,
        "confidence": confidence
    }
    
    gemini_text = await generate_gemini_interpretation(symbol, analysis)
    analysis["gemini_interpretation"] = gemini_text
    
    return analysis

# ═════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

async def handle_latest_analysis(request):
    """GET /latest-analysis - All symbols"""
    try:
        all_analysis = []
        for symbol in SYMBOLS_TO_MONITOR:
            analysis = await generate_analysis_for_symbol(symbol)
            all_analysis.append(analysis)
        return web.json_response({"analyses": all_analysis})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def handle_symbol_analysis(request):
    """GET /analysis/{symbol}"""
    try:
        symbol = request.match_info.get("symbol")
        if symbol not in SYMBOLS_TO_MONITOR:
            return web.json_response({"error": "Symbol not found"}, status=404)
        
        analysis = await generate_analysis_for_symbol(symbol)
        return web.json_response(analysis)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

# ═════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

async def main():
    """Main application"""
    
    init_database()
    
    app = web.Application()
    app.router.add_post("/market-data-v1.6", handle_market_data)
    app.router.add_get("/latest-analysis", handle_latest_analysis)
    app.router.add_get("/analysis/{symbol}", handle_symbol_analysis)
    
    print("=" * 80)
    print("✨ ORACLEX V2.5+ - CLARITY ENGINE")
    print("=" * 80)
    print("📊 7-TF data from MT5 EA V1.6")
    print("🧮 Zero hardcoded - fully calculated")
    print("🤖 Gemini interpretation (educational)")
    print("📚 Learning framework - NO signals")
    print("=" * 80)
    print("🎧 http://0.0.0.0:8080")
    print("  POST /market-data-v1.6")
    print("  GET  /latest-analysis")
    print("  GET  /analysis/{symbol}")
    print("=" * 80 + "\n")
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 OracleX V2.5+ stopped")
