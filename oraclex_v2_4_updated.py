#!/usr/bin/env python3
"""
ORACLEX V2.5+ - Production Backend for Railway
Safe deployment with proper error handling
"""

import os
import json
import asyncio
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Dict, List
from collections import deque

try:
    from aiohttp import web
    import pandas as pd
    import numpy as np
    from scipy import stats
    import google.generativeai as genai
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    exit(1)

# ═════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("⚠️  GEMINI_API_KEY not set - AI interpretation will be disabled")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.error(f"❌ Gemini config error: {e}")
        GEMINI_API_KEY = None

SYMBOLS = ["XAUUSD", "XAGUUSD", "BTCUSD", "ETHUSD", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]

# ═════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═════════════════════════════════════════════════════════════════════════════

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect("oraclex_analysis.db")
    c = conn.cursor()
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS market_updates (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
            price REAL,
            confluence REAL,
            confidence REAL
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
            tf: {"candles": deque(maxlen=200), "indicators": {}}
            for tf in ["M1", "M5", "M15", "H1", "H4", "D1", "W1"]
        }
    }
    for symbol in SYMBOLS
}

# ═════════════════════════════════════════════════════════════════════════════
# INDICATOR STATE FUNCTIONS (Zero Hardcoded)
# ═════════════════════════════════════════════════════════════════════════════

def get_rsi_state(rsi_value: float) -> Dict:
    """RSI state"""
    if rsi_value >= 70:
        return {"state": "OVERBOUGHT", "value": rsi_value}
    elif rsi_value <= 30:
        return {"state": "OVERSOLD", "value": rsi_value}
    else:
        return {"state": "NEUTRAL", "value": rsi_value}

def get_macd_state(histogram: float) -> Dict:
    """MACD state"""
    if histogram > 0:
        return {"state": "BULLISH_CROSS", "value": histogram}
    elif histogram < 0:
        return {"state": "BEARISH_CROSS", "value": histogram}
    else:
        return {"state": "NEUTRAL", "value": histogram}

def get_bb_state(price: float, upper: float, lower: float) -> Dict:
    """Bollinger Bands position"""
    if price > upper * 0.95:
        return {"state": "UPPER_BAND", "value": price}
    elif price < lower * 1.05:
        return {"state": "LOWER_BAND", "value": price}
    else:
        return {"state": "MIDDLE", "value": price}

def get_atr_state(atr: float) -> Dict:
    """ATR state"""
    return {"state": "VOLATILITY", "value": atr}

def get_adx_state(adx: float) -> Dict:
    """ADX trend strength"""
    if adx > 25:
        return {"state": "TREND", "value": adx}
    else:
        return {"state": "NO_TREND", "value": adx}

# ═════════════════════════════════════════════════════════════════════════════
# CONFLUENCE CALCULATION
# ═════════════════════════════════════════════════════════════════════════════

def calculate_confluence(symbol_data: Dict) -> Dict:
    """Calculate multi-timeframe confluence"""
    try:
        timeframes = ["M1", "M5", "M15", "H1", "H4", "D1", "W1"]
        bullish_count = 0
        bearish_count = 0
        
        for tf in timeframes:
            tf_data = symbol_data["timeframes"][tf]
            if not tf_data["indicators"]:
                continue
            
            ind = tf_data["indicators"]
            
            # Simple scoring: count bullish indicators
            bullish = 0
            
            if ind.get("rsi", {}).get("state") in ["OVERBOUGHT", "NEUTRAL"]:
                bullish += 1
            if ind.get("macd", {}).get("state") == "BULLISH_CROSS":
                bullish += 1
            if ind.get("bb", {}).get("state") != "LOWER_BAND":
                bullish += 1
            if ind.get("adx", {}).get("state") == "TREND":
                bullish += 1
            
            if bullish >= 2:
                bullish_count += 1
            else:
                bearish_count += 1
        
        total = bullish_count + bearish_count if (bullish_count + bearish_count) > 0 else 1
        confluence = (bullish_count / total * 100) if total > 0 else 50
        
        return {
            "total_confluence": round(confluence, 1),
            "bullish_tfs": bullish_count,
            "bearish_tfs": bearish_count,
            "direction": "BULLISH" if confluence > 55 else "BEARISH" if confluence < 45 else "NEUTRAL"
        }
    except Exception as e:
        logger.error(f"Confluence error: {e}")
        return {"total_confluence": 50, "bullish_tfs": 0, "bearish_tfs": 0, "direction": "NEUTRAL"}

# ═════════════════════════════════════════════════════════════════════════════
# DYNAMIC CONFIDENCE
# ═════════════════════════════════════════════════════════════════════════════

def calculate_confidence(symbol_data: Dict, confluence: Dict) -> Dict:
    """Calculate confidence (NOT hardcoded 71%)"""
    try:
        base_confluence = confluence["total_confluence"]
        
        # Adjust for volatility
        atr = symbol_data["timeframes"]["M1"]["indicators"].get("atr", {}).get("value", 1)
        volatility_factor = min(1.0, max(0.7, 1.0 / (atr + 0.001)))
        
        # Adjust for spread
        bid = symbol_data.get("bid", 0)
        ask = symbol_data.get("ask", 0)
        spread_pct = ((ask - bid) / bid * 100) if bid else 0.05
        spread_factor = max(0.5, 1.0 - (spread_pct * 10))
        
        # Calculate final confidence
        confidence_score = (
            base_confluence * 0.5 +
            (volatility_factor * 100) * 0.3 +
            (spread_factor * 100) * 0.2
        )
        
        confidence_score = max(0, min(100, confidence_score))
        
        if confidence_score >= 80:
            interpretation = "VERY_CLEAR"
        elif confidence_score >= 65:
            interpretation = "CLEAR"
        elif confidence_score >= 50:
            interpretation = "MODERATE"
        else:
            interpretation = "AMBIGUOUS"
        
        return {
            "total_confidence": round(confidence_score, 1),
            "interpretation": interpretation
        }
    except Exception as e:
        logger.error(f"Confidence error: {e}")
        return {"total_confidence": 50, "interpretation": "NEUTRAL"}

# ═════════════════════════════════════════════════════════════════════════════
# GEMINI INTERPRETATION
# ═════════════════════════════════════════════════════════════════════════════

async def generate_interpretation(symbol: str, analysis: Dict) -> str:
    """Generate Gemini interpretation"""
    if not GEMINI_API_KEY:
        return f"Analysis at {analysis['confidence']['total_confidence']:.1f}% clarity"
    
    try:
        prompt = f"""Explain {symbol} technical analysis in 1-2 sentences. 
Confluence: {analysis['confluence']['total_confluence']}%
Direction: {analysis['confluence']['direction']}
Confidence: {analysis['confidence']['interpretation']}

NO trading advice. Just factual analysis."""
        
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.warning(f"Gemini error: {e}")
        return f"Clarity level: {analysis['confidence']['interpretation']}"

# ═════════════════════════════════════════════════════════════════════════════
# API HANDLERS
# ═════════════════════════════════════════════════════════════════════════════

async def handle_market_data(request):
    """Receive 7-TF data from MT5 EA"""
    try:
        data = await request.json()
        market_data = data.get("market_data", [])
        
        for symbol_data in market_data:
            symbol = symbol_data.get("symbol")
            if not symbol or symbol not in SYMBOLS:
                continue
            
            # Store current price
            market_data_cache[symbol]["current_price"] = symbol_data.get("price")
            market_data_cache[symbol]["bid"] = symbol_data.get("bid")
            market_data_cache[symbol]["ask"] = symbol_data.get("ask")
            
            # Process timeframes
            for tf_data in symbol_data.get("timeframes", []):
                tf = tf_data.get("timeframe")
                if tf not in market_data_cache[symbol]["timeframes"]:
                    continue
                
                # Store candles
                for c in tf_data.get("candles", []):
                    market_data_cache[symbol]["timeframes"][tf]["candles"].append(c)
                
                # Store indicators
                indicators = tf_data.get("indicators", {})
                if indicators:
                    market_data_cache[symbol]["timeframes"][tf]["indicators"] = {
                        "rsi": get_rsi_state(indicators.get("rsi", 50)),
                        "macd": get_macd_state(indicators.get("macd_histogram", 0)),
                        "bb": get_bb_state(
                            market_data_cache[symbol]["current_price"] or 0,
                            indicators.get("bb_upper", 0),
                            indicators.get("bb_lower", 0)
                        ),
                        "atr": get_atr_state(indicators.get("atr", 0)),
                        "adx": get_adx_state(indicators.get("adx", 0))
                    }
        
        return web.json_response({"status": "ok", "symbols": len(market_data)})
    
    except Exception as e:
        logger.error(f"Market data error: {e}")
        return web.json_response({"error": str(e)}, status=400)

async def handle_analysis(request):
    """GET /analysis/{symbol}"""
    try:
        symbol = request.match_info.get("symbol")
        if symbol not in SYMBOLS:
            return web.json_response({"error": "Symbol not found"}, status=404)
        
        symbol_data = market_data_cache[symbol]
        confluence = calculate_confluence(symbol_data)
        confidence = calculate_confidence(symbol_data, confluence)
        interpretation = await generate_interpretation(symbol, {
            "confluence": confluence,
            "confidence": confidence
        })
        
        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": symbol_data["current_price"],
            "confluence": confluence,
            "confidence": confidence,
            "interpretation": interpretation
        }
        
        return web.json_response(analysis)
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_all_analysis(request):
    """GET /latest-analysis"""
    try:
        all_analysis = []
        for symbol in SYMBOLS:
            symbol_data = market_data_cache[symbol]
            confluence = calculate_confluence(symbol_data)
            confidence = calculate_confidence(symbol_data, confluence)
            interpretation = await generate_interpretation(symbol, {
                "confluence": confluence,
                "confidence": confidence
            })
            
            all_analysis.append({
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "price": symbol_data["current_price"],
                "confluence": confluence,
                "confidence": confidence,
                "interpretation": interpretation
            })
        
        return web.json_response({"analyses": all_analysis})
    
    except Exception as e:
        logger.error(f"All analysis error: {e}")
        return web.json_response({"error": str(e)}, status=500)

# ═════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═════════════════════════════════════════════════════════════════════════════

async def handle_health(request):
    """GET / - Health check"""
    return web.json_response({
        "status": "ok",
        "version": "2.5+",
        "symbols": SYMBOLS,
        "has_gemini": bool(GEMINI_API_KEY)
    })

# ═════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

async def main():
    """Start application"""
    
    init_db()
    
    app = web.Application()
    
    # Routes
    app.router.add_get("/", handle_health)
    app.router.add_post("/market-data-v1.6", handle_market_data)
    app.router.add_get("/analysis/{symbol}", handle_analysis)
    app.router.add_get("/latest-analysis", handle_all_analysis)
    
    logger.info("=" * 80)
    logger.info("✨ ORACLEX V2.5+ - PRODUCTION BACKEND")
    logger.info("=" * 80)
    logger.info("📊 Receiving 7-TF data from MT5 EA V1.6")
    logger.info("🧮 Dynamic calculations (zero hardcoded values)")
    logger.info(f"🤖 Gemini AI: {'✅ Enabled' if GEMINI_API_KEY else '⚠️  Disabled'}")
    logger.info("📚 Educational framework - NO trading signals")
    logger.info("=" * 80)
    logger.info("🎧 Listening on http://0.0.0.0:8080")
    logger.info("  GET  / - Health check")
    logger.info("  POST /market-data-v1.6 - Receive EA data")
    logger.info("  GET  /latest-analysis - All symbols")
    logger.info("  GET  /analysis/{symbol} - Specific symbol")
    logger.info("=" * 80 + "\n")
    
    port = int(os.getenv("PORT", 8080))
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    
    logger.info(f"✅ Server started on port {port}")
    
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 OracleX V2.5+ stopped")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
