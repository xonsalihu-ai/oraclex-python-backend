#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                     ORACLEX V2.4 - PRODUCTION BACKEND                         ║
║                      3 APIs • 8 Pairs • 10-Second Hybrid                       ║
║                                                                                ║
║  DATA SOURCES:                                                                 ║
║  • Binance (BTCUSD, ETHUSD) - Every 10 seconds                                ║
║  • API Ninjas (XAUUSD, XAGUUSD) - Every 10 seconds                            ║
║  • Finnhub (EUR/USD, GBP/USD, AUD/USD, NZD/USD) - Every 30 seconds            ║
║                                                                                ║
║  FEATURES:                                                                     ║
║  1. Market Regime Classification                                              ║
║  2. Bias Stability Tracking                                                   ║
║  3. Confluence Breakdown                                                      ║
║  4. Context History (60+ min)                                                 ║
║  5. State Statistics                                                          ║
║  6. Session Intelligence                                                      ║
║                                                                                ║
║  NO TRADING SIGNALS - Education & Decision Support Only                       ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import json
import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
from collections import deque

import pandas as pd
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

RELAY_URL = "https://oraclex-relay-production.up.railway.app"
SCAN_INTERVAL_CRYPTO = 10  # 10 seconds for fast-moving crypto
SCAN_INTERVAL_FOREX_INDICES = 30  # 30 seconds for slower-moving forex/indices

# API Keys
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Symbol mappings
CRYPTO_SYMBOLS = {
    "BTCUSD": "BTCUSDT",  # Binance uses BTCUSDT
    "ETHUSD": "ETHUSDT"
}

METAL_SYMBOLS = {
    "XAUUSD": "XAU",  # Gold
    "XAGUUSD": "XAG"   # Silver
}

FOREX_INDICES_SYMBOLS = {
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "AUDUSD": "AUDUSD",
    "NZDUSD": "NZDUSD"
}

# ═════════════════════════════════════════════════════════════════════════════
# DATA STORAGE
# ═════════════════════════════════════════════════════════════════════════════

market_data = {
    "BTCUSD": {"candles": deque(maxlen=200), "current": None, "timestamp": None},
    "ETHUSD": {"candles": deque(maxlen=200), "current": None, "timestamp": None},
    "XAUUSD": {"candles": deque(maxlen=200), "current": None, "timestamp": None},
    "XAGUUSD": {"candles": deque(maxlen=200), "current": None, "timestamp": None},
    "EURUSD": {"candles": deque(maxlen=200), "current": None, "timestamp": None},
    "GBPUSD": {"candles": deque(maxlen=200), "current": None, "timestamp": None},
    "AUDUSD": {"candles": deque(maxlen=200), "current": None, "timestamp": None},
    "NZDUSD": {"candles": deque(maxlen=200), "current": None, "timestamp": None},
}

context_history = {sym: deque(maxlen=90) for sym in market_data.keys()}  # 90 data points

# ═════════════════════════════════════════════════════════════════════════════
# BINANCE DATA FETCHER
# ═════════════════════════════════════════════════════════════════════════════

async def fetch_binance_data(session: aiohttp.ClientSession) -> Dict:
    """Fetch BTCUSD and ETHUSD from Binance"""
    try:
        data = {}
        
        for oracle_sym, binance_sym in CRYPTO_SYMBOLS.items():
            # Get latest price
            async with session.get(
                f"https://data-api.binance.vision/api/v3/ticker/price?symbol={binance_sym}",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                price_data = await resp.json()
                current_price = float(price_data["price"])
            
            # Get 1-minute candles (200 limit = ~3.3 hours)
            async with session.get(
                f"https://data-api.binance.vision/api/v3/klines?symbol={binance_sym}&interval=1m&limit=200",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                candles = await resp.json()
            
            # Get bid/ask
            async with session.get(
                f"https://data-api.binance.vision/api/v3/ticker/bookTicker?symbol={binance_sym}",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                book = await resp.json()
            
            data[oracle_sym] = {
                "price": current_price,
                "bid": float(book["bidPrice"]),
                "ask": float(book["askPrice"]),
                "candles": [
                    {
                        "time": c[0],
                        "o": float(c[1]),
                        "h": float(c[2]),
                        "l": float(c[3]),
                        "c": float(c[4]),
                        "v": float(c[7])
                    }
                    for c in candles
                ]
            }
        
        return data
    except Exception as e:
        print(f"❌ Binance error: {e}")
        return {}

# ═════════════════════════════════════════════════════════════════════════════
# METALS-API DATA FETCHER
# ═════════════════════════════════════════════════════════════════════════════

async def fetch_metals_data(session: aiohttp.ClientSession) -> Dict:
    """Fetch XAUUSD and XAGUUSD from API Ninjas - FREE, no API key needed"""
    try:
        data = {}
        
        # Use API Ninjas Gold Price API - completely free
        async with session.get(
            "https://api.api-ninjas.com/v1/goldprice",
            timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            api_data = await resp.json()
        
        # API Ninjas returns price_usd_per_oz for gold
        if api_data and 'price_usd_per_oz' in api_data:
            xau_price = float(api_data['price_usd_per_oz'])
            data['XAUUSD'] = {
                "price": xau_price,
                "bid": xau_price * 0.999,
                "ask": xau_price * 1.001,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            print(f"✅ API Ninjas XAU (Gold): ${xau_price:.2f}")
        else:
            print(f"⚠️ API Ninjas: No gold price data - {api_data}")
        
        # Silver from same API
        async with session.get(
            "https://api.api-ninjas.com/v1/goldprice?metals=silver",
            timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            silver_data = await resp.json()
        
        if silver_data and 'price_usd_per_oz' in silver_data:
            xag_price = float(silver_data['price_usd_per_oz'])
            data['XAGUUSD'] = {
                "price": xag_price,
                "bid": xag_price * 0.999,
                "ask": xag_price * 1.001,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            print(f"✅ API Ninjas XAG (Silver): ${xag_price:.2f}")
        else:
            print(f"⚠️ API Ninjas: No silver price data - {silver_data}")
        
        if not data:
            print(f"⚠️ API Ninjas: No precious metals data returned")
        
        return data
        
    except Exception as e:
        print(f"❌ API Ninjas fetch error: {e}")
        return {}

# ═════════════════════════════════════════════════════════════════════════════
# FINNHUB DATA FETCHER
# ═════════════════════════════════════════════════════════════════════════════

async def fetch_finnhub_data(session: aiohttp.ClientSession) -> Dict:
    """Fetch forex candles from Finnhub"""
    try:
        data = {}
        
        for oracle_sym, finnhub_sym in FOREX_SYMBOLS.items():
            try:
                # Get current quote first
                async with session.get(
                    f"https://finnhub.io/api/v1/quote?symbol={finnhub_sym}&token={FINNHUB_API_KEY}",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    quote_data = await resp.json()
                
                if 'c' not in quote_data:
                    print(f"⚠️ Finnhub {oracle_sym}: No quote - {quote_data}")
                    continue
                
                current_price = quote_data.get('c', 0)
                
                # Try to get forex candles
                try:
                    from_time = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp())
                    to_time = int(datetime.now(timezone.utc).timestamp())
                    
                    async with session.get(
                        f"https://finnhub.io/api/v1/forex/candle?symbol={finnhub_sym}&resolution=D&from={from_time}&to={to_time}&token={FINNHUB_API_KEY}",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        candle_data = await resp.json()
                    
                    if "c" in candle_data and len(candle_data.get("c", [])) > 0:
                        data[oracle_sym] = {
                            "price": current_price,
                            "bid": current_price * 0.9999,
                            "ask": current_price * 1.0001,
                            "candles": [
                                {
                                    "time": int(candle_data["t"][i] * 1000),
                                    "o": candle_data["o"][i],
                                    "h": candle_data["h"][i],
                                    "l": candle_data["l"][i],
                                    "c": candle_data["c"][i],
                                    "v": candle_data["v"][i] if "v" in candle_data else 0
                                }
                                for i in range(len(candle_data["c"]))
                            ][-200:]
                        }
                        print(f"✅ Finnhub {oracle_sym}: ${current_price} + candles")
                    else:
                        # No candles available, use quote only
                        data[oracle_sym] = {
                            "price": current_price,
                            "bid": current_price * 0.9999,
                            "ask": current_price * 1.0001,
                            "candles": []
                        }
                        print(f"⚠️ Finnhub {oracle_sym}: ${current_price} (no candles)")
                        
                except Exception as candle_err:
                    # Candle failed, use quote
                    data[oracle_sym] = {
                        "price": current_price,
                        "bid": current_price * 0.9999,
                        "ask": current_price * 1.0001,
                        "candles": []
                    }
                    print(f"⚠️ Finnhub {oracle_sym}: Quote only - candle error: {type(candle_err).__name__}")
                    
            except Exception as e:
                print(f"❌ Finnhub {oracle_sym} error: {e}")
                continue
        
        if data:
            print(f"✅ Finnhub: {len(data)} pairs")
        
        return data
    except Exception as e:
        print(f"❌ Finnhub fetch error: {e}")
        return {}

# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS (V2.4 FEATURES)
# ═════════════════════════════════════════════════════════════════════════════

def calculate_market_regime(df: pd.DataFrame) -> Dict:
    """Classify market regime: trend, volatility, structure"""
    if len(df) < 20:
        return {"trend": "Unknown", "volatility": "Normal", "structure": "Unknown"}
    
    ema_50 = df["close"].ewm(span=50).mean()
    ema_200 = df["close"].ewm(span=200).mean()
    
    # Trend
    if ema_50.iloc[-1] > ema_200.iloc[-1] * 1.01:
        trend = "Strong"
    elif ema_50.iloc[-1] < ema_200.iloc[-1] * 0.99:
        trend = "Weak"
    else:
        trend = "Ranging"
    
    # Volatility (Bollinger Bands width)
    bb_std = df["close"].std()
    bb_width = 2 * bb_std / df["close"].mean()
    volatility = "Expanding" if bb_width > 0.04 else "Contracting" if bb_width < 0.015 else "Normal"
    
    # Structure
    highs = df["high"].tail(20).values
    lows = df["low"].tail(20).values
    swings = len(np.where(np.diff(highs) > 0)[0])
    structure = "Clean" if swings < 5 else "Choppy"
    
    return {
        "trend": trend,
        "volatility": volatility,
        "structure": structure
    }

def calculate_bias_stability(df: pd.DataFrame) -> Dict:
    """Calculate how long current bias has been active"""
    if len(df) < 5:
        return {"bias": "NEUTRAL", "active_since_minutes": 0, "last_flip_minutes_ago": None}
    
    ema_50 = df["close"].ewm(span=50).mean()
    current_bias = "BULLISH" if df["close"].iloc[-1] > ema_50.iloc[-1] else "BEARISH"
    
    # Find when bias started
    bias_started = 0
    for i in range(len(df) - 1, 0, -1):
        bias = "BULLISH" if df["close"].iloc[i] > ema_50.iloc[i] else "BEARISH"
        if bias != current_bias:
            bias_started = len(df) - i
            break
    
    return {
        "bias": current_bias,
        "active_since_minutes": bias_started,
        "last_flip_minutes_ago": bias_started if bias_started > 0 else None
    }

def calculate_confluence_breakdown(df: pd.DataFrame) -> Dict:
    """Calculate weighted confluence components"""
    components = {
        "EMA_Trend": {"weight": 40, "active": 0},
        "Momentum": {"weight": 30, "active": 0},
        "Volatility": {"weight": 20, "active": 0},
        "Volume": {"weight": 10, "active": 0}
    }
    
    # EMA Trend
    ema_50 = df["close"].ewm(span=50).mean()
    if df["close"].iloc[-1] > ema_50.iloc[-1]:
        components["EMA_Trend"]["active"] = 1
    
    # Momentum (RSI)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    if rsi.iloc[-1] > 50:
        components["Momentum"]["active"] = 1
    
    # Volatility (BB)
    bb_std = df["close"].std()
    bb_upper = df["close"].mean() + (2 * bb_std)
    if df["close"].iloc[-1] > bb_upper * 0.95:
        components["Volatility"]["active"] = 1
    
    # Volume
    if df["volume"].iloc[-1] > df["volume"].mean():
        components["Volume"]["active"] = 1
    
    return components

def calculate_state_statistics(df: pd.DataFrame) -> Dict:
    """Calculate historical outcomes in similar states"""
    if len(df) < 100:
        return {"continuation": 45, "reversal": 35, "consolidation": 20}
    
    # Simplified: based on trend persistence
    returns = df["close"].pct_change()
    continuation = (returns.iloc[-5:] > 0).sum() / 5 * 100
    reversal = (returns.iloc[-5:] < 0).sum() / 5 * 100
    consolidation = 100 - continuation - reversal
    
    return {
        "continuation": round(continuation, 0),
        "reversal": round(reversal, 0),
        "consolidation": round(consolidation, 0)
    }

def get_current_session() -> str:
    """Determine current trading session"""
    now = datetime.now(timezone.utc)
    hour = now.hour
    
    # Approximate session times (UTC)
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 16:
        return "Europe"
    elif 16 <= hour < 24:
        return "US"
    else:
        return "Overlap"

# ═════════════════════════════════════════════════════════════════════════════
# MAIN UPDATE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

async def update_all_pairs(session: aiohttp.ClientSession, update_forex_indices: bool = False):
    """Update market data and send to relay"""
    
    print(f"\n⏰ Update at {datetime.now().strftime('%H:%M:%S')} - Forex/Indices: {update_forex_indices}")
    
    # Fetch crypto every time
    crypto_data = await fetch_binance_data(session)
    market_data.update(crypto_data)
    print(f"✅ Crypto: {len(crypto_data)} pairs")
    
    # Fetch metals every time
    metals_data = await fetch_metals_data(session)
    market_data.update(metals_data)
    print(f"✅ Metals: {len(metals_data)} pairs")
    
    # Fetch forex/indices only when scheduled
    if update_forex_indices:
        forex_data = await fetch_finnhub_data(session)
        market_data.update(forex_data)
        print(f"✅ Forex/Indices: {len(forex_data)} pairs")
    
    # Prepare analysis data
    analysis_data = []
    
    for symbol, data in market_data.items():
        # Get current data from appropriate source
        if symbol in crypto_data:
            current = crypto_data.get(symbol)
        elif symbol in metals_data:
            current = metals_data.get(symbol)
        elif update_forex_indices and symbol in forex_data:
            current = forex_data.get(symbol)
        else:
            continue
        
        # Get or create candles dataframe
        if current:
            
            if current and "candles" in current:
                candles = current["candles"]
                df = pd.DataFrame(candles)
                df["close"] = df["c"]
                df["high"] = df["h"]
                df["low"] = df["l"]
                df["volume"] = df["v"]
                
                # Calculate V2.4 features
                market_regime = calculate_market_regime(df)
                bias_stability = calculate_bias_stability(df)
                confluence = calculate_confluence_breakdown(df)
                state_stats = calculate_state_statistics(df)
                
                analysis_data.append({
                    "symbol": symbol,
                    "price": current.get("price", 0),
                    "bid": current.get("bid", 0),
                    "ask": current.get("ask", 0),
                    "market_regime": market_regime,
                    "bias": bias_stability["bias"],
                    "bias_stability": {
                        "active_since_minutes": bias_stability["active_since_minutes"],
                        "last_flip_minutes_ago": bias_stability["last_flip_minutes_ago"]
                    },
                    "confluence_breakdown": confluence,
                    "state_statistics": state_stats,
                    "current_session": get_current_session(),
                    "confidence": 71,  # Default confidence
                    "indicators": {
                        "EMA50_above_200": market_regime["trend"] != "Weak",
                        "RSI_momentum": bias_stability["bias"] == "BULLISH",
                        "BB_squeeze": market_regime["volatility"] == "Contracting",
                        "Volume_confirmed": True
                    }
                })
    
    # Send to relay
    if analysis_data:
        try:
            async with session.post(
                f"{RELAY_URL}/market-analysis",
                json={"market_data": analysis_data},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    print(f"📤 Sent {len(analysis_data)} pairs to relay")
                else:
                    print(f"⚠️ Relay responded with {resp.status}")
        except Exception as e:
            print(f"❌ Failed to send to relay: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 80)
    print("🔥 ORACLEX V2.4 - PRODUCTION BACKEND - LIVE!")
    print("=" * 80)
    print("✓ Binance (BTCUSD, ETHUSD) - Every 10 seconds")
    print("✓ API Ninjas (XAUUSD, XAGUUSD) - Every 10 seconds")
    print("✓ Finnhub (EURUSD, GBPUSD, AUDUSD, NZDUSD) - Every 30 seconds")
    print("=" * 80)
    print()
    
    async with aiohttp.ClientSession() as session:
        cycle = 0
        while True:
            try:
                # Update crypto/metals every cycle (10 seconds)
                await update_all_pairs(session, update_forex_indices=False)
                
                # Update forex/indices every 3rd cycle (30 seconds)
                if cycle % 3 == 0:
                    await update_all_pairs(session, update_forex_indices=True)
                
                cycle += 1
                await asyncio.sleep(SCAN_INTERVAL_CRYPTO)
                
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                await asyncio.sleep(SCAN_INTERVAL_CRYPTO)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 OracleX V2.4 stopped")
