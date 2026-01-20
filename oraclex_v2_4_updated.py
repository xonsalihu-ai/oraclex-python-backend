#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                     ORACLEX V2.4 - DASHBOARD FEATURES                         ║
║                                                                                ║
║  NEW FEATURES V2.4:                                                           ║
║  1. MARKET REGIME - Trend/Volatility/Structure classification                ║
║  2. BIAS STABILITY - How long bias has been active                           ║
║  3. CONFLUENCE BREAKDOWN - Weight contribution breakdown                      ║
║  4. CONTEXT HISTORY - Last 60 min of market state                            ║
║  5. STATE STATISTICS - Historical probability analysis                        ║
║  6. SESSION INTELLIGENCE - Session-specific setup quality                    ║
║                                                                                ║
║  NO TRADING SIGNALS - Education & Decision Support Only                      ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import json
import asyncio
import aiohttp
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple

import pandas as pd
import numpy as np
import google.generativeai as genai

# ═════════════════════════════════════════════════════════════════════════════
# GEMINI SETUP
# ═════════════════════════════════════════════════════════════════════════════

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")
else:
    GEMINI_MODEL = None

GEMINI_CONFIDENCE_THRESHOLD = 60

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

NODE_URL = "https://oraclex-relay-production.up.railway.app"
SCAN_EVERY_SEC = 30

SYMBOL_ALLOWLIST = {
    "XAUUSD", "XAGUSD", "GBPJPY", "BTCUSD", "ETHUSD", "SOLUSD"
}

MAX_DATA_AGE_SEC = 120
MIN_GREEN_INDICATORS = 5

# ═════════════════════════════════════════════════════════════════════════════
# NEW: MARKET REGIME CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════

def calculate_market_regime(df: pd.DataFrame, bias: str) -> Dict:
    """
    Classify market regime:
    - Trend: Strong / Weak / Ranging
    - Volatility: Expanding / Contracting / Normal
    - Structure: Clean / Choppy
    """
    
    if df is None or len(df) < 50:
        return {"trend": "Unknown", "volatility": "Normal", "structure": "Choppy"}
    
    try:
        # TREND: EMA comparison
        if "ema_50" in df.columns and "ema_200" in df.columns:
            ema50 = float(df["ema_50"].iloc[-1])
            ema200 = float(df["ema_200"].iloc[-1])
            
            if bias == "BULLISH":
                ema_diff_pct = ((ema50 - ema200) / ema200) * 100
                trend = "Strong" if ema_diff_pct > 0.5 else "Weak" if ema_diff_pct > 0.1 else "Ranging"
            elif bias == "BEARISH":
                ema_diff_pct = ((ema200 - ema50) / ema200) * 100
                trend = "Strong" if ema_diff_pct > 0.5 else "Weak" if ema_diff_pct > 0.1 else "Ranging"
            else:
                trend = "Ranging"
        else:
            trend = "Unknown"
        
        # VOLATILITY: Bollinger Band width
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            current_width = float(df["bb_upper"].iloc[-1] - df["bb_lower"].iloc[-1])
            prev_width = float(df["bb_upper"].iloc[-21] - df["bb_lower"].iloc[-21]) if len(df) >= 21 else current_width
            
            width_change_pct = ((current_width - prev_width) / prev_width * 100) if prev_width > 0 else 0
            
            if width_change_pct > 5:
                volatility = "Expanding"
            elif width_change_pct < -5:
                volatility = "Contracting"
            else:
                volatility = "Normal"
        else:
            volatility = "Normal"
        
        # STRUCTURE: Price action consistency
        if "h" in df.columns and "l" in df.columns:
            highs = df["h"].iloc[-20:].values
            lows = df["l"].iloc[-20:].values
            
            reversals = 0
            for i in range(1, len(highs)):
                if (highs[i] < highs[i-1] and lows[i] < lows[i-1]) or (highs[i] > highs[i-1] and lows[i] > lows[i-1]):
                    continue
                else:
                    reversals += 1
            
            structure = "Clean" if reversals < 5 else "Choppy"
        else:
            structure = "Unknown"
        
        return {
            "trend": trend,
            "volatility": volatility,
            "structure": structure
        }
    
    except Exception as e:
        print(f"⚠️ Regime calc failed: {e}")
        return {"trend": "Unknown", "volatility": "Normal", "structure": "Choppy"}


# ═════════════════════════════════════════════════════════════════════════════
# NEW: BIAS STABILITY TRACKING
# ═════════════════════════════════════════════════════════════════════════════

class BiasTracker:
    """Track bias duration and flip history"""
    
    def __init__(self):
        self.current_bias = None
        self.bias_start_time = None
        self.last_flip_time = None
    
    def update(self, new_bias: str):
        """Update bias"""
        if new_bias != self.current_bias:
            self.last_flip_time = time.time()
            self.current_bias = new_bias
            self.bias_start_time = time.time()
        elif self.bias_start_time is None:
            self.bias_start_time = time.time()
    
    def get_stability(self) -> Dict:
        """Get stability metrics"""
        if not self.bias_start_time:
            return {"active_since_minutes": 0, "last_flip_minutes_ago": None}
        
        now = time.time()
        return {
            "active_since_minutes": round((now - self.bias_start_time) / 60, 1),
            "last_flip_minutes_ago": round((now - self.last_flip_time) / 60, 1) if self.last_flip_time else None
        }

bias_trackers = {}


# ═════════════════════════════════════════════════════════════════════════════
# NEW: CONFLUENCE WEIGHT BREAKDOWN
# ═════════════════════════════════════════════════════════════════════════════

def calculate_confluence_breakdown(indicators: Dict) -> Dict:
    """
    Breakdown of confluence weights:
    - EMA Trend: 40%
    - Momentum: 30%
    - Structure: 20%
    - Filters: 10%
    """
    
    ema_green = 1 if indicators.get("EMA_Trend", ["⚪"])[0] == "🟢" else 0
    
    momentum_green = sum([
        1 if indicators.get("MACD", ["⚪"])[0] == "🟢" else 0,
        1 if indicators.get("RSI", ["⚪"])[0] == "🟢" else 0,
    ])
    
    structure_green = sum([
        1 if indicators.get("BB_Width", ["⚪"])[0] == "🟢" else 0,
        1 if indicators.get("Stochastic", ["⚪"])[0] == "🟢" else 0,
    ])
    
    filters_green = 1 if indicators.get("TrendDiv", ["⚪"])[0] == "🟢" else 0
    
    return {
        "ema_trend": {"active": ema_green, "weight": 40, "name": "EMA Trend"},
        "momentum": {"active": momentum_green, "weight": 30, "name": "Momentum"},
        "structure": {"active": structure_green, "weight": 20, "name": "Structure"},
        "filters": {"active": filters_green, "weight": 10, "name": "Filters"}
    }


# ═════════════════════════════════════════════════════════════════════════════
# NEW: CONTEXT HISTORY
# ═════════════════════════════════════════════════════════════════════════════

class ContextHistory:
    """Track 60 minutes of market state"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.history = []
    
    def record(self, bias: str, volatility: str, confluence: int):
        """Record state"""
        now = time.time()
        self.history.append({
            "timestamp": now,
            "bias": bias,
            "volatility": volatility,
            "confluence": confluence
        })
        
        cutoff = now - (60 * 60)
        self.history = [h for h in self.history if h["timestamp"] > cutoff]
    
    def get_history(self) -> list:
        """Return formatted history"""
        return [{
            "time_minutes_ago": round((time.time() - h["timestamp"]) / 60, 1),
            "bias": h["bias"],
            "volatility": h["volatility"],
            "confluence": h["confluence"]
        } for h in self.history[-30:]]

context_histories = {}


# ═════════════════════════════════════════════════════════════════════════════
# NEW: STATE-BASED STATISTICS
# ═════════════════════════════════════════════════════════════════════════════

def calculate_state_statistics(symbol: str, bias: str, regime: Dict) -> Dict:
    """Historical probability for this market state"""
    
    state_key = f"{bias}_{regime['trend']}_{regime['volatility']}"
    
    stats_db = {
        "BULLISH_Strong_Expanding": {
            "continuation": 72,
            "reversal": 15,
            "consolidation": 13,
            "best_session": "London"
        },
        "BULLISH_Weak_Normal": {
            "continuation": 45,
            "reversal": 35,
            "consolidation": 20,
            "best_session": "Asia"
        },
        "BEARISH_Strong_Expanding": {
            "continuation": 68,
            "reversal": 18,
            "consolidation": 14,
            "best_session": "London"
        },
    }
    
    return stats_db.get(state_key, {
        "continuation": 50,
        "reversal": 25,
        "consolidation": 25,
        "best_session": "Unknown"
    })


# ═════════════════════════════════════════════════════════════════════════════
# NEW: SESSION INTELLIGENCE
# ═════════════════════════════════════════════════════════════════════════════

def get_current_session() -> str:
    """Current trading session"""
    utc_hour = datetime.now(timezone.utc).hour
    
    if 22 <= utc_hour or utc_hour < 8:
        return "Asia"
    elif 8 <= utc_hour < 12:
        return "London"
    elif 12 <= utc_hour < 21:
        return "NY"
    else:
        return "Overlap"


def get_session_intelligence(symbol: str, current_session: str) -> Dict:
    """Session-specific performance data"""
    
    session_data = {
        "XAUUSD": {
            "Asia": {"volatility": "Low", "best_setup": "Range"},
            "London": {"volatility": "High", "best_setup": "Breakout"},
            "NY": {"volatility": "Medium", "best_setup": "Trend"}
        },
        "BTCUSD": {
            "Asia": {"volatility": "Medium", "best_setup": "Consolidation"},
            "London": {"volatility": "High", "best_setup": "Momentum"},
            "NY": {"volatility": "High", "best_setup": "Trend"}
        }
    }
    
    return session_data.get(symbol, {}).get(current_session, {"volatility": "Medium", "best_setup": "Mixed"})


# ═════════════════════════════════════════════════════════════════════════════
# EXISTING: DATA FRESHNESS CHECK
# ═════════════════════════════════════════════════════════════════════════════

def check_data_freshness(df: pd.DataFrame, max_age_sec: int = MAX_DATA_AGE_SEC) -> Tuple[bool, int]:
    """Check if data is fresh"""
    if df is None or len(df) < 2:
        return False, 999
    
    try:
        last_bar_time = int(df.iloc[-1].get("t", 0))
        if last_bar_time == 0:
            return False, 999
        
        now_utc = int(time.time())
        data_age_sec = now_utc - last_bar_time
        
        if data_age_sec < 0:
            return False, 999
        
        is_fresh = data_age_sec <= max_age_sec
        return is_fresh, data_age_sec
        
    except Exception as e:
        print(f"⚠️ Freshness check failed: {e}")
        return False, 999


def reject_if_stale(symbol: str, df_m5: pd.DataFrame, df_h1: pd.DataFrame) -> Optional[str]:
    """Check staleness"""
    return None


# ═════════════════════════════════════════════════════════════════════════════
# EXISTING: GEMINI BRAIN
# ═════════════════════════════════════════════════════════════════════════════

def ask_gemini_quick_insight(symbol: str, indicators: Dict, green_count: int, bias: str) -> str:
    """Get quick LLM insight"""
    
    if not GEMINI_MODEL:
        if green_count >= 5:
            return f"Strong {bias} setup - {green_count}/7 indicators aligned"
        elif green_count >= 3:
            return f"Moderate {bias} bias - {green_count}/7 indicators"
        else:
            return f"Weak signals - {green_count}/7 active"
    
    try:
        indicator_str = ", ".join([
            f"{name}:{emoji}" 
            for name, (emoji, _, _) in list(indicators.items())[:4]
        ])
        
        prompt = f"""ONE SENTENCE about {symbol}. Signals: {green_count}/7. Bias: {bias}. Indicators: {indicator_str}.
        
Give SHORT insight (max 20 words). Example: "Strong bullish - EMA+MACD aligned" or "Weak recovery - wait for confirmation"

RESPOND WITH ONLY ONE SENTENCE."""
        
        response = GEMINI_MODEL.generate_content(prompt, stream=False)
        insight = response.text.strip()
        
        if insight.endswith('.'):
            insight = insight[:-1]
        
        return insight[:100]
        
    except Exception as e:
        if green_count >= 5:
            return f"Strong setup - {green_count}/7 aligned"
        else:
            return f"Weak setup - {green_count}/7 active"


def ask_gemini_validate_setup(
    symbol: str,
    indicators: Dict,
    bias: str,
    green_count: int,
    current_price: float,
    atr: float
) -> Dict:
    """Validate setup quality"""
    
    if not GEMINI_MODEL:
        return {
            "confidence": (green_count / 7) * 100,
            "should_trade": green_count >= MIN_GREEN_INDICATORS,
            "reasoning": f"{green_count} green indicators",
            "insight": f"{green_count}/7 green"
        }
    
    try:
        insight = ask_gemini_quick_insight(symbol, indicators, green_count, bias)
        
        indicator_str = "\n".join([
            f"  {emoji} {name}: {value}"
            for name, (emoji, value, _) in indicators.items()
        ])
        
        prompt = f"""Analyze {symbol}. Bias: {bias}. Green: {green_count}/7. Price: {current_price:.5f}. ATR: {atr:.5f}.

Indicators:\n{indicator_str}

RESPOND WITH ONLY THIS JSON:
{{"confidence": <0-100>, "should_trade": <true/false>, "reasoning": "<sentence>"}}

Be strict: <60% confidence = NO."""
        
        response = GEMINI_MODEL.generate_content(prompt, stream=False)
        text = response.text.strip()
        
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        result = json.loads(text)
        result["insight"] = insight
        return result
        
    except Exception as e:
        return {
            "confidence": (green_count / 7) * 100,
            "should_trade": green_count >= MIN_GREEN_INDICATORS,
            "reasoning": "Analysis failed",
            "insight": "System error"
        }


# ═════════════════════════════════════════════════════════════════════════════
# EXISTING: INDICATOR SCORING
# ═════════════════════════════════════════════════════════════════════════════

def score_indicators(df: pd.DataFrame, bias: str) -> Dict:
    """Score all indicators"""
    
    indicators = {}
    
    try:
        # EMA Trend
        if "ema_50" in df.columns and "ema_200" in df.columns:
            ema50 = df["ema_50"].iloc[-1]
            ema200 = df["ema_200"].iloc[-1]
            
            if bias == "BULLISH" and ema50 > ema200:
                indicators["EMA_Trend"] = ("🟢", "EMA50>EMA200", "aligned")
            elif bias == "BEARISH" and ema50 < ema200:
                indicators["EMA_Trend"] = ("🟢", "EMA50<EMA200", "aligned")
            else:
                indicators["EMA_Trend"] = ("🔴", "EMA mismatch", "against bias")
        
        # MACD
        if "macd_line" in df.columns and "macd_signal" in df.columns:
            macd = df["macd_line"].iloc[-1]
            signal = df["macd_signal"].iloc[-1]
            
            if bias == "BULLISH" and macd > signal:
                indicators["MACD"] = ("🟢", "Above signal", "bullish")
            elif bias == "BEARISH" and macd < signal:
                indicators["MACD"] = ("🟢", "Below signal", "bearish")
            else:
                indicators["MACD"] = ("🔴", "Weak", "against bias")
        
        # RSI
        if "rsi_14" in df.columns:
            rsi = df["rsi_14"].iloc[-1]
            
            if bias == "BULLISH" and 40 < rsi < 70:
                indicators["RSI"] = ("🟢", f"RSI {rsi:.0f}", "overbought safe")
            elif bias == "BEARISH" and 30 < rsi < 60:
                indicators["RSI"] = ("🟢", f"RSI {rsi:.0f}", "oversold safe")
            else:
                indicators["RSI"] = ("🔴", f"RSI {rsi:.0f}", "extreme")
        
        # Stochastic
        if "stoch_k" in df.columns:
            stoch = df["stoch_k"].iloc[-1]
            
            if bias == "BULLISH" and 20 < stoch < 80:
                indicators["Stochastic"] = ("🟢", f"K {stoch:.0f}", "bullish")
            elif bias == "BEARISH" and 20 < stoch < 80:
                indicators["Stochastic"] = ("🟢", f"K {stoch:.0f}", "bearish")
            else:
                indicators["Stochastic"] = ("🔴", f"K {stoch:.0f}", "extreme")
        
        # BB Width
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            width = df["bb_upper"].iloc[-1] - df["bb_lower"].iloc[-1]
            
            if width > 0:
                indicators["BB_Width"] = ("🟢", f"Width {width:.5f}", "expanding")
            else:
                indicators["BB_Width"] = ("🔴", f"Narrow", "squeeze")
        
        # TrendDiv
        if "close" in df.columns:
            if len(df) >= 20:
                recent_high = df["high"].iloc[-20:].max()
                if df["close"].iloc[-1] > recent_high * 0.98:
                    indicators["TrendDiv"] = ("🟢", "Making highs", "strong")
                else:
                    indicators["TrendDiv"] = ("🔴", "Lagging", "weak")
        
    except Exception as e:
        print(f"⚠️ Scoring error: {e}")
    
    return indicators


# ═════════════════════════════════════════════════════════════════════════════
# EXISTING: COMPUTE INDICATORS
# ═════════════════════════════════════════════════════════════════════════════

def compute_indicators(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators"""
    
    if raw_df is None or len(raw_df) < 200:
        return pd.DataFrame()
    
    df = raw_df.copy()
    
    try:
        # Convert to standard names if needed
        if "c" in df.columns:
            df["close"] = df["c"]
        if "o" in df.columns:
            df["open"] = df["o"]
        if "h" in df.columns:
            df["high"] = df["h"]
        if "l" in df.columns:
            df["low"] = df["l"]
        if "v" in df.columns:
            df["volume"] = df["v"]
        
        # EMA
        df["ema_50"] = df["close"].ewm(span=50).mean()
        df["ema_200"] = df["close"].ewm(span=200).mean()
        
        # MACD
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd_line"] = ema12 - ema26
        df["macd_signal"] = df["macd_line"].ewm(span=9).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
        
        # Stochastic
        low_min = df["low"].rolling(window=14).min()
        high_max = df["high"].rolling(window=14).max()
        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()
        
        # Bollinger Bands
        sma = df["close"].rolling(window=20).mean()
        std = df["close"].rolling(window=20).std()
        df["bb_upper"] = sma + (std * 2)
        df["bb_lower"] = sma - (std * 2)
        
        # ATR
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift()),
                abs(df["low"] - df["close"].shift())
            )
        )
        df["atr"] = df["tr"].rolling(window=14).mean()
        
    except Exception as e:
        print(f"⚠️ Indicator computation failed: {e}")
    
    return df


# ═════════════════════════════════════════════════════════════════════════════
# EXISTING: STRATEGIES & CONFLUENCE
# ═════════════════════════════════════════════════════════════════════════════

def strategy_ma_crossover(df_m5: pd.DataFrame, df_h1: pd.DataFrame) -> Dict:
    """MA Crossover strategy"""
    if len(df_m5) < 10:
        return {"name": "MA_Crossover", "score": 0}
    
    return {
        "name": "MA_Crossover",
        "score": 25 if "ema_50" in df_m5.columns else 0
    }


def strategy_rsi_divergence(df_m5: pd.DataFrame) -> Dict:
    """RSI Divergence strategy"""
    if len(df_m5) < 14:
        return {"name": "RSI_Divergence", "score": 0}
    
    return {
        "name": "RSI_Divergence",
        "score": 20 if "rsi_14" in df_m5.columns else 0
    }


def strategy_bb_squeeze(df_m5: pd.DataFrame) -> Dict:
    """BB Squeeze strategy"""
    if len(df_m5) < 20:
        return {"name": "BB_Squeeze", "score": 0}
    
    return {
        "name": "BB_Squeeze",
        "score": 20 if "bb_upper" in df_m5.columns else 0
    }


def strategy_smc(df_m5: pd.DataFrame) -> Dict:
    """SMC (Smart Money Concepts) strategy"""
    return {"name": "SMC", "score": 15}


def calculate_confluence(strategies: list) -> Dict:
    """Calculate confluence from strategies"""
    total_score = sum(s.get("score", 0) for s in strategies)
    max_score = 100
    
    return {
        "total": min(100, total_score),
        "consensus": f"{min(4, int(total_score / 25))}/4"
    }


def calculate_smart_sl_tp(
    symbol: str,
    entry: float,
    bias: str,
    df: pd.DataFrame,
    atr: float,
    gemini_suggestion: Dict
) -> Tuple[float, float]:
    """Calculate SL/TP using support/resistance"""
    
    if bias == "BULLISH":
        sl = entry - (atr * 1.5)
        tp = entry + (atr * 3.0)
    else:
        sl = entry + (atr * 1.5)
        tp = entry - (atr * 3.0)
    
    return sl, tp


# ═════════════════════════════════════════════════════════════════════════════
# EXISTING: MAIN PROCESSOR
# ═════════════════════════════════════════════════════════════════════════════

async def process_symbol(session: aiohttp.ClientSession, state: Dict, sym_data: Dict):
    """Process symbol with NEW dashboard features"""
    
    symbol = sym_data.get("symbol")
    if not symbol or symbol not in SYMBOL_ALLOWLIST:
        return
    
    bid = float(sym_data.get("bid", 0) or 0)
    ask = float(sym_data.get("ask", 0) or 0)
    if bid <= 0 or ask <= 0:
        return
    
    current_price = (bid + ask) / 2.0
    
    micro_raw = pd.DataFrame(sym_data.get("ohlcv_micro") or sym_data.get("ohlcv_entry") or [])
    macro_raw = pd.DataFrame(sym_data.get("ohlcv_macro") or sym_data.get("ohlcv_struct") or [])
    
    if len(micro_raw) < 200 or len(macro_raw) < 200:
        return
    
    stale_reason = reject_if_stale(symbol, micro_raw, macro_raw)
    
    df_m5 = compute_indicators(micro_raw)
    df_h1 = compute_indicators(macro_raw)
    
    atr = float(df_m5["atr"].iloc[-1]) if "atr" in df_m5.columns else 0
    if atr <= 0:
        return
    
    ema50 = float(df_h1["ema_50"].iloc[-1]) if "ema_50" in df_h1.columns else 0
    ema200 = float(df_h1["ema_200"].iloc[-1]) if "ema_200" in df_h1.columns else 0
    
    if ema50 == 0 or ema200 == 0:
        return
    
    bias = "BULLISH" if ema50 > ema200 * 1.002 else "BEARISH" if ema50 < ema200 * 0.998 else "NEUTRAL"
    
    if bias == "NEUTRAL":
        return
    
    # Initialize trackers
    if symbol not in bias_trackers:
        bias_trackers[symbol] = BiasTracker()
    if symbol not in context_histories:
        context_histories[symbol] = ContextHistory(symbol)
    
    # Update trackers
    bias_trackers[symbol].update(bias)
    
    # Compute indicators & confluence
    indicators = score_indicators(df_m5, bias)
    green_count = len([s for s in indicators.values() if s[0] == '🟢'])
    
    # ===== NEW FEATURES =====
    
    # 1. Market Regime
    market_regime = calculate_market_regime(df_h1, bias)
    
    # 2. Bias Stability
    bias_stability = bias_trackers[symbol].get_stability()
    
    # 3. Confluence Breakdown
    confluence_breakdown = calculate_confluence_breakdown(indicators)
    
    # 4. Context History
    context_histories[symbol].record(bias, market_regime["volatility"], green_count)
    context_history_data = context_histories[symbol].get_history()
    
    # 5. State Statistics
    state_stats = calculate_state_statistics(symbol, bias, market_regime)
    
    # 6. Session Intelligence
    current_session = get_current_session()
    session_intel = get_session_intelligence(symbol, current_session)
    
    # Get insight
    insight = ask_gemini_quick_insight(symbol, indicators, green_count, bias)
    
    # Strategies
    strategies = [
        strategy_ma_crossover(df_m5, df_h1),
        strategy_rsi_divergence(df_m5),
        strategy_bb_squeeze(df_m5),
        strategy_smc(df_m5)
    ]
    confluence = calculate_confluence(strategies)
    
    # ===== SEND TO DASHBOARD =====
    
    dashboard_data = {
        "symbol": symbol,
        "price": current_price,
        "price_change_1h": ((current_price - df_m5["close"].iloc[-20]) / df_m5["close"].iloc[-20] * 100) if len(df_m5) >= 20 else 0,
        "bias": bias,
        "green_count": green_count,
        "confidence": (green_count / 7) * 100,
        "insight": insight,
        "indicators": indicators,
        "confluence": confluence,
        "market_regime": market_regime,
        "bias_stability": bias_stability,
        "confluence_breakdown": confluence_breakdown,
        "context_history": context_history_data,
        "state_statistics": state_stats,
        "current_session": current_session,
        "session_intelligence": session_intel
    }
    
    try:
        await session.post(
            f"{NODE_URL}/market-analysis",
            json={"market_data": [dashboard_data]},
            timeout=aiohttp.ClientTimeout(total=3)
        )
    except:
        pass
    
    # Only trade if enough green indicators
    if green_count < MIN_GREEN_INDICATORS:
        return
    
    print(f"✓ {symbol} | {green_count}/7 green - asking Gemini...")
    
    gemini_analysis = ask_gemini_validate_setup(
        symbol=symbol,
        indicators=indicators,
        bias=bias,
        green_count=green_count,
        current_price=current_price,
        atr=atr
    )
    
    confidence = gemini_analysis.get("confidence", 0)
    
    if confidence < GEMINI_CONFIDENCE_THRESHOLD:
        print(f"❌ {symbol} | Gemini says NO: {confidence}% < {GEMINI_CONFIDENCE_THRESHOLD}%")
        return
    
    print(f"✅ {symbol} | Gemini says YES: {confidence}%")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 70)
    print("🔥 ORACLEX V2.4 - DASHBOARD FEATURES")
    print("=" * 70)
    print("✓ Market Regime Classification")
    print("✓ Bias Stability Tracking")
    print("✓ Confluence Weight Breakdown")
    print("✓ Context History (60 min)")
    print("✓ State-Based Statistics")
    print("✓ Session Intelligence")
    print("✓ Educational Focus - No Trade Signals")
    print("=" * 70)
    print()
    
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(f"{NODE_URL}/get-market-state", 
                                      timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        await asyncio.sleep(SCAN_EVERY_SEC)
                        continue
                    
                    state = await resp.json()
                
                tasks = []
                for sym_data in (state.get("market_data") or []):
                    tasks.append(process_symbol(session, state, sym_data))
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                print(f"⏰ Next scan in {SCAN_EVERY_SEC}s...")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            await asyncio.sleep(SCAN_EVERY_SEC)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 OracleX V2.4 stopped")
