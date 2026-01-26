#!/usr/bin/env python3
"""
ORACLEX PYTHON BACKEND V2.0 - PRODUCTION GRADE
Clean, simple, real calculations from actual market data
"""

from aiohttp import web
import numpy as np
from datetime import datetime, timezone

# In-memory market data cache
market_cache = {}

class OracleXAnalyzer:
    """Market analysis - real calculations from EA data"""
    
    @staticmethod
    def calculate_bias_from_ema(timeframes):
        """
        Simple, correct bias calculation:
        - Get M5 latest candle
        - Compare close to ema_20
        - That's the bias
        """
        if 'M5' not in timeframes:
            return 'NEUTRAL', 0
        
        m5 = timeframes['M5']
        candles = m5.get('candles', [])
        
        if not candles:
            return 'NEUTRAL', 0
        
        latest = candles[-1]
        close = latest.get('c', 0)
        ema20 = latest.get('ema_20', close)
        
        # THIS IS THE ONLY LOGIC NEEDED
        if close > ema20:
            bias = 'BULLISH'
            strength = close - ema20
        elif close < ema20:
            bias = 'BEARISH'
            strength = ema20 - close
        else:
            bias = 'NEUTRAL'
            strength = 0
        
        return bias, strength
    
    @staticmethod
    def calculate_multi_tf_agreement(timeframes):
        """Count how many timeframes agree with M5 bias"""
        m5_bias, _ = OracleXAnalyzer.calculate_bias_from_ema(timeframes)
        
        if m5_bias == 'NEUTRAL':
            return 0, {}
        
        agreement_count = 0
        total_tfs = 0
        tf_bias_map = {}
        
        for tf_name in ['M1', 'M5', 'M15', 'H1', 'H4', 'D1', 'W1']:
            if tf_name not in timeframes:
                continue
            
            tf_data = timeframes[tf_name]
            candles = tf_data.get('candles', [])
            
            if not candles:
                continue
            
            latest = candles[-1]
            close = latest.get('c', 0)
            ema20 = latest.get('ema_20', close)
            
            # Determine this TF's bias
            if close > ema20:
                tf_bias = 'BULLISH'
            elif close < ema20:
                tf_bias = 'BEARISH'
            else:
                tf_bias = 'NEUTRAL'
            
            tf_bias_map[tf_name] = {
                'bias': tf_bias,
                'close': close,
                'strength': abs(close - ema20)
            }
            
            total_tfs += 1
            
            # Count agreements
            if tf_bias == m5_bias:
                agreement_count += 1
        
        # Calculate agreement percentage
        agreement_pct = (agreement_count / total_tfs * 100) if total_tfs > 0 else 0
        
        return agreement_pct, tf_bias_map
    
    @staticmethod
    def calculate_confluence_score(timeframes):
        """
        Count bullish indicators:
        - RSI > 60 or < 40 (extreme)
        - MACD histogram non-zero
        - Price above EMA20
        Score = (hits / total_checks) * 100
        """
        confluence_points = 0
        total_checks = 0
        
        for tf_name in ['M1', 'M5', 'M15', 'H1', 'H4']:  # Focus on shorter timeframes
            if tf_name not in timeframes:
                continue
            
            tf_data = timeframes[tf_name]
            candles = tf_data.get('candles', [])
            indicators = tf_data.get('indicators', {})
            
            if not candles:
                continue
            
            latest = candles[-1]
            
            # Check 1: RSI extremes
            rsi = indicators.get('rsi', 50)
            if rsi > 60 or rsi < 40:
                confluence_points += 1
            total_checks += 1
            
            # Check 2: MACD histogram direction
            macd_hist = indicators.get('macd_histogram', 0)
            if abs(macd_hist) > 0.00001:  # Non-zero with floating point tolerance
                confluence_points += 1
            total_checks += 1
            
            # Check 3: Price above EMA
            close = latest.get('c', 0)
            ema20 = latest.get('ema_20', close)
            if close > ema20:
                confluence_points += 1
            total_checks += 1
        
        score = (confluence_points / total_checks * 100) if total_checks > 0 else 50
        return min(100, max(0, score))
    
    @staticmethod
    def calculate_market_regime(timeframes):
        """Trend from H1, volatility from ATR"""
        
        trend = 'Unknown'
        volatility = 'Normal'
        structure = 'Unknown'
        
        # TREND from H1
        if 'H1' in timeframes:
            h1 = timeframes['H1']
            candles = h1.get('candles', [])
            
            if candles:
                latest = candles[-1]
                close = latest.get('c', 0)
                ema20 = latest.get('ema_20', close)
                
                trend = 'Strong Up' if close > ema20 else 'Strong Down'
        
        # VOLATILITY from H1 ATR
        if 'H1' in timeframes:
            h1 = timeframes['H1']
            candles = h1.get('candles', [])
            indicators = h1.get('indicators', {})
            
            if candles and len(candles) >= 20:
                atr = indicators.get('atr', 0)
                
                # Calculate ATR mean from last 20 candles
                ranges = [max(0.0001, c.get('h', 0) - c.get('l', 0)) for c in candles[-20:]]
                atr_mean = np.mean(ranges)
                
                if atr > atr_mean * 1.5:
                    volatility = 'Extreme'
                elif atr > atr_mean * 1.2:
                    volatility = 'Elevated'
                elif atr < atr_mean * 0.8:
                    volatility = 'Quiet'
                else:
                    volatility = 'Normal'
        
        # STRUCTURE from H1 - count up vs down candles
        if 'H1' in timeframes:
            h1 = timeframes['H1']
            candles = h1.get('candles', [])
            
            if candles and len(candles) >= 10:
                up_candles = sum(1 for c in candles[-10:] if c.get('c', 0) > c.get('o', 0))
                structure = 'Clean' if up_candles >= 5 else 'Choppy'
        
        return {
            'trend': trend,
            'volatility': volatility,
            'structure': structure
        }
    
    @staticmethod
    def calculate_liquidity(timeframes):
        """Extract highs/lows from D1 and H4 as S/R"""
        
        support = []
        resistance = []
        
        for tf_name in ['D1', 'H4']:
            if tf_name not in timeframes:
                continue
            
            tf_data = timeframes[tf_name]
            candles = tf_data.get('candles', [])
            
            if not candles:
                continue
            
            # Take last 20 candles
            for candle in candles[-20:]:
                h = candle.get('h', 0)
                l = candle.get('l', 0)
                
                if h > 0:
                    resistance.append({'price': h, 'tf': tf_name})
                if l > 0:
                    support.append({'price': l, 'tf': tf_name})
        
        # Remove duplicates and sort
        if support:
            support = sorted(list({s['price']: s for s in support}.values()), 
                           key=lambda x: x['price'], reverse=True)[:3]
        if resistance:
            resistance = sorted(list({r['price']: r for r in resistance}.values()), 
                              key=lambda x: x['price'])[:3]
        
        return {
            'support': support,
            'resistance': resistance
        }
    
    @staticmethod
    def calculate_microstructure(market_data):
        """Bid/ask/spread"""
        
        bid = market_data.get('bid', 0)
        ask = market_data.get('ask', 0)
        spread_points = market_data.get('spread_points', 0)
        
        mid = (bid + ask) / 2 if bid and ask else 0
        spread_pct = (spread_points / mid * 100) if mid > 0 else 0
        
        interpretation = 'Tight spread' if spread_pct < 0.05 else 'Wide spread'
        
        return {
            'bid': bid,
            'ask': ask,
            'spread_pct': spread_pct,
            'interpretation': interpretation
        }
    
    @staticmethod
    def get_session():
        """Current session based on UTC hour"""
        
        utc_hour = datetime.now(timezone.utc).hour
        
        if 0 <= utc_hour < 9:
            session = 'Asia'
            hours = '00:00-09:00 UTC'
            volatility = 30
        elif 8 <= utc_hour < 17:
            session = 'Europe'
            hours = '08:00-17:00 UTC'
            volatility = 70
        elif 16 <= utc_hour < 24:
            session = 'US'
            hours = '16:00-23:59 UTC'
            volatility = 80
        else:
            session = 'Overlap'
            hours = 'Overlapping'
            volatility = 60
        
        return {
            'current_session': session,
            'session_hours': hours,
            'typical_volatility': volatility
        }
    
    @staticmethod
    def generate_interpretation(symbol, data):
        """Human-readable market description"""
        
        bias = data.get('bias', 'NEUTRAL')
        confidence = data.get('confidence', 50)
        trend = data.get('market_regime', {}).get('trend', 'Unknown')
        confluence = data.get('confluence', 50)
        
        clarity = 'strong' if confidence > 70 else 'moderate' if confidence > 50 else 'weak'
        strength = 'strong' if confluence > 70 else 'moderate' if confluence > 50 else 'weak'
        
        text = f"{symbol} showing {clarity} {bias.lower()} bias. "
        text += f"Market shows {trend.lower()} trend. "
        text += f"Confluence at {confluence:.0f}% indicates {strength} agreement."
        
        return text


async def handle_market_data(request):
    """Receive market data from relay"""
    try:
        body = await request.json()
        market_data_list = body.get('market_data', [])
        
        if isinstance(market_data_list, dict):
            market_data_list = [market_data_list]
        
        stored = 0
        for item in market_data_list:
            symbol = item.get('symbol')
            if symbol:
                market_cache[symbol] = item
                stored += 1
        
        print(f"[RECEIVE] {stored} symbols stored. Cache: {list(market_cache.keys())}")
        
        return web.json_response({'status': 'stored', 'count': stored})
    
    except Exception as e:
        print(f"[ERROR] handle_market_data: {e}")
        return web.json_response({'error': str(e)}, status=400)


async def handle_analysis(request):
    """Generate analysis for symbol"""
    try:
        symbol = request.match_info.get('symbol', '').upper()
        
        if symbol not in market_cache:
            return web.json_response({'error': f'{symbol} not found'}, status=404)
        
        market_data = market_cache[symbol]
        timeframes = market_data.get('timeframes', {})
        price = market_data.get('price', 0)
        
        # Calculate all metrics
        bias, bias_strength = OracleXAnalyzer.calculate_bias_from_ema(timeframes)
        agreement_pct, tf_bias_map = OracleXAnalyzer.calculate_multi_tf_agreement(timeframes)
        confluence = OracleXAnalyzer.calculate_confluence_score(timeframes)
        market_regime = OracleXAnalyzer.calculate_market_regime(timeframes)
        liquidity = OracleXAnalyzer.calculate_liquidity(timeframes)
        microstructure = OracleXAnalyzer.calculate_microstructure(market_data)
        session = OracleXAnalyzer.get_session()
        
        # Confidence = average of agreement and confluence
        confidence = (agreement_pct + confluence) / 2
        
        # Find dominant TF
        dominant_tf = 'H1' if 'H1' in tf_bias_map else 'H4' if 'H4' in tf_bias_map else (list(tf_bias_map.keys())[0] if tf_bias_map else 'Unknown')
        
        # Build analysis data for interpretation
        analysis_data = {
            'bias': bias,
            'confidence': confidence,
            'market_regime': market_regime,
            'confluence': confluence
        }
        
        interpretation = OracleXAnalyzer.generate_interpretation(symbol, analysis_data)
        
        # Build response - EXACTLY what dashboard needs
        response = {
            'symbol': symbol,
            'price': price,
            'bid': market_data.get('bid', 0),
            'ask': market_data.get('ask', 0),
            'bias': bias,
            'confluence': confluence,
            'confidence': confidence,
            'market_regime': market_regime,
            'bias_stability': {
                'bias': bias,
                'active_since_minutes': 0,  # TODO: calculate from consecutive candles
                'multi_tf_agreement': agreement_pct
            },
            'multi_timeframe': {
                'dominant_tf': dominant_tf,
                'agreement_score': agreement_pct,
                'timeframe_bias': tf_bias_map
            },
            'liquidity': liquidity,
            'microstructure': microstructure,
            'session': session,
            'interpretation': interpretation
        }
        
        return web.json_response(response)
    
    except Exception as e:
        print(f"[ERROR] handle_analysis: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({'error': str(e)}, status=500)


async def handle_latest_analysis(request):
    """Get list of cached symbols"""
    try:
        return web.json_response({'analyses': list(market_cache.keys())})
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def handle_health(request):
    """Health check"""
    return web.json_response({
        'status': 'ok',
        'cached_symbols': list(market_cache.keys()),
        'timestamp': datetime.now().isoformat()
    })


app = web.Application()
app.router.add_post('/market-data-v1.6', handle_market_data)
app.router.add_get('/analysis/{symbol}', handle_analysis)
app.router.add_get('/latest-analysis', handle_latest_analysis)
app.router.add_get('/', handle_health)

if __name__ == '__main__':
    print('\n' + '='*80)
    print('✨ ORACLEX PYTHON BACKEND V2.0 - PRODUCTION')
    print('='*80)
    print('Real calculations from actual market data')
    print('Port: 8080')
    print('='*80 + '\n')
    
    web.run_app(app, port=8080)
