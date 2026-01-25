#!/usr/bin/env python3
"""
ORACLEX PYTHON BACKEND - PRODUCTION GRADE
Implements complete market analysis per specification
"""

from aiohttp import web
import numpy as np
import json
from datetime import datetime, timezone

SYMBOLS = ['XAUUSD', 'XAGUUSD', 'BTCUSD', 'ETHUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']
market_cache = {}

class OracleXAnalyzer:
    """Market analysis engine - implements specification exactly"""
    
    def calculate_confluence(self, symbol, timeframes):
        """
        Count agreeing indicators across timeframes
        RSI extreme + MACD histogram + Price above EMA20 = points
        """
        if not timeframes:
            return 0
        
        confluence_points = 0
        total_indicators = 0
        
        try:
            for tf_name, tf_data in timeframes.items():
                if not tf_data or not isinstance(tf_data, dict):
                    continue
                
                candles = tf_data.get('candles', [])
                indicators = tf_data.get('indicators', {})
                
                if not candles or len(candles) < 1:
                    continue
                
                recent = candles[-1]
                
                # RSI check
                rsi = indicators.get('rsi', 50)
                if rsi > 60 or rsi < 40:
                    confluence_points += 1
                total_indicators += 1
                
                # MACD histogram
                macd_hist = indicators.get('macd_histogram', 0)
                if macd_hist != 0:
                    confluence_points += 1
                total_indicators += 1
                
                # Price above EMA20
                close = recent.get('c', 0)
                ema20 = recent.get('ema_20', close)
                if close > ema20:
                    confluence_points += 1
                total_indicators += 1
        
        except Exception as e:
            print(f"[ERROR] Confluence calc: {e}")
            return 50
        
        if total_indicators == 0:
            return 50
        
        score = (confluence_points / total_indicators) * 100
        return min(100, max(0, score))
    
    def calculate_market_regime(self, symbol, timeframes):
        """Determine trend, volatility, structure from H1 candles"""
        
        try:
            if not timeframes or 'H1' not in timeframes:
                return {
                    'trend': 'Unknown',
                    'volatility': 'Normal',
                    'structure': 'Unknown'
                }
            
            h1_data = timeframes['H1']
            if not h1_data or not isinstance(h1_data, dict):
                return {
                    'trend': 'Unknown',
                    'volatility': 'Normal',
                    'structure': 'Unknown'
                }
            
            candles = h1_data.get('candles', [])
            indicators = h1_data.get('indicators', {})
            
            if len(candles) < 2:
                return {
                    'trend': 'Unknown',
                    'volatility': 'Normal',
                    'structure': 'Unknown'
                }
            
            recent = candles[-1]
            close = recent.get('c', 0)
            ema20 = recent.get('ema_20', close)
            
            # TREND
            trend = 'Strong Up' if close > ema20 else 'Strong Down'
            
            # VOLATILITY via ATR
            atr = indicators.get('atr', 0)
            if atr > 0:
                atr_mean = np.mean([max(0.0001, c.get('h', 0) - c.get('l', 0)) for c in candles[-20:]])
                if atr > atr_mean * 1.5:
                    volatility = 'Extreme'
                elif atr > atr_mean * 1.2:
                    volatility = 'Elevated'
                elif atr < atr_mean * 0.8:
                    volatility = 'Quiet'
                else:
                    volatility = 'Normal'
            else:
                volatility = 'Normal'
            
            # STRUCTURE - count up vs down candles in last 10
            up_candles = sum(1 for c in candles[-10:] if c.get('c', 0) > c.get('o', 0))
            structure = 'Clean' if up_candles >= 5 else 'Choppy'
            
            return {
                'trend': trend,
                'volatility': volatility,
                'structure': structure
            }
        
        except Exception as e:
            print(f"[ERROR] Market regime: {e}")
            return {
                'trend': 'Unknown',
                'volatility': 'Normal',
                'structure': 'Unknown'
            }
    
    def calculate_bias_stability(self, symbol, timeframes):
        """Bias from M5, active duration, multi-TF agreement"""
        
        try:
            if not timeframes or 'M5' not in timeframes:
                return {
                    'bias': 'NEUTRAL',
                    'active_since_minutes': 0,
                    'multi_tf_agreement': 0
                }
            
            m5_data = timeframes['M5']
            if not m5_data or not isinstance(m5_data, dict):
                return {
                    'bias': 'NEUTRAL',
                    'active_since_minutes': 0,
                    'multi_tf_agreement': 0
                }
            
            candles = m5_data.get('candles', [])
            if len(candles) < 1:
                return {
                    'bias': 'NEUTRAL',
                    'active_since_minutes': 0,
                    'multi_tf_agreement': 0
                }
            
            recent = candles[-1]
            close = recent.get('c', 0)
            ema20 = recent.get('ema_20', close)
            
            # Current bias
            current_bias = 'BULLISH' if close > ema20 else 'BEARISH'
            
            # Active since - count consecutive candles with same bias
            active_candles = 1
            for i in range(len(candles) - 2, -1, -1):
                c = candles[i]
                candle_bias = 'BULLISH' if c.get('c', 0) > c.get('ema_20', 0) else 'BEARISH'
                if candle_bias == current_bias:
                    active_candles += 1
                else:
                    break
            
            active_since_minutes = active_candles * 5
            
            # Multi-TF agreement
            agreeing_tfs = 0
            total_tfs = 0
            for tf_name, tf_data in timeframes.items():
                if not tf_data or not isinstance(tf_data, dict):
                    continue
                tf_candles = tf_data.get('candles', [])
                if not tf_candles or len(tf_candles) < 1:
                    continue
                
                tf_recent = tf_candles[-1]
                tf_bias = 'BULLISH' if tf_recent.get('c', 0) > tf_recent.get('ema_20', 0) else 'BEARISH'
                
                if tf_bias == current_bias:
                    agreeing_tfs += 1
                total_tfs += 1
            
            multi_tf_agreement = (agreeing_tfs / total_tfs * 100) if total_tfs > 0 else 0
            
            return {
                'bias': current_bias,
                'active_since_minutes': active_since_minutes,
                'multi_tf_agreement': multi_tf_agreement
            }
        
        except Exception as e:
            print(f"[ERROR] Bias stability: {e}")
            return {
                'bias': 'NEUTRAL',
                'active_since_minutes': 0,
                'multi_tf_agreement': 0
            }
    
    def calculate_multi_timeframe_confluence(self, symbol, timeframes):
        """Agreement across all timeframes"""
        
        try:
            if not timeframes:
                return {
                    'dominant_tf': 'Unknown',
                    'agreement_score': 0,
                    'timeframe_bias': {}
                }
            
            timeframe_bias = {}
            bullish_count = 0
            bearish_count = 0
            
            for tf_name, tf_data in timeframes.items():
                if not tf_data or not isinstance(tf_data, dict):
                    continue
                
                candles = tf_data.get('candles', [])
                if not candles or len(candles) < 1:
                    continue
                
                recent = candles[-1]
                close = recent.get('c', 0)
                ema20 = recent.get('ema_20', close)
                
                bias = 'BULLISH' if close > ema20 else 'BEARISH'
                strength = abs(close - ema20)
                
                timeframe_bias[tf_name] = {
                    'bias': bias,
                    'close': close,
                    'strength': strength
                }
                
                if bias == 'BULLISH':
                    bullish_count += 1
                else:
                    bearish_count += 1
            
            total = len(timeframe_bias)
            agreement_score = (max(bullish_count, bearish_count) / total * 100) if total > 0 else 0
            
            # Dominant TF preference
            dominant_tf = 'H1'
            if 'H1' not in timeframe_bias and 'H4' in timeframe_bias:
                dominant_tf = 'H4'
            elif 'H1' not in timeframe_bias and 'H4' not in timeframe_bias and timeframe_bias:
                dominant_tf = list(timeframe_bias.keys())[0]
            
            return {
                'dominant_tf': dominant_tf,
                'agreement_score': agreement_score,
                'timeframe_bias': timeframe_bias
            }
        
        except Exception as e:
            print(f"[ERROR] Multi-TF confluence: {e}")
            return {
                'dominant_tf': 'Unknown',
                'agreement_score': 0,
                'timeframe_bias': {}
            }
    
    def calculate_liquidity(self, symbol, timeframes):
        """Support/resistance from H4 and D1"""
        
        try:
            support = []
            resistance = []
            
            for tf_name in ['D1', 'H4']:
                if tf_name not in timeframes:
                    continue
                
                tf_data = timeframes[tf_name]
                if not tf_data or not isinstance(tf_data, dict):
                    continue
                
                candles = tf_data.get('candles', [])
                if not candles:
                    continue
                
                for c in candles[-20:]:
                    h = c.get('h', 0)
                    l = c.get('l', 0)
                    
                    if h > 0:
                        resistance.append({'price': h, 'type': 'resistance', 'tf': tf_name})
                    if l > 0:
                        support.append({'price': l, 'type': 'support', 'tf': tf_name})
            
            # Remove duplicates and sort
            support = sorted(list({s['price']: s for s in support}.values()), 
                           key=lambda x: x['price'], reverse=True)[:3]
            resistance = sorted(list({r['price']: r for r in resistance}.values()), 
                              key=lambda x: x['price'])[:3]
            
            return {
                'support': support,
                'resistance': resistance
            }
        
        except Exception as e:
            print(f"[ERROR] Liquidity: {e}")
            return {'support': [], 'resistance': []}
    
    def calculate_microstructure(self, symbol, market_data):
        """Bid/ask/spread"""
        
        try:
            bid = market_data.get('bid', 0)
            ask = market_data.get('ask', 0)
            spread_points = market_data.get('spread_points', 0)
            
            mid = (bid + ask) / 2 if bid and ask else 0
            spread_pct = (spread_points / mid * 100) if mid > 0 else 0
            
            interpretation = 'Tight spread - liquidity present' if spread_pct < 0.05 else 'Wide spread - low liquidity'
            
            return {
                'bid': bid,
                'ask': ask,
                'spread_pct': spread_pct,
                'interpretation': interpretation
            }
        
        except Exception as e:
            print(f"[ERROR] Microstructure: {e}")
            return {
                'bid': 0,
                'ask': 0,
                'spread_pct': 0,
                'interpretation': 'Error'
            }
    
    def get_session(self, symbol):
        """Current trading session"""
        
        try:
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
        
        except Exception as e:
            print(f"[ERROR] Session: {e}")
            return {
                'current_session': 'Unknown',
                'session_hours': 'Unknown',
                'typical_volatility': 50
            }
    
    def generate_interpretation(self, symbol, data):
        """Readable market description"""
        
        try:
            bias = data.get('bias_stability', {}).get('bias', 'NEUTRAL')
            confidence = data.get('confidence_level', 50)
            regime = data.get('market_regime', {})
            confluence = data.get('confluence_score', 50)
            
            clarity = 'strong clarity' if confidence > 70 else 'moderate clarity' if confidence > 50 else 'weak clarity'
            trend = regime.get('trend', 'unknown').lower()
            volatility = regime.get('volatility', 'normal').lower()
            strength = 'strong' if confluence > 70 else 'moderate' if confluence > 50 else 'weak'
            
            text = f"{symbol} showing {clarity} with {bias.lower()} bias. "
            text += f"Market shows {trend} trend and {volatility} volatility. "
            text += f"Confluence at {confluence:.0f}% indicates {strength} agreement."
            
            return text
        
        except Exception as e:
            print(f"[ERROR] Interpretation: {e}")
            return "Unable to generate interpretation"

analyzer = OracleXAnalyzer()

async def handle_market_data(request):
    """Receive and store market data"""
    try:
        data = await request.json()
        market_data_list = data.get('market_data', [])
        
        print(f"\n[RECEIVE] {len(market_data_list)} symbols at {datetime.now().isoformat()}")
        
        if isinstance(market_data_list, dict):
            market_data_list = [market_data_list]
        
        stored = 0
        for item in market_data_list:
            symbol = item.get('symbol')
            if symbol:
                market_cache[symbol] = item
                stored += 1
        
        print(f"[STORE] {stored} symbols. Cache: {list(market_cache.keys())}\n")
        
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
        
        # Calculate all metrics
        confluence_score = analyzer.calculate_confluence(symbol, timeframes)
        market_regime = analyzer.calculate_market_regime(symbol, timeframes)
        bias_stability = analyzer.calculate_bias_stability(symbol, timeframes)
        multi_timeframe = analyzer.calculate_multi_timeframe_confluence(symbol, timeframes)
        liquidity = analyzer.calculate_liquidity(symbol, timeframes)
        microstructure = analyzer.calculate_microstructure(symbol, market_data)
        session = analyzer.get_session(symbol)
        
        # Confidence = average agreement
        confidence_level = (multi_timeframe.get('agreement_score', 0) + confluence_score) / 2
        
        # Data for interpretation
        analysis_data = {
            'confluence_score': confluence_score,
            'confidence_level': confidence_level,
            'bias_stability': bias_stability,
            'market_regime': market_regime
        }
        
        interpretation = analyzer.generate_interpretation(symbol, analysis_data)
        
        # Build response
        response = {
            'symbol': symbol,
            'price': market_data.get('price', 0),
            'bid': market_data.get('bid', 0),
            'ask': market_data.get('ask', 0),
            'bias': bias_stability.get('bias', 'NEUTRAL'),
            'confluence': confluence_score,
            'confidence': confidence_level,
            'market_regime': market_regime,
            'bias_stability': bias_stability,
            'multi_timeframe': multi_timeframe,
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
    """Get all cached symbols"""
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
    print('✨ ORACLEX PYTHON BACKEND - PRODUCTION')
    print('='*80)
    print('Port: 8080')
    print('='*80 + '\n')
    
    web.run_app(app, port=8080)
