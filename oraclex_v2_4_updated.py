#!/usr/bin/env python3
"""
ORACLEX V2.5+ FINAL - Fixed to handle EA data format correctly
"""

from aiohttp import web
import pandas as pd
import numpy as np
import json
from datetime import datetime

SYMBOLS = ['XAUUSD', 'XAGUUSD', 'BTCUSD', 'ETHUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']
market_cache = {}

class OracleXAnalyzer:
    
    def calculate_multi_timeframe_confluence(self, symbol, timeframes_data):
        if not timeframes_data:
            return {'dominant_tf': 'Unknown', 'agreement_score': 0, 'timeframe_bias': {}}
        
        timeframe_biases = {}
        bullish_count = 0
        bearish_count = 0
        
        try:
            for tf_name, tf_data in timeframes_data.items():
                if not tf_data or not tf_data.get('candles') or len(tf_data['candles']) < 2:
                    continue
                
                candles = tf_data['candles']
                recent = candles[-1]
                ema_trend = 'BULLISH' if recent.get('c', 0) > recent.get('ema_20', recent.get('c', 0)) else 'BEARISH'
                
                timeframe_biases[tf_name] = {
                    'bias': ema_trend,
                    'close': recent.get('c', 0),
                    'strength': abs(recent.get('c', 0) - recent.get('ema_20', recent.get('c', 0)))
                }
                
                if ema_trend == 'BULLISH':
                    bullish_count += 1
                else:
                    bearish_count += 1
        except Exception as e:
            print(f"[ERROR] Multi-TF calc: {e}")
        
        dominant_tf = 'H1' if 'H1' in timeframes_data else 'H4' if 'H4' in timeframes_data else (list(timeframe_biases.keys())[0] if timeframe_biases else 'Unknown')
        total_tfs = len(timeframe_biases)
        agreement_score = (max(bullish_count, bearish_count) / total_tfs * 100) if total_tfs > 0 else 0
        
        return {
            'dominant_tf': dominant_tf,
            'agreement_score': agreement_score,
            'timeframe_bias': timeframe_biases,
            'bullish_tfs': bullish_count,
            'bearish_tfs': bearish_count,
            'total_tfs': total_tfs
        }
    
    def calculate_liquidity_levels(self, symbol, market_data):
        if not market_data or 'timeframes' not in market_data:
            return {'levels': [], 'support': [], 'resistance': []}
        
        levels = []
        
        try:
            for tf_name, tf_data in market_data.get('timeframes', {}).items():
                if tf_name not in ['D1', 'H4']:
                    continue
                
                candles = tf_data.get('candles', []) if tf_data else []
                if len(candles) < 2:
                    continue
                
                for candle in candles[-20:]:
                    high = candle.get('h', 0)
                    low = candle.get('l', 0)
                    
                    if high > 0:
                        levels.append({'price': high, 'type': 'resistance', 'tf': tf_name, 'strength': 'high'})
                    if low > 0:
                        levels.append({'price': low, 'type': 'support', 'tf': tf_name, 'strength': 'high'})
        except Exception as e:
            print(f"[ERROR] Liquidity calc: {e}")
        
        support_levels = sorted([l for l in levels if l['type'] == 'support'], key=lambda x: x['price'], reverse=True)
        resistance_levels = sorted([l for l in levels if l['type'] == 'resistance'], key=lambda x: x['price'])
        
        return {
            'levels': levels[:10],
            'support': support_levels[:3],
            'resistance': resistance_levels[:3],
            'nearest_support': support_levels[0] if support_levels else None,
            'nearest_resistance': resistance_levels[0] if resistance_levels else None
        }
    
    def calculate_microstructure(self, symbol, market_data):
        try:
            bid = market_data.get('bid', 0)
            ask = market_data.get('ask', 0)
            spread_points = market_data.get('spread_points', 0)
            
            mid = (bid + ask) / 2 if bid and ask else 0
            spread_pct = (spread_points / mid * 100) if mid > 0 else 0
            
            return {
                'bid': bid,
                'ask': ask,
                'mid': mid,
                'spread_points': spread_points,
                'spread_pct': spread_pct,
                'imbalance': 0.5,
                'interpretation': 'Tight spread - liquidity present' if spread_pct < 0.05 else 'Wide spread - low liquidity'
            }
        except Exception as e:
            print(f"[ERROR] Microstructure calc: {e}")
            return {
                'bid': 0,
                'ask': 0,
                'mid': 0,
                'spread_points': 0,
                'spread_pct': 0,
                'imbalance': 0.5,
                'interpretation': 'Error calculating'
            }
    
    def calculate_risk_opportunity(self, symbol, confluence_score, volatility_regime, microstructure):
        try:
            spread_risk = min(microstructure.get('spread_pct', 0) * 20, 30)
            volatility_risk = volatility_regime.get('volatility_points', 50)
            risk_score = min(spread_risk + volatility_risk, 100)
            
            opportunity_score = min(confluence_score, 100)
            overall_score = min(max((opportunity_score - (risk_score / 2)) / 1.5, 0), 100)
            
            if overall_score >= 80:
                grade = 'A+'
            elif overall_score >= 70:
                grade = 'A'
            elif overall_score >= 60:
                grade = 'B+'
            elif overall_score >= 50:
                grade = 'B'
            else:
                grade = 'C'
            
            return {
                'risk_score': risk_score,
                'opportunity_score': opportunity_score,
                'overall_score': overall_score,
                'grade': grade,
                'risk_level': 'Low' if risk_score < 33 else 'Medium' if risk_score < 66 else 'High'
            }
        except Exception as e:
            print(f"[ERROR] Risk/Opp calc: {e}")
            return {
                'risk_score': 50,
                'opportunity_score': 50,
                'overall_score': 50,
                'grade': 'B',
                'risk_level': 'Medium'
            }
    
    def calculate_market_regime(self, symbol, market_data):
        try:
            timeframes = market_data.get('timeframes', {})
            
            if not timeframes or 'H1' not in timeframes:
                return {'trend': 'Unknown', 'volatility': 'Normal', 'structure': 'Unknown', 'volatility_points': 50}
            
            h1_data = timeframes['H1']
            candles = h1_data.get('candles', []) if h1_data else []
            
            if len(candles) < 2:
                return {'trend': 'Insufficient Data', 'volatility': 'Normal', 'structure': 'Unknown', 'volatility_points': 50}
            
            recent = candles[-1]
            trend = 'Strong Up' if recent.get('c', 0) > recent.get('ema_20', 0) else 'Strong Down'
            
            atr = h1_data.get('indicators', {}).get('atr', 0) if h1_data.get('indicators') else 0
            atr_mean = np.mean([max(0.0001, c.get('h', 0) - c.get('l', 0)) for c in candles[-20:]]) if candles else 1
            
            if atr > atr_mean * 1.5:
                volatility, volatility_points = 'Extreme', 80
            elif atr > atr_mean * 1.2:
                volatility, volatility_points = 'Elevated', 60
            elif atr < atr_mean * 0.8:
                volatility, volatility_points = 'Quiet', 30
            else:
                volatility, volatility_points = 'Normal', 50
            
            structure = 'Clean' if len([c for c in candles[-10:] if c.get('h', 0) > c.get('o', 0)]) >= 5 else 'Choppy'
            
            return {'trend': trend, 'volatility': volatility, 'structure': structure, 'volatility_points': volatility_points}
        except Exception as e:
            print(f"[ERROR] Market regime calc: {e}")
            return {'trend': 'Unknown', 'volatility': 'Normal', 'structure': 'Unknown', 'volatility_points': 50}
    
    def calculate_bias_stability(self, symbol, market_data):
        try:
            timeframes = market_data.get('timeframes', {})
            
            if not timeframes or 'M5' not in timeframes:
                return {'bias': 'NEUTRAL', 'active_since_minutes': 0, 'multi_tf_agreement': 0, 'flip_probability': 50}
            
            m5_data = timeframes['M5']
            candles = m5_data.get('candles', []) if m5_data else []
            
            if len(candles) < 2:
                return {'bias': 'NEUTRAL', 'active_since_minutes': 0, 'multi_tf_agreement': 0, 'flip_probability': 50}
            
            recent = candles[-1]
            bias = 'BULLISH' if recent.get('c', 0) > recent.get('ema_20', 0) else 'BEARISH'
            
            active_candles = 1
            for i in range(len(candles) - 2, -1, -1):
                candle = candles[i]
                candle_bias = 'BULLISH' if candle.get('c', 0) > candle.get('ema_20', 0) else 'BEARISH'
                if candle_bias == bias:
                    active_candles += 1
                else:
                    break
            
            active_since_minutes = active_candles * 5
            
            agreement = sum(1 for tf_data in timeframes.values() if tf_data and tf_data.get('candles') and ('BULLISH' if tf_data['candles'][-1].get('c', 0) > tf_data['candles'][-1].get('ema_20', 0) else 'BEARISH') == bias)
            multi_tf_agreement = (agreement / len(timeframes) * 100) if timeframes else 0
            
            flip_probability = max(min(50 - (multi_tf_agreement / 4), 100), 0)
            
            return {'bias': bias, 'active_since_minutes': active_since_minutes, 'multi_tf_agreement': multi_tf_agreement, 'flip_probability': flip_probability}
        except Exception as e:
            print(f"[ERROR] Bias stability calc: {e}")
            return {'bias': 'NEUTRAL', 'active_since_minutes': 0, 'multi_tf_agreement': 0, 'flip_probability': 50}
    
    def calculate_dynamic_confluence(self, symbol, market_data):
        try:
            timeframes = market_data.get('timeframes', {})
            confluence_points = 0
            total_indicators = 0
            
            for tf_name, tf_data in timeframes.items():
                candles = tf_data.get('candles', []) if tf_data else []
                if not candles:
                    continue
                
                indicators = tf_data.get('indicators', {}) if tf_data else {}
                recent = candles[-1]
                
                rsi = indicators.get('rsi', 50)
                if rsi > 60 or rsi < 40:
                    confluence_points += 1
                total_indicators += 1
                
                macd_hist = indicators.get('macd_histogram', 0)
                if macd_hist != 0:
                    confluence_points += 1
                total_indicators += 1
                
                if recent.get('c', 0) > recent.get('ema_20', 0):
                    confluence_points += 1
                total_indicators += 1
            
            confluence_score = (confluence_points / total_indicators * 100) if total_indicators > 0 else 50
            
            return {'confluence_score': confluence_score, 'confluence_points': confluence_points, 'total_indicators': total_indicators}
        except Exception as e:
            print(f"[ERROR] Confluence calc: {e}")
            return {'confluence_score': 50, 'confluence_points': 0, 'total_indicators': 0}
    
    def get_session_intelligence(self, symbol):
        try:
            from datetime import datetime, timezone
            
            utc_hour = datetime.now(timezone.utc).hour
            
            if 0 <= utc_hour < 9:
                session, volatility = 'Asia', 30
            elif 8 <= utc_hour < 17:
                session, volatility = 'Europe', 70
            elif 16 <= utc_hour < 24:
                session, volatility = 'US', 80
            else:
                session, volatility = 'Overlap', 60
            
            sessions_map = {
                'Asia': '00:00-09:00 UTC',
                'Europe': '08:00-17:00 UTC',
                'US': '16:00-23:59 UTC',
                'Overlap': 'Overlapping hours'
            }
            
            return {'current_session': session, 'typical_volatility': volatility, 'session_hours': sessions_map.get(session)}
        except Exception as e:
            print(f"[ERROR] Session calc: {e}")
            return {'current_session': 'Unknown', 'typical_volatility': 50, 'session_hours': 'Unknown'}
    
    def generate_interpretation(self, symbol, analysis_data):
        try:
            confluence = analysis_data.get('confluence_score', 0)
            confidence = analysis_data.get('confidence_level', 0)
            bias = analysis_data.get('bias_stability', {}).get('bias', 'NEUTRAL')
            regime = analysis_data.get('market_regime', {})
            
            interpretation = f"{symbol} showing "
            interpretation += "strong clarity" if confidence > 70 else "moderate clarity" if confidence > 50 else "weak clarity"
            interpretation += f" with {bias.lower()} bias. Market shows {regime.get('trend', 'Unknown').lower()} trend and {regime.get('volatility', 'Normal').lower()} volatility. "
            interpretation += f"Confluence at {confluence:.0f}% indicates {'strong' if confluence > 70 else 'moderate' if confluence > 50 else 'weak'} agreement."
            
            return interpretation
        except Exception as e:
            print(f"[ERROR] Interpretation gen: {e}")
            return "Unable to generate interpretation"

analyzer = OracleXAnalyzer()

async def handle_market_data(request):
    try:
        data = await request.json()
        market_data_list = data.get('market_data', [])
        
        print(f"\n{'='*80}")
        print(f"[RECEIVED] {len(market_data_list)} symbols")
        print(f"{'='*80}")
        
        # Handle both list and dict formats
        if isinstance(market_data_list, dict):
            market_data_list = [market_data_list]
        
        for idx, market_data in enumerate(market_data_list):
            try:
                symbol = market_data.get('symbol', f'SYMBOL_{idx}')
                market_cache[symbol] = market_data
                print(f"✅ {symbol} - stored")
            except Exception as e:
                print(f"⚠️  Error processing item {idx}: {e}")
        
        print(f"✅ Processed {len(market_data_list)} symbols\n")
        return web.json_response({'stored': len(market_data_list), 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        print(f"[ERROR] Market data handler: {e}")
        return web.json_response({'error': str(e)}, status=400)

async def handle_analysis(request):
    try:
        symbol = request.match_info.get('symbol', '').upper()
        
        if symbol not in market_cache:
            return web.json_response({'error': f'{symbol} not found'}, status=404)
        
        market_data = market_cache[symbol]
        
        multi_tf = analyzer.calculate_multi_timeframe_confluence(symbol, market_data.get('timeframes', {}))
        liquidity = analyzer.calculate_liquidity_levels(symbol, market_data)
        microstructure = analyzer.calculate_microstructure(symbol, market_data)
        regime = analyzer.calculate_market_regime(symbol, market_data)
        bias = analyzer.calculate_bias_stability(symbol, market_data)
        confluence = analyzer.calculate_dynamic_confluence(symbol, market_data)
        session = analyzer.get_session_intelligence(symbol)
        
        risk_opp = analyzer.calculate_risk_opportunity(symbol, confluence.get('confluence_score', 50), regime, microstructure)
        
        confidence_level = (multi_tf.get('agreement_score', 0) + confluence.get('confluence_score', 0)) / 2
        
        analysis_data = {
            'price': market_data.get('price', 0),
            'confluence_score': confluence.get('confluence_score', 0),
            'confidence_level': confidence_level,
            'bias_stability': bias,
            'market_regime': regime,
            'liquidity': liquidity,
            'microstructure': microstructure,
            'risk_opportunity': risk_opp
        }
        
        interpretation = analyzer.generate_interpretation(symbol, analysis_data)
        
        analysis = {
            'symbol': symbol,
            'price': market_data.get('price', 0),
            'bid': market_data.get('bid', 0),
            'ask': market_data.get('ask', 0),
            'bias': bias.get('bias'),
            'confluence': confluence.get('confluence_score', 0),
            'confidence': confidence_level,
            'market_regime': regime,
            'bias_stability': bias,
            'multi_timeframe': multi_tf,
            'liquidity': liquidity,
            'microstructure': microstructure,
            'risk_opportunity': risk_opp,
            'session': session,
            'interpretation': interpretation
        }
        
        return web.json_response(analysis)
    except Exception as e:
        print(f"[ERROR] Analysis handler: {e}")
        return web.json_response({'error': str(e)}, status=500)

async def handle_latest_analysis(request):
    try:
        analyses = []
        
        for symbol in SYMBOLS:
            if symbol in market_cache:
                market_data = market_cache[symbol]
                confluence = analyzer.calculate_dynamic_confluence(symbol, market_data)
                regime = analyzer.calculate_market_regime(symbol, market_data)
                bias = analyzer.calculate_bias_stability(symbol, market_data)
                
                analyses.append({
                    'symbol': symbol,
                    'price': market_data.get('price', 0),
                    'confluence': confluence.get('confluence_score', 0),
                    'bias': bias.get('bias'),
                    'regime': regime.get('volatility')
                })
        
        return web.json_response({'analyses': analyses})
    except Exception as e:
        print(f"[ERROR] Latest analysis handler: {e}")
        return web.json_response({'error': str(e)}, status=500)

async def handle_health(request):
    return web.json_response({'status': 'OK', 'cached_symbols': list(market_cache.keys()), 'timestamp': datetime.now().isoformat()})

app = web.Application()
app.router.add_post('/market-data-v1.6', handle_market_data)
app.router.add_get('/analysis/{symbol}', handle_analysis)
app.router.add_get('/latest-analysis', handle_latest_analysis)
app.router.add_get('/', handle_health)

if __name__ == '__main__':
    print('\n' + '='*80)
    print('✨ ORACLEX V2.5+ - INSTITUTIONAL BACKEND (FINAL FIX)')
    print('='*80)
    print('🚀 Listening on port 8080')
    print('='*80 + '\n')
    
    web.run_app(app, port=8080)
