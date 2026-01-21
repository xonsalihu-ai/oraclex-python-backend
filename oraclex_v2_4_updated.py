#!/usr/bin/env python3
"""
ORACLEX V2.5+ DIAGNOSTIC - Debug what data EA is sending
"""

from aiohttp import web
import json
from datetime import datetime

SYMBOLS = ['XAUUSD', 'XAGUUSD', 'BTCUSD', 'ETHUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']
market_cache = {}

async def handle_market_data(request):
    try:
        data = await request.json()
        market_data_list = data.get('market_data', [])
        
        print(f"\n{'='*80}")
        print(f"[RECEIVED] {len(market_data_list)} symbols at {datetime.now().isoformat()}")
        print(f"{'='*80}")
        
        for market_data in market_data_list:
            symbol = market_data.get('symbol', 'UNKNOWN')
            price = market_data.get('price', 0)
            has_timeframes = 'timeframes' in market_data
            timeframes_count = len(market_data.get('timeframes', {}))
            
            print(f"\n📊 {symbol}")
            print(f"   Price: ${price}")
            print(f"   Has timeframes data: {has_timeframes}")
            print(f"   Timeframes count: {timeframes_count}")
            
            if has_timeframes:
                for tf_name, tf_data in market_data.get('timeframes', {}).items():
                    if tf_data:
                        candles_count = len(tf_data.get('candles', []))
                        has_indicators = 'indicators' in tf_data
                        print(f"      {tf_name}: {candles_count} candles, indicators={has_indicators}")
                    else:
                        print(f"      {tf_name}: NULL/Empty data")
            else:
                print(f"   ⚠️  NO TIMEFRAMES DATA - this is the problem!")
            
            # Store in cache
            market_cache[symbol] = market_data
        
        print(f"\n✅ Stored {len(market_data_list)} symbols in cache")
        print(f"{'='*80}\n")
        
        return web.json_response({'stored': len(market_data_list), 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        print(f"[ERROR] Market data handler: {e}")
        return web.json_response({'error': str(e)}, status=400)

async def handle_analysis(request):
    try:
        symbol = request.match_info.get('symbol', '').upper()
        
        if symbol not in market_cache:
            print(f"[ANALYSIS] {symbol} - NOT IN CACHE")
            return web.json_response({'error': f'{symbol} not found'}, status=404)
        
        market_data = market_cache[symbol]
        
        # Check what data we have
        has_timeframes = 'timeframes' in market_data and market_data['timeframes']
        
        print(f"[ANALYSIS] {symbol} - has_timeframes={has_timeframes}")
        
        if not has_timeframes:
            print(f"[ANALYSIS] {symbol} - CANNOT ANALYZE: No timeframe data!")
            # Return basic data only
            return web.json_response({
                'symbol': symbol,
                'price': market_data.get('price', 0),
                'bid': market_data.get('bid', 0),
                'ask': market_data.get('ask', 0),
                'error': 'Insufficient data - no timeframes',
                'confluence': 0,
                'confidence': 0,
                'bias': 'NEUTRAL',
                'market_regime': {'trend': 'Unknown', 'volatility': 'Unknown', 'structure': 'Unknown', 'volatility_points': 50},
                'bias_stability': {'bias': 'NEUTRAL', 'active_since_minutes': 0, 'multi_tf_agreement': 0},
                'multi_timeframe': {'dominant_tf': 'Unknown', 'agreement_score': 0, 'timeframe_bias': {}},
                'liquidity': {'support': [], 'resistance': []},
                'microstructure': {'bid': market_data.get('bid', 0), 'ask': market_data.get('ask', 0), 'spread_pct': 0},
                'risk_opportunity': {'risk_score': 50, 'opportunity_score': 0, 'grade': 'C'},
                'session': {'current_session': 'Unknown', 'typical_volatility': 50},
                'interpretation': f'{symbol} - Waiting for timeframe data from EA'
            })
        
        # If we have timeframes, do basic analysis
        timeframes = market_data.get('timeframes', {})
        
        # Simple confluence from available data
        confluence_score = 50  # Default
        if 'H1' in timeframes and timeframes['H1'].get('candles'):
            confluence_score = 60
        if 'H4' in timeframes and timeframes['H4'].get('candles'):
            confluence_score = 70
        
        analysis = {
            'symbol': symbol,
            'price': market_data.get('price', 0),
            'bid': market_data.get('bid', 0),
            'ask': market_data.get('ask', 0),
            'confluence': confluence_score,
            'confidence': confluence_score / 2,
            'bias': 'NEUTRAL',
            'market_regime': {'trend': 'Unknown', 'volatility': 'Unknown', 'structure': 'Unknown', 'volatility_points': 50},
            'bias_stability': {'bias': 'NEUTRAL', 'active_since_minutes': 0, 'multi_tf_agreement': 0},
            'multi_timeframe': {'dominant_tf': 'H1', 'agreement_score': 0, 'timeframe_bias': {}},
            'liquidity': {'support': [], 'resistance': []},
            'microstructure': {'bid': market_data.get('bid', 0), 'ask': market_data.get('ask', 0), 'spread_pct': 0},
            'risk_opportunity': {'risk_score': 50, 'opportunity_score': confluence_score, 'grade': 'B'},
            'session': {'current_session': 'Unknown', 'typical_volatility': 50},
            'interpretation': f'{symbol} - Limited analysis: {len(timeframes)} timeframes available, confluence {confluence_score:.0f}%'
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
                analyses.append({
                    'symbol': symbol,
                    'price': market_cache[symbol].get('price', 0),
                    'confluence': 50,
                    'bias': 'NEUTRAL'
                })
        return web.json_response({'analyses': analyses})
    except Exception as e:
        print(f"[ERROR] Latest analysis: {e}")
        return web.json_response({'analyses': [], 'error': str(e)}, status=500)

async def handle_health(request):
    return web.json_response({'status': 'OK', 'cached_symbols': list(market_cache.keys())})

app = web.Application()
app.router.add_post('/market-data-v1.6', handle_market_data)
app.router.add_get('/analysis/{symbol}', handle_analysis)
app.router.add_get('/latest-analysis', handle_latest_analysis)
app.router.add_get('/', handle_health)

if __name__ == '__main__':
    print('\n' + '='*80)
    print('🔍 ORACLEX V2.5+ DIAGNOSTIC MODE')
    print('='*80)
    print('This will show exactly what data the EA is sending\n')
    print('🚀 Listening on port 8080')
    print('='*80 + '\n')
    
    web.run_app(app, port=8080)
