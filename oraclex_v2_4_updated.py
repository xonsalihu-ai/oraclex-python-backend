#!/usr/bin/env python3
"""
ORACLEX - MINIMAL DIAGNOSTIC VERSION
Just receive -> store -> return data. No complex calculations.
Purpose: Verify the relay <-> Python communication works end-to-end.
"""

from aiohttp import web
import json
from datetime import datetime

# Simple in-memory storage
data_store = {}

async def handle_market_data(request):
    """Receive market data from relay"""
    try:
        body = await request.json()
        market_data = body.get('market_data', [])
        
        print(f"\n{'='*80}")
        print(f"[RECEIVE] Market data at {datetime.now().isoformat()}")
        print(f"[RECEIVE] Got {len(market_data)} symbols")
        
        for item in market_data:
            symbol = item.get('symbol')
            if symbol:
                data_store[symbol] = item
                print(f"  ✅ {symbol} -> STORED")
        
        print(f"[STORE] Current keys: {list(data_store.keys())}")
        print(f"{'='*80}\n")
        
        return web.json_response({'status': 'stored', 'count': len(market_data)})
    except Exception as e:
        print(f"[ERROR] {e}")
        return web.json_response({'error': str(e)}, status=400)

async def handle_analysis(request):
    """Return analysis for a symbol"""
    symbol = request.match_info.get('symbol', '').upper()
    
    print(f"[REQUEST] /analysis/{symbol}")
    print(f"[LOOKUP] Available symbols: {list(data_store.keys())}")
    
    if symbol not in data_store:
        print(f"  ❌ {symbol} NOT FOUND")
        return web.json_response({'error': f'{symbol} not found'}, status=404)
    
    print(f"  ✅ {symbol} FOUND")
    
    data = data_store[symbol]
    
    # Return simple analysis
    response = {
        'symbol': symbol,
        'price': data.get('price', 0),
        'bid': data.get('bid', 0),
        'ask': data.get('ask', 0),
        'spread_points': data.get('spread_points', 0),
        'timeframes_count': len(data.get('timeframes', {})),
        'bias': 'BULLISH',  # Dummy
        'confluence': 50,  # Dummy
        'confidence': 50,  # Dummy
        'market_regime': {
            'trend': 'Unknown',
            'volatility': 'Normal',
            'structure': 'Unknown'
        },
        'bias_stability': {
            'bias': 'NEUTRAL',
            'active_since_minutes': 0,
            'multi_tf_agreement': 0
        },
        'multi_timeframe': {
            'dominant_tf': 'H1',
            'agreement_score': 0,
            'timeframe_bias': {}
        },
        'liquidity': {
            'support': [],
            'resistance': []
        },
        'microstructure': {
            'bid': data.get('bid', 0),
            'ask': data.get('ask', 0),
            'spread_pct': 0
        },
        'session': {
            'current_session': 'Unknown',
            'typical_volatility': 50
        },
        'interpretation': f'Diagnostic response for {symbol}'
    }
    
    print(f"  📤 Returning analysis")
    return web.json_response(response)

async def handle_health(request):
    """Health check"""
    return web.json_response({
        'status': 'ok',
        'stored_symbols': list(data_store.keys()),
        'timestamp': datetime.now().isoformat()
    })

app = web.Application()
app.router.add_post('/market-data-v1.6', handle_market_data)
app.router.add_get('/analysis/{symbol}', handle_analysis)
app.router.add_get('/latest-analysis', handle_health)
app.router.add_get('/', handle_health)

if __name__ == '__main__':
    print('\n' + '='*80)
    print('🔧 ORACLEX DIAGNOSTIC BACKEND')
    print('='*80)
    print('Purpose: Test relay <-> Python communication')
    print('Port: 8080')
    print('='*80 + '\n')
    
    web.run_app(app, port=8080)
