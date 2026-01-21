#!/usr/bin/env python3
import os
import asyncio
from aiohttp import web
import json

# Store received market data
market_cache = {}

async def health(request):
    return web.json_response({"status": "ok", "version": "2.1", "cached_symbols": len(market_cache)})

async def receive_market_data(request):
    """Receive market data from Relay (internal Railway network)"""
    try:
        payload = await request.json()
        market_data = payload.get("market_data", [])
        
        # Store all symbol data
        for sym_data in market_data:
            sym = sym_data.get("symbol")
            if sym:
                market_cache[sym] = sym_data
                print(f"  ✓ {sym}")
        
        print(f"✅ Python received {len(market_data)} symbols from Relay. Cached: {len(market_cache)}")
        return web.json_response({"status": "ok", "stored": len(market_cache)})
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return web.json_response({"error": str(e)}, status=400)

async def get_all_analysis(request):
    """Return analysis for all symbols"""
    analyses = []
    
    symbols = ["XAUUSD", "BTCUSD", "ETHUSD", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]
    
    for sym in symbols:
        if sym in market_cache:
            # We have real data
            price = market_cache[sym].get("price", 0)
            analyses.append({
                "symbol": sym,
                "price": price,
                "confluence": 72.5,
                "confidence": 78.0,
                "market_regime": {"trend": "Strong", "volatility": "Normal", "structure": "Clean"},
                "bias": "BULLISH",
                "interpretation": f"{sym} data received"
            })
        else:
            # Return default
            analyses.append({
                "symbol": sym,
                "price": 0,
                "confluence": 50.0,
                "confidence": 50.0,
                "market_regime": {"trend": "Unknown", "volatility": "Normal", "structure": "Unknown"},
                "bias": "NEUTRAL",
                "interpretation": f"{sym} waiting for data"
            })
    
    return web.json_response({"analyses": analyses})

async def get_symbol_analysis(request):
    """Get analysis for single symbol"""
    sym = request.match_info.get("symbol", "").upper()
    
    if sym in market_cache:
        price = market_cache[sym].get("price", 0)
        return web.json_response({
            "symbol": sym,
            "price": price,
            "confluence": 72.5,
            "confidence": 78.0,
            "market_regime": {"trend": "Strong", "volatility": "Normal", "structure": "Clean"},
            "bias": "BULLISH",
            "interpretation": f"{sym} data received"
        })
    else:
        return web.json_response({
            "symbol": sym,
            "price": 0,
            "confluence": 50.0,
            "confidence": 50.0,
            "market_regime": {"trend": "Unknown", "volatility": "Normal", "structure": "Unknown"},
            "bias": "NEUTRAL",
            "interpretation": f"{sym} waiting for data"
        })

async def main():
    app = web.Application()
    
    # Routes
    app.router.add_get("/", health)
    app.router.add_post("/market-data-v1.6", receive_market_data)
    app.router.add_post("/market-data", receive_market_data)
    app.router.add_get("/latest-analysis", get_all_analysis)
    app.router.add_get("/analysis/{symbol}", get_symbol_analysis)
    
    port = int(os.getenv("PORT", 8080))
    
    print("\n" + "="*80)
    print("✨ ORACLEX PYTHON V2.1 (PRIVATE)")
    print("="*80)
    print(f"🔒 Private Railway network only")
    print(f"🚀 Listening on port {port}")
    print("="*80 + "\n")
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
