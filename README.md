# OracleX Python Backend V2.4

Real-time market analysis engine with dashboard features.

## Features

- **Market Regime Classification** - Trend, volatility, structure detection
- **Bias Stability Tracking** - Directional bias duration monitoring
- **Confluence Breakdown** - Weight contribution analysis
- **Context History** - 60+ minute market state timeline
- **State-Based Statistics** - Historical probability analysis
- **Session Intelligence** - Asia, London, NY session detection

## Deployment on Railway

### Environment Variables

Add these in Railway Project Settings:

```
GEMINI_API_KEY=AIzaSyCx8nWHF7cWrCIU8T7jqgw9Q1pAKZkVCpA
```

### How It Works

1. Reads market data from Railway Relay (`/get-market-state`)
2. Analyzes each symbol with V2.4 features
3. POSTs analysis back to Relay (`/market-analysis`)
4. Frontend fetches complete merged data

### Data Flow

```
MQL5 → Relay (/update-market-state)
↓
Python reads from Relay (/get-market-state)
↓
Python analyzes all 6 symbols (XAUUSD, BTCUSD, SOLUSD, GBPJPY, ETHUSD, XAGUSD)
↓
Python POSTs V2.4 analysis back to Relay (/market-analysis)
↓
Frontend merges MQL5 price data + Python analysis
↓
Dashboard displays complete real-time intelligence
```

## Local Development

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
python3 oraclex_v2_4_updated.py
```

## Production

Railway automatically:
1. Installs dependencies from `requirements.txt`
2. Runs `Procfile` worker process
3. Uses Python 3.11.7 from `runtime.txt`
4. Keeps process running continuously
