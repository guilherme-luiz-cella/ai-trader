# Backend

This directory contains the Python backend for the app.

Main entrypoint:

- `python -m backend.app`

Remote automation:

- GitHub Actions deploy workflow: `.github/workflows/deploy-windows.yml`
- GitHub Actions train workflow: `.github/workflows/train-windows.yml`

Responsibilities:

- model-backed signal generation
- dashboard data aggregation
- autopilot start/stop/status
- live market capture
- wallet and account snapshots
- trade preview and execution
- AI command and support chat endpoints
- optional macro data fetch from FRED

Key files:

- `app.py`: HTTP server and route wiring
- `services.py`: trading, wallet, autopilot, and support logic

## API Docs

The backend now exposes a docs route:

- `GET /docs`

This returns endpoint and environment details in JSON, including FRED support.

## FRED Integration

FRED (Federal Reserve Economic Data) is available via:

- `GET /macro/fred?series_id=DFF&limit=24`

Query parameters:

- `series_id` (required in practice, default `DFF`)
- `limit` (optional, default `24`, max `1000`)
- `start_date` (optional, format `YYYY-MM-DD`)
- `end_date` (optional, format `YYYY-MM-DD`)

Required environment variable:

- `FRED_API_KEY`

Optional environment variables:

- `FRED_BASE_URL` (default `https://api.stlouisfed.org/fred`)
- `FRED_TIMEOUT_SECONDS` (default `20`)
- `FRED_BYPASS_ENV_PROXY` (default `true`)
