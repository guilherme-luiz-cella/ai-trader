# Backend

This directory contains the Python backend for the app.

Main entrypoint:

- `python -m backend.app`

Responsibilities:

- model-backed signal generation
- dashboard data aggregation
- autopilot start/stop/status
- live market capture
- wallet and account snapshots
- trade preview and execution
- AI command and support chat endpoints

Key files:

- `app.py`: HTTP server and route wiring
- `services.py`: trading, wallet, autopilot, and support logic
