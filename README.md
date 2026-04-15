# Harvard Algorithmic Trading with AI

## [![Video Tutorial](https://img.shields.io/badge/Watch-Video%20Tutorial-red?style=for-the-badge&logo=youtube)](https://youtu.be/Vu62g43_1aE)

## The RBI System: Research, Backtest, Implement

Welcome to the Harvard Algorithmic Trading with AI repository. This open-source project teaches students how to become algorithmic traders by following a systematic approach to automating trading strategies.

### Why Algorithmic Trading?

Most traders face two major challenges:

- **Emotional decision-making** leading to poor trades
- **Time waste** staring at screens all day with little to no profit

This project is built on the hypothesis shared by Jim Simons (who built a $30 billion net worth): **the only way to trade effectively is with robots**.

### The RBI System Explained

#### 1. Research (R)

Research is the foundation of algorithmic trading. Before writing a single line of code, you must:

- Study proven trading strategies
- Understand market behaviors
- Analyze what works for different market conditions
- Build a solid hypothesis

#### 2. Backtest (B)

Backtesting validates your strategy against historical data:

- Test your strategy against OHLCV (Open, High, Low, Close, Volume) data
- Evaluate performance metrics
- Refine parameters
- Avoid survivorship bias

#### 3. Implement (I)

Only after successful research and backtesting should you:

- Code your strategy
- Set up proper risk management
- Deploy with careful monitoring
- Scale gradually

### What This Repository Contains

- Step-by-step guides for each stage of the RBI system
- Code examples and templates
- Resources for data collection and analysis
- A starter model-training workflow for supervised learning on historical candles
- AI integration techniques for strategy enhancement
- Case studies and lessons learned

### Learning From Mistakes

I'll share personal stories of rushing to implementation without proper research and backtesting, resulting in significant losses. These cautionary tales will emphasize why the RBI system exists and how to avoid common pitfalls.

### Important Note

While historical performance can provide confidence in a strategy, past results never guarantee future performance. The RBI system is designed to maximize your probability of success, not to eliminate risk entirely.

## Getting Started

[More content to be added as the project develops]

## Resources

### Tools I Use

- **[Cursor](https://cursor.sh/)** - AI-powered IDE where all coding for this project takes place
- **[Flow Pro](https://wisprflow.ai/)** - Voice-to-text AI tool used for documentation and code comments

### Prerequisites

- **[CS50: Introduction to Computer Science](https://www.youtube.com/watch?v=3LPJfIKxwWc&list=PLhQjrBD2T381WAHyx1pq-sBfykqMBI7V4)** - This Harvard course is a prerequisite to understanding the programming concepts in this class

### Python Packages

- pandas - super important to learn its like excel for python
- backtesting.py - this is how i backtest
- yfinance - this is where we get free daily data
- talib - indicators for finance (optional local dependency for backtest/bot scripts)
- ccxt - if you want to connect to other exchanges other than hyperliquid

## Oracle Free VM Deployment

If you want full Docker Compose deployment on a free VM, use the Oracle guide:

- `deploy/oracle/ORACLE_DEPLOY.md`

Quick start on the VM:

1. `./deploy/oracle/bootstrap_vm.sh`
2. Configure `.env` (you can copy from `.env.example`)
3. `./deploy/oracle/deploy_app.sh`

## App Architecture

The app is now split into:

- React frontend in `frontend/` for controls, charts, wallet/account views, and AI support UI
- Python backend in `backend/` for signals, autopilot, trade previews, execution, wallet access, and AI services

### Backend API

Run locally:

`python -m backend.app`

Endpoints:

- `GET /health`
- `GET /signal`
- `GET /dashboard`
- `GET /autopilot/status`
- `POST /autopilot/start`
- `POST /autopilot/stop`
- `POST /trade/preview`
- `POST /trade/action`
- `POST /support/chat`
- `POST /ai/command`

`/signal` accepts optional query parameters:

- `buy_threshold`
- `sell_threshold`
- `adaptive_threshold_enabled`

### Docker split stack

- Backend image: `Dockerfile.backend`
- Frontend image: `Dockerfile.frontend`
- Backend deps: `requirements-api.txt`
- Local stack: `docker-compose.yml`

Start both services locally:

`docker compose up --build`

Then open:

- Frontend: `http://localhost:3000`
- Backend health: `http://localhost:8765/health`

## Remote Windows Build / Train

The repo includes GitHub Actions workflows for a Windows self-hosted runner:

- `.github/workflows/deploy-windows.yml`
- `.github/workflows/train-windows.yml`

Expected runner labels:

- `self-hosted`
- `windows`
- `x64`
- `ai-trader-windows`

Expected runner setup:

- the repo is checked out on the Windows machine
- Docker Desktop is installed and running
- `.env` exists in the repo workspace on Windows
- Python 3.12+ is available for training jobs

Workflow behavior:

- deploy workflow runs automatically on pushes to `main`
- training workflow runs manually from GitHub Actions with `workflow_dispatch`

This lets you push from the Mac and have the Windows machine rebuild or train remotely without sitting near the desktop.
