# Implement: Account Connection

Use this folder to verify your trading account connection before running any live strategy.

## 1) Check account connection (read-only)

Run:

```bash
./.venv/bin/python implement/connect_account.py
```

What this checks:

- `HYPER_LIQUID_KEY` is loaded from `.env`
- your account address is valid
- account value and margin usage
- open orders and open positions
- current best bid/ask for `CHECK_SYMBOL` (default `BTC`)

## 2) Optional symbol override

Set in `.env`:

```env
CHECK_SYMBOL=ETH
```

Then rerun the connection check.

## 3) Safety reminder

`connect_account.py` is read-only and does not place orders.
Keep strategy execution in dry-run mode until risk limits are configured.
