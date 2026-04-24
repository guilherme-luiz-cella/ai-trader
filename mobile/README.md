# AI Trader Mobile

Personal-use iOS client for the `ai-trader` backend.

## Scope

This mobile app is intentionally dry-run only:

- sign in against `/auth/login`
- read `/health`
- read `/dashboard`
- read `/autopilot/status`
- request `/trade/preview`

It does **not** call `/trade/action` or any live execution endpoint.

## DDD structure

- `src/domain`
  - entities and value objects for session, health, signal, dashboard, autopilot, and trade preview
- `src/application`
  - repository ports and use cases for sign-in, overview loading, and trade preview orchestration
- `src/infrastructure`
  - HTTP client plus backend repository adapter and payload mapping
- `src/interface`
  - React Native screen, hooks, styles, and presentational components

## Run

1. `cd /Users/guilherme-luiz-cella/Desktop/ai-trader/mobile`
2. `npm install`
3. `npm run test`
4. `npm run typecheck`
5. `npm run ios`

For a real iPhone on the same network, do not use `127.0.0.1`. Use your Mac LAN IP or a hosted HTTPS URL for the backend field inside the app.
