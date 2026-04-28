# AI Trader Mobile

Personal-use iOS client for the `ai-trader` backend.

## Scope

This mobile app currently supports:

- sign in against `/auth/login`
- read `/health`
- read `/dashboard`
- read `/autopilot/status`
- request `/trade/preview`
- request `/trade/action`

It can now send both dry-run and live trade actions, depending on the selected mode in the trade desk.

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
2. optional: create `.env` from `.env.example` and set `EXPO_PUBLIC_API_BASE_URL`
3. `npm install`
4. `npm run test`
5. `npm run typecheck`
6. `npm run ios`

This repo now defaults the mobile app to the hosted backend:

- `https://cella.website/api`

## Hosted backend

If you want the iPhone app to work without your Mac, the backend must be hosted remotely.

For this repo's Oracle setup, the public entrypoint should normally be:

- `https://cella.website/api`

Do not point the phone at Oracle port `8765` directly. In this repo, the Oracle deployment is designed to expose only `80/443` via Caddy and proxy `/api/*` internally to the backend.

Example mobile `.env`:

```bash
EXPO_PUBLIC_API_BASE_URL=https://cella.website/api
```

Then the app starts with that hosted API as its default backend URL.

## Local fallback

If you still want to test against your Mac sometimes:

- use `http://127.0.0.1:8765` in the iOS simulator
- use `http://YOUR_MAC_LAN_IP:8765` on a real iPhone over local Wi-Fi
