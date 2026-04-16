# Oracle Cloud Deployment Guide

This is the recommended practical Oracle Cloud deployment for this project.

## Recommended architecture

Use Oracle Cloud as the main hosted environment for:

- Python backend API
- React frontend
- autopilot runtime
- durable persistence and restart recovery files
- alerting and WhatsApp escalation triggers

Use a separate inference runtime for the trained model unless you intentionally deploy a much smaller quantized model.

## Best practical Oracle layout

### Oracle VM

Run one Ubuntu VM on OCI Compute, preferably an Always Free Ampere shape if available for your tenancy.

Host on that VM:

- `signal-api` container for the Python backend
- `web` container for static frontend + reverse proxy via Caddy
- persistent Docker volumes for:
  - autopilot runtime memory
  - reconciliation journals
  - Caddy certificates/config

### Reverse proxy

Use Caddy in front of the backend and frontend:

- serves the React build
- proxies `/api/*` to the backend
- can terminate HTTPS automatically when `APP_DOMAIN` and `ACME_EMAIL` are configured

### Persistence

Persistence is file-based today and is already wired for unattended hardening:

- `research/runtime_memory`
- `autopilot_state.json`
- execution journal JSONL
- run summary JSONL

These are persisted through Docker volumes in `docker-compose.oracle.yml`.

### Notifications

Use Twilio for WhatsApp warnings first.

Optional higher-severity paging:

- PagerDuty Events API
- Twilio SMS/WhatsApp fallback

### AI runtime

Do **not** make the Oracle free-tier VM the primary host for the full trained local model by default.

Best practical choice:

- app/backend/autopilot on Oracle
- AI inference separate

Use one of:

1. external OpenAI-compatible runtime
2. your own stronger remote inference box
3. a smaller quantized model on Oracle only if latency/quality are acceptable

## Why the trained model should usually be separated

The current app is a better fit for Oracle CPU hosting than for heavyweight local-transformers inference on a free-tier VM.

Reasons:

- Oracle Always Free compute is good for app hosting, not strong local model inference
- no guaranteed GPU
- local `transformers` inference for your trained model is likely too heavy for a stable low-cost unattended bot host
- inference spikes can compete with autopilot/network/persistence work

Recommended runtime split:

- Oracle VM: frontend, backend, autopilot, persistence, notifications
- separate model runtime: LM Studio/vLLM/another OpenAI-compatible endpoint

If you still want Oracle-hosted inference later:

- switch to a quantized smaller model
- use a server-based runtime instead of full `local_transformers`
- validate latency before enabling live autopilot dependence on it

## What the Oracle stack now does

`docker-compose.oracle.yml` now runs:

- `signal-api` internally on the Docker network
- `web` publicly on ports `80` and `443`

The backend is no longer intended to be exposed directly on `8765` publicly in Oracle.

## Files used

- `docker-compose.oracle.yml`
- `Dockerfile.backend`
- `Dockerfile.web`
- `deploy/caddy/Caddyfile`
- `deploy/oracle/bootstrap_vm.sh`
- `deploy/oracle/deploy_app.sh`
- `deploy/oracle/deploy_api_only.sh`
- `.env.oracle.example`

## Environment variables you should set

### Core app

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `BINANCE_TESTNET`
- `CHECK_SYMBOL`
- `DECISION_MIN_CONFIDENCE`
- `PROFIT_PARKING_STABLE_ASSET`

### Oracle web/proxy

- `APP_DOMAIN=your.domain.com`
- `ACME_EMAIL=you@example.com`

If you do not have a domain yet:

- you can deploy over plain HTTP on the VM public IP for dry-run validation first
- only enable live public HTTPS after a real domain is attached

### AI runtime

If model is external:

- `LLM_ENABLED=true`
- `LLM_PROVIDER=openai_compatible`
- `LLM_BASE_URL=https://your-inference-endpoint/v1`
- `PRIMARY_MODEL=your-trained-model-name`
- `LLM_API_KEY=...`

If you keep AI disabled on Oracle during first deployment:

- `LLM_ENABLED=false`

If you want to run the trained model inside the Oracle backend container:

- `LLM_ENABLED=true`
- `LLM_PROVIDER=local_transformers`
- `PRIMARY_MODEL=merged_20260415_234248`
- `PRIMARY_MODEL_PATH=/app/research/artifacts/merged_models/merged_20260415_234248`

Important:

- use a Linux/container path on Oracle, not your Windows path
- the backend image already includes `research/artifacts/merged_models/...`
- if the VM has too little RAM or latency is poor, switch Oracle to an external OpenAI-compatible runtime instead

### WhatsApp / paging

- `AUTOPILOT_ALERTING_ENABLED=true`
- `AUTOPILOT_ALERT_WEBHOOK_URL=` optional generic webhook
- `AUTOPILOT_TWILIO_ACCOUNT_SID=...`
- `AUTOPILOT_TWILIO_AUTH_TOKEN=...`
- `AUTOPILOT_TWILIO_FROM_NUMBER=+14155238886` or your WhatsApp-enabled sender
- `AUTOPILOT_TWILIO_TO_NUMBER=+55...`
- `AUTOPILOT_TWILIO_CHANNEL=whatsapp`

Optional PagerDuty:

- `AUTOPILOT_PAGERDUTY_ROUTING_KEY=...`

### Unattended gate / burn-in

- `AUTOPILOT_UNATTENDED_MIN_BURNIN_RUNS`
- `AUTOPILOT_UNATTENDED_MAX_SKIP_RATE`
- `AUTOPILOT_UNATTENDED_MAX_RECONCILIATION_INCIDENTS`
- `AUTOPILOT_BURNIN_MAX_ERROR_RATE`
- `AUTOPILOT_BURNIN_MIN_FINALIZATION_SUCCESS_RATE`

## Deploy order

### Phase 1: infrastructure + dry-run

1. Create OCI VM
2. Open ingress:
   - `22` for SSH
   - `80` for HTTP
   - `443` for HTTPS
3. Bootstrap Docker:

```bash
chmod +x deploy/oracle/bootstrap_vm.sh
./deploy/oracle/bootstrap_vm.sh
```

4. Create `.env`
   - easiest path: copy `.env.oracle.example` to `.env` on the Oracle VM and fill it in
5. Deploy app:

```bash
chmod +x deploy/oracle/deploy_app.sh
./deploy/oracle/deploy_app.sh
```

6. Verify:

```bash
docker compose -f docker-compose.oracle.yml ps
curl http://127.0.0.1/health
curl http://127.0.0.1/api/health
```

### Phase 2: notification validation

Before live trading:

- configure Twilio WhatsApp
- verify outbound test alert
- verify health shows notification config status

### Phase 3: supervised live only

Run:

- dry-run first
- then supervised tiny-size live
- gather burn-in evidence

### Phase 4: unattended evaluation

Only after:

- burn-in report is clean
- notification paging is proven
- reconciliation works after forced restart
- AI runtime is stable enough for production

## Simple deployment commands

Full app:

```bash
cd /opt/ai-trader
./deploy/oracle/deploy_app.sh
```

Backend only:

```bash
cd /opt/ai-trader
./deploy/oracle/deploy_api_only.sh
```

Logs:

```bash
docker compose -f docker-compose.oracle.yml logs -f --tail=100
```

Restart:

```bash
docker compose -f docker-compose.oracle.yml restart
```

## Security recommendations

Use these as the baseline:

- do not expose backend port `8765` publicly
- expose only `80/443`
- use OCI Bastion or locked-down SSH source rules for admin access
- keep `.env` only on the VM
- rotate Binance/Twilio secrets after testing if they were ever shared insecurely
- move production secrets to OCI Vault later if you want stronger secret management
- use HTTPS with a real domain before supervised live internet exposure

## Health and recovery checks

Useful endpoints:

- `GET /health`
- `GET /api/health`
- `GET /api/autopilot/status`
- `GET /api/autopilot/reconcile`
- `GET /api/autopilot/readiness`

These now expose:

- autopilot state
- burn-in report
- notification status
- reconciliation state

## Final recommendation

Best Oracle deployment plan for this project:

1. Oracle VM for frontend + backend + autopilot + persistence + notifications
2. Caddy in front for reverse proxy and HTTPS
3. Twilio WhatsApp for warnings
4. keep the trained model off the Oracle free-tier VM unless you intentionally deploy a much smaller inference runtime
5. stay in dry-run first, then supervised live, then unattended only after burn-in says eligible

This is the most practical low-cost Oracle setup for the app today.
