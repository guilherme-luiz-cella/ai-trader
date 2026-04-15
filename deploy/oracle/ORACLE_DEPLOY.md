# Oracle Always Free Deployment

This deploys two containers on an Oracle VM:

- React frontend on port `3000`
- Backend API on port `8765`

If you only want the backend API in the cloud, you can deploy just `signal-api`.

## 1) VM prerequisites

Use Ubuntu (22.04/24.04) on Oracle Always Free.

Open Oracle network ingress rules for:

- `22` (SSH)
- `3000` (frontend)
- `8765` (backend API, optional if only internal)

## 2) Bootstrap Docker

```bash
chmod +x deploy/oracle/bootstrap_vm.sh
./deploy/oracle/bootstrap_vm.sh
# then re-login SSH or run: newgrp docker
```

## 3) Configure environment

Create and edit `.env` in repo root with your keys.

The frontend is already wired to talk to the API through Docker Compose.

## 4) Deploy services

```bash
chmod +x deploy/oracle/deploy_app.sh
./deploy/oracle/deploy_app.sh
```

Backend-only deploy:

```bash
chmod +x deploy/oracle/deploy_api_only.sh
./deploy/oracle/deploy_api_only.sh
```

## 5) Verify

```bash
docker compose -f docker-compose.oracle.yml logs -f --tail=100
curl http://localhost:8765/health
```

Frontend URL:

- `http://<your-vm-public-ip>:3000`

Backend API URL:

- `http://<your-vm-public-ip>:8765/health`
- `http://<your-vm-public-ip>:8765/signal`

## 6) Update app after pushes

```bash
./deploy/oracle/deploy_app.sh
```

For backend-only updates:

```bash
./deploy/oracle/deploy_api_only.sh
```

## Notes

- Containers are set to `restart: unless-stopped`.
- Keep API key secrets in `.env` on the VM only (do not commit).
- If you do not want the API exposed publicly, do not open port `8765` in Oracle ingress rules.
