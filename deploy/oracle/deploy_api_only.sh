#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_APP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

APP_DIR=${APP_DIR:-$DEFAULT_APP_DIR}
REPO_URL=${REPO_URL:-https://github.com/guilherme-luiz-cella/ai-trader.git}
BRANCH=${BRANCH:-main}
SKIP_GIT_SYNC=${SKIP_GIT_SYNC:-false}

if [[ ! -d "$APP_DIR/.git" ]]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"

if [[ ! -f docker-compose.oracle.yml ]]; then
  echo "ERROR: docker-compose.oracle.yml not found in $APP_DIR"
  echo "If you want to deploy a separate clone, run with APP_DIR=/path/to/clone"
  exit 1
fi

if [[ "$SKIP_GIT_SYNC" != "true" ]]; then
  git fetch origin "$BRANCH" || true
  git checkout "$BRANCH" || true
  if ! git pull --ff-only origin "$BRANCH"; then
    echo "WARNING: git fast-forward pull failed. Continuing with local working tree."
    echo "Set SKIP_GIT_SYNC=true to suppress git sync attempts."
  fi
else
  echo "Skipping git sync because SKIP_GIT_SYNC=true"
fi

if [[ ! -f .env ]]; then
  cp .env.example .env 2>/dev/null || true
  echo "WARNING: .env missing. Create $APP_DIR/.env before starting services."
fi

docker compose -f docker-compose.oracle.yml up -d --build signal-api

docker compose -f docker-compose.oracle.yml ps signal-api
