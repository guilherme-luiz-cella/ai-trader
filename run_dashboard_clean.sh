#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Prevent inherited shell/network proxy variables from hijacking API calls.
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY
unset http_proxy
unset https_proxy
unset all_proxy

# Preserve localhost exceptions only.
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

exec ./.venv/bin/python -m streamlit run research/streamlit_dashboard.py --server.headless true --server.port "${PORT:-8501}"
