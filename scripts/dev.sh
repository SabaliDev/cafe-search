#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
  set +a
fi

pip install -r "$ROOT_DIR/requirements.txt" -r "$ROOT_DIR/requirements-dev.txt"

docker-compose up -d db redis

export DATABASE_URL="${DATABASE_URL:-postgresql+asyncpg://postgres:postgres@localhost:5432/cafesearch}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

alembic upgrade head

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
