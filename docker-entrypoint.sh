#!/bin/sh
set -e

if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

python -m alembic -c alembic.ini upgrade head

exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}
