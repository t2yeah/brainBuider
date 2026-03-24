#!/usr/bin/env bash

echo "kill old api..."
fuser -k 8010/tcp 2>/dev/null

echo "start api..."
uvicorn app.main:app --host 0.0.0.0 --port 8010
