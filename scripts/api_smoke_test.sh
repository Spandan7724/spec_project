#!/usr/bin/env bash
set -euo pipefail

# Simple API smoke test for the FastAPI backend.
# Usage:
#   BASE=http://localhost:8000 bash scripts/api_smoke_test.sh
# or just run without BASE to use the default.

BASE=${BASE:-http://localhost:8000}

JQ=$(command -v jq || true)
pp() {
  if [[ -n "${JQ}" ]]; then
    echo "$1" | ${JQ}
  else
    echo "$1"
  fi
}

gen_uuid() {
  if command -v uuidgen >/dev/null 2>&1; then
    uuidgen
  else
    python - <<'PY'
import uuid; print(uuid.uuid4())
PY
  fi
}

SID=${SID:-$(gen_uuid)}
CORR1=${CORR1:-$(gen_uuid)}
CORR2=${CORR2:-$(gen_uuid)}
CORR3=${CORR3:-$(gen_uuid)}

echo "== Using BASE=${BASE}"
echo "== Session ID: ${SID}"
echo "== Correlation IDs: ${CORR1}, ${CORR2}, ${CORR3}"
echo

echo "== Conversation: first turn"
resp1=$(curl -sS -X POST "$BASE/api/conversation/message" \
  -H 'Content-Type: application/json' \
  -d "{\"user_input\":\"I need to convert 1000 USD to EUR\",\"session_id\":\"$SID\"}")
pp "$resp1"
if [[ -n "$JQ" ]]; then
  sid_new=$(echo "$resp1" | jq -r '.session_id // empty' || true)
  [[ -n "$sid_new" ]] && SID="$sid_new"
fi
echo

echo "== Conversation: follow-up turn (risk + urgency)"
resp2=$(curl -sS -X POST "$BASE/api/conversation/message" \
  -H 'Content-Type: application/json' \
  -d "{\"user_input\":\"Risk moderate and urgency normal\",\"session_id\":\"$SID\"}")
pp "$resp2"
echo

echo "== Conversation: get session"
pp "$(curl -sS "$BASE/api/conversation/session/$SID")"
echo

echo "== Analysis: start (legacy categorical timeframe)"
pp "$(curl -sS -X POST "$BASE/api/analysis/start" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"correlation_id\":\"$CORR1\",\"currency_pair\":\"USD/EUR\",\"base_currency\":\"USD\",\"quote_currency\":\"EUR\",\"amount\":5000,\"risk_tolerance\":\"moderate\",\"urgency\":\"normal\",\"timeframe\":\"1_week\"}")"
echo

echo "== Analysis: start (free-text timeframe)"
pp "$(curl -sS -X POST "$BASE/api/analysis/start" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"correlation_id\":\"$CORR2\",\"currency_pair\":\"USD/EUR\",\"base_currency\":\"USD\",\"quote_currency\":\"EUR\",\"amount\":5000,\"risk_tolerance\":\"moderate\",\"urgency\":\"normal\",\"timeframe_text\":\"in 10 days\"}")"
echo

echo "== Analysis: start (canonical timeframe)"
pp "$(curl -sS -X POST "$BASE/api/analysis/start" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"correlation_id\":\"$CORR3\",\"currency_pair\":\"USD/EUR\",\"base_currency\":\"USD\",\"quote_currency\":\"EUR\",\"amount\":5000,\"risk_tolerance\":\"moderate\",\"urgency\":\"normal\",\"timeframe_days\":14,\"timeframe_mode\":\"duration\"}")"
echo

poll_status() {
  local id="$1"; local attempts=30; local i=0
  while (( i < attempts )); do
    st=$(curl -sS "$BASE/api/analysis/status/$id")
    if [[ -n "$JQ" ]]; then echo "$st" | ${JQ}; else echo "$st"; fi
    status=$(echo "$st" | sed -n 's/.*"status"\s*:\s*"\([^"]*\)".*/\1/p')
    if [[ "$status" == "completed" || "$status" == "error" ]]; then
      break
    fi
    sleep 1
    i=$((i+1))
  done
}

echo "== Poll status (categorical)"
poll_status "$CORR1"
echo
echo "== Poll status (free-text)"
poll_status "$CORR2"
echo
echo "== Poll status (canonical)"
poll_status "$CORR3"
echo

echo "== Get results"
echo "-- Result $CORR1"
pp "$(curl -sS "$BASE/api/analysis/result/$CORR1")"
echo "-- Result $CORR2"
pp "$(curl -sS "$BASE/api/analysis/result/$CORR2")"
echo "-- Result $CORR3"
pp "$(curl -sS "$BASE/api/analysis/result/$CORR3")"
echo

echo "== Visualization: confidence (for $CORR1)"
pp "$(curl -sS "$BASE/api/viz/confidence/$CORR1")"
echo

echo "== Health"
pp "$(curl -sS "$BASE/health")"
echo

echo "== Root"
pp "$(curl -sS "$BASE/")"
echo

echo "== Reset session"
pp "$(curl -sS -X POST "$BASE/api/conversation/reset/$SID")"
echo

echo "== Done"

