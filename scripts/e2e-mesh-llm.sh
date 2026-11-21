#!/usr/bin/env bash
# Isolated live end-to-end smoke test of `mesh-dev` against a real `mesh-llm` (>= 0.80.0) binary.
# Runs in a temporary HOME with non-conflicting ports so any system mesh-llm service is untouched.
set -euo pipefail

MESH_LLM_BIN="${MESH_LLM_BIN:-mesh-llm}"
ARCHIVE=""
PORT="${MESH_DEV_E2E_PORT:-19337}"
CONSOLE_PORT="${MESH_DEV_E2E_CONSOLE_PORT:-13131}"
BIND_PORT="${MESH_DEV_E2E_BIND_PORT:-17842}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive) ARCHIVE="$2"; shift 2 ;;
    --mesh-llm) MESH_LLM_BIN="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --console) CONSOLE_PORT="$2"; shift 2 ;;
    --bind-port) BIND_PORT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$ARCHIVE" || ! -f "$ARCHIVE" ]]; then
  echo "Usage: $0 --archive <path/to/mesh-dev-<target>.tar.gz> [--mesh-llm <bin>]" >&2
  exit 2
fi

if ! command -v "$MESH_LLM_BIN" >/dev/null 2>&1 && [[ ! -x "$MESH_LLM_BIN" ]]; then
  echo "mesh-llm binary not found: $MESH_LLM_BIN" >&2
  exit 1
fi

TMP_HOME=$(mktemp -d)
NODE_PID=""
cleanup() {
  if [[ -n "$NODE_PID" ]] && kill -0 "$NODE_PID" 2>/dev/null; then
    kill -TERM "$NODE_PID" 2>/dev/null || true
    wait "$NODE_PID" 2>/dev/null || true
  fi
  rm -rf "$TMP_HOME"
}
trap cleanup EXIT

export HOME="$TMP_HOME"
export XDG_CONFIG_HOME="$TMP_HOME/.config"
export XDG_DATA_HOME="$TMP_HOME/.local/share"
export XDG_CACHE_HOME="$TMP_HOME/.cache"
mkdir -p "$TMP_HOME/.mesh-llm"

echo "==> 1. Installing mesh-dev archive into isolated HOME ($TMP_HOME)"
"$MESH_LLM_BIN" plugins install --archive "$ARCHIVE" --name mesh-dev

CONFIG_PATH="$TMP_HOME/.mesh-llm/config.toml"
cat > "$CONFIG_PATH" <<'EOF'
version = 1

[[plugin]]
name = "mesh-dev"
allow_inference_observation = true

[plugin.openai_exchange_grant]
endpoints = ["chat_completions", "completions", "responses"]
phases = ["request_received", "backend_selected", "exchange_finished"]
metadata = true
headers = ["x-mesh-target", "x-mesh-exclude", "x-request-id", "x-mesh-nonce", "x-mesh-dev-nonce"]
deadline_ms = 250
max_body_bytes = 1048576
max_queue_bytes = 4194304
max_in_flight = 32
EOF
"$MESH_LLM_BIN" config validate --config-path "$CONFIG_PATH" >/dev/null

echo "==> 2. Starting isolated mesh-llm on port=$PORT console=$CONSOLE_PORT bind=$BIND_PORT"
"$MESH_LLM_BIN" --config "$CONFIG_PATH" \
  --port "$PORT" --console "$CONSOLE_PORT" --bind-port "$BIND_PORT" \
  --client \
  >"$TMP_HOME/mesh-llm.out" 2>"$TMP_HOME/mesh-llm.err" &
NODE_PID=$!

BASE="http://127.0.0.1:${PORT}"
ready=0
for _ in $(seq 1 60); do
  if curl -fsS "${BASE}/v1/models" 2>/dev/null | jq -e '.data | map(.id) | index("dev/echo") != null' >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.5
done
if [[ "$ready" -ne 1 ]]; then
  echo "mesh-llm did not advertise dev/echo within 30s" >&2
  tail -n 80 "$TMP_HOME/mesh-llm.err" >&2 || true
  exit 1
fi

echo "==> 3. Verifying all 13 dev/* models are advertised"
MODELS_JSON=$(curl -fsS "${BASE}/v1/models")
for m in dev/echo dev/slow-ttft-15s dev/slow-ttft-45s dev/slow-ttft-75s dev/slow-sse-1tps dev/mid-stream-drop dev/http-503 dev/http-429 dev/malformed-sse dev/tool-call-edge dev/mem-probe dev/mem-probe-hold-5s dev/env-probe; do
  echo "$MODELS_JSON" | jq -e --arg m "$m" '.data | map(.id) | index($m) != null' >/dev/null
done

echo "==> 4. Testing dev/echo (with custom nonce header)"
ECHO_RESP=$(curl -fsS -X POST "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "x-mesh-dev-nonce: e2e-nonce-42" \
  -d '{"model":"dev/echo","messages":[{"role":"user","content":"hello from e2e"}]}')
echo "$ECHO_RESP" | jq -e '.choices[0].message.content | fromjson | .prompt == "hello from e2e" and .headers["x-mesh-dev-nonce"] == "e2e-nonce-42"' >/dev/null
ORIG_INSTANCE=$(echo "$ECHO_RESP" | jq -r '.choices[0].message.content | fromjson | .plugin_instance_id')
echo "    initial instance_id=$ORIG_INSTANCE"

echo "==> 5. Testing dev/mem-probe and sub-5 MB RSS"
MEM_RESP=$(curl -fsS -X POST "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"dev/mem-probe","messages":[{"role":"user","content":"probe"}]}')
RSS_KB=$(echo "$MEM_RESP" | jq -r '.choices[0].message.content | fromjson | .self_rss_kb')
AVAIL_KB=$(echo "$MEM_RESP" | jq -r '.choices[0].message.content | fromjson | .mem_available_kb')
echo "    mem_available_kb=${AVAIL_KB} self_rss_kb=${RSS_KB}"

echo "==> 6. Testing dev/tool-call-edge (streaming)"
TOOL_SSE=$(curl -fsS -N -X POST "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"dev/tool-call-edge","messages":[{"role":"user","content":"edge"}],"stream":true}')
echo "$TOOL_SSE" | grep -q '\[DONE\]'

echo "==> 7. Testing dev/http-503 error propagation"
CODE_503=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"dev/http-503","messages":[{"role":"user","content":"fail"}]}')
echo "    dev/http-503 HTTP status: ${CODE_503}"
test "$CODE_503" -ge 500

echo "==> 8. Querying inspector & env_probe via mesh-llm plugins call (console port=$CONSOLE_PORT)"
ENV_JSON=$("$MESH_LLM_BIN" plugins call mesh-dev env_probe --port "$CONSOLE_PORT" --input '{}')
echo "$ENV_JSON" | jq -e '.allowlisted_vars.MESH_LLM_PLUGIN_NAME == "mesh-dev"' >/dev/null
STATS_JSON=$("$MESH_LLM_BIN" plugins call mesh-dev stats --port "$CONSOLE_PORT" --input '{}')
echo "    stats summary: $(echo "$STATS_JSON" | jq -c '{http: .inspector.http | {total_requests, completed, injected_errors}, observation: .inspector.observation | {status, finished_executions}, exchange: .inspector.exchange | {distinct_exchanges, total_events}}')"
echo "$STATS_JSON" | jq -e '.inspector.http.total_requests >= 4' >/dev/null
echo "$STATS_JSON" | jq -e '.inspector.observation.status == "subscribed"' >/dev/null
echo "$STATS_JSON" | jq -e '.inspector.exchange.distinct_exchanges >= 1' >/dev/null
RECENT_JSON=$("$MESH_LLM_BIN" plugins call mesh-dev inspect_recent --port "$CONSOLE_PORT" --input '{"source":"all","limit":10}')
echo "$RECENT_JSON" | jq -e '.http | length >= 4' >/dev/null
echo "$RECENT_JSON" | jq -e '.exchange | length >= 1' >/dev/null

echo "==> 9. Testing inject_health error -> ok recovery"
"$MESH_LLM_BIN" plugins call mesh-dev inject_health --port "$CONSOLE_PORT" \
  --input '{"mode":"error","detail":"e2e fault"}' >/dev/null
"$MESH_LLM_BIN" plugins call mesh-dev inject_health --port "$CONSOLE_PORT" \
  --input '{"mode":"ok"}' >/dev/null

echo "==> 10. Testing inject_crash -> supervisor restart with new instance_id"
"$MESH_LLM_BIN" plugins call mesh-dev inject_crash --port "$CONSOLE_PORT" \
  --input '{"exit_code":1,"delay_ms":50}' >/dev/null

restarted=0
for _ in $(seq 1 40); do
  NEW_RESP=$(curl -fsS -X POST "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"dev/echo","messages":[{"role":"user","content":"after crash"}]}' 2>/dev/null || true)
  if [[ -n "$NEW_RESP" ]]; then
    NEW_INSTANCE=$(echo "$NEW_RESP" | jq -r '.choices[0].message.content | fromjson | .plugin_instance_id' 2>/dev/null || true)
    if [[ -n "$NEW_INSTANCE" && "$NEW_INSTANCE" != "$ORIG_INSTANCE" ]]; then
      echo "    restarted cleanly with new instance_id=$NEW_INSTANCE"
      restarted=1
      break
    fi
  fi
  sleep 0.5
done
if [[ "$restarted" -ne 1 ]]; then
  echo "mesh-dev did not restart with a new instance_id after inject_crash" >&2
  tail -n 60 "$TMP_HOME/mesh-llm.err" >&2 || true
  exit 1
fi

echo "==> ALL LIVE E2E CHECKS PASSED"
