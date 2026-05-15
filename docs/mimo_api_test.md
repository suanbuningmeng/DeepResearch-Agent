# Xiaomi MiMo API Connectivity Test

Xiaomi MiMo can be used as an OpenAI-compatible backend for DeepResearch-Agent. Start with the minimal `chat/completions` test before running the full agent pipeline.

Do not write API keys into source code, README files, screenshots, trace files, logs, or chat messages. Use environment variables instead.

## 1. Set Environment Variables

PowerShell:

```powershell
$env:XIAOMI_MIMO_API_KEY="your_key"
$env:XIAOMI_MIMO_API_BASE="https://api.xiaomimimo.com/v1"
$env:XIAOMI_MIMO_MODEL="xiaomi/mimo-v2-flash"
```

`XIAOMI_MIMO_API_BASE` defaults to `https://api.xiaomimimo.com/v1`.

`XIAOMI_MIMO_MODEL` defaults to `xiaomi/mimo-v2-flash`. This default is only an example. Use the model name shown in the Xiaomi console. Possible names may include:

- `xiaomi/mimo-v2-flash`
- `xiaomi/mimo-v2-pro`
- `xiaomi/mimo-v2-omni`
- another model name shown in your console

## 2. Run the Minimal Connectivity Test

```powershell
python scripts/test_mimo_api.py
```

The script sends one OpenAI-compatible request to:

```text
{XIAOMI_MIMO_API_BASE}/chat/completions
```

It prints:

- `status_code`
- `model`
- `api_base`
- a short assistant response or sanitized error

It never prints the API key.

## 3. Run the Minimal DeepResearch-Agent Demo

After the minimal API test succeeds, run a small `openai-compatible` demo:

```powershell
python scripts/run_demo.py `
  --question "What are the main challenges of long-context LLM evaluation?" `
  --backend openai-compatible `
  --mode dag `
  --max-concurrency 1 `
  --api-base $env:XIAOMI_MIMO_API_BASE `
  --api-key-env XIAOMI_MIMO_API_KEY `
  --model $env:XIAOMI_MIMO_MODEL `
  --max-tokens 768 `
  --request-timeout 180 `
  --global-timeout-seconds 300
```

## 4. Run the Full Pipeline Demo

After the minimal demo works, try the full local pipeline:

```powershell
python scripts/run_demo.py `
  --question "What are the main challenges of long-context LLM evaluation?" `
  --backend openai-compatible `
  --mode dag `
  --max-concurrency 1 `
  --api-base $env:XIAOMI_MIMO_API_BASE `
  --api-key-env XIAOMI_MIMO_API_KEY `
  --model $env:XIAOMI_MIMO_MODEL `
  --max-tokens 768 `
  --request-timeout 180 `
  --global-timeout-seconds 300 `
  --memory-backend sqlite `
  --clear-memory `
  --enable-context-compression `
  --enable-conflict-detection `
  --enable-red-blue `
  --enable-web-search `
  --search-provider mock
```

The full pipeline uses `--search-provider mock` first so you can test the MiMo LLM backend without introducing a separate real search API, search quota, or network-fetch failure.

## 5. Troubleshooting

`XIAOMI_MIMO_API_KEY is not set.`

- Set `$env:XIAOMI_MIMO_API_KEY`.
- Confirm the terminal running Python is the same terminal where the variable was set.

`401` or `403`

- The API key may be wrong, expired, or missing permission.
- Check `XIAOMI_MIMO_API_KEY`.

`404 model not found`

- The model name may not match your Xiaomi console.
- Check `XIAOMI_MIMO_MODEL`.

`404 endpoint not found`

- The API base may be wrong.
- Check `XIAOMI_MIMO_API_BASE`.
- The final request URL should be `{XIAOMI_MIMO_API_BASE}/chat/completions`.

`429`

- Rate limit or quota issue.
- Wait and retry, or check account quota.

`timeout`

- Usually a network or proxy issue.
- Try setting or clearing:

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
$env:ALL_PROXY="http://127.0.0.1:7890"
```

Use the actual HTTP proxy port shown by your proxy client.

`400`

- Payload parameters may be incompatible.
- Try a simpler request with only `model`, `messages`, and `max_tokens`.

JSON parse failure in the agent pipeline

- Some real models return extra prose around JSON.
- Some OpenAI-compatible models, including MiMo, may return natural-language judge comments instead of strict JudgeAgent JSON.
- MiMo and other OpenAI-compatible models may return non-JSON text, field-name variants, JSON fragments, or truncated JSON.
- DeepResearch-Agent now applies a robust structured output strategy for PlannerAgent, ResearcherAgent, and JudgeAgent:
  - direct JSON parse after removing thinking blocks and fences;
  - lenient JSON parse for trailing commas, Python literals, and text around JSON;
  - schema coercion and partial JSON salvage for field-name variants or truncated arrays;
  - LLM-based JSON repair retry with a strict schema prompt;
  - local fallback if repair still fails.
- JudgeAgent additionally tries to extract rubric scores from natural-language text before using repair/fallback.
- In `demo_trace.json`, check `planner_stats.repair_attempted` and `planner_stats.repair_success`.
- In `demo_trace.json`, check `planner_stats.schema_coercion_success` and `planner_stats.partial_repair_used`.
- In `demo_trace.json`, check `judge_stats.repair_attempted` and `judge_stats.repair_success`.
- In `demo_trace.json`, check `judge_stats.schema_coercion_success`.
- Researcher repair details are stored in each evidence item's `metadata`, including `json_parse_success`, `repair_attempted`, `repair_success`, and `fallback_parse`.
- Researcher salvage details are stored as `schema_coercion_success` and `partial_json_salvaged`.
- If `schema_coercion_success=true`, the system recovered structured data from non-standard JSON instead of falling directly back.
- If `demo_trace.json` has `judge_stats.extracted_from_text=true`, the score was parsed from natural-language rubric text.
- If `demo_trace.json` has `judge_stats.fallback_used=true`, the rule-based fallback judge score was used.
- Raw failed structured outputs are written locally for debugging:
  - `outputs/debug_last_planner_output.txt`
  - `outputs/debug_last_researcher_output_<task_id>.txt`
  - `outputs/debug_last_judge_output.txt`
- Do not commit `outputs/`.
- For first tests keep `--max-concurrency 1` and moderate `--max-tokens`.
