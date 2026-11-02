# Engine API contract

The formats the engine writes and serves for the web site (`web/`) and other clients: the trace file `engine bench --trace` writes, and the HTTP endpoints of `engine serve`. The site's replay mode reads a trace file and its live mode reads the endpoints, so a step looks the same in both.

## Conventions

- JSON, UTF-8. Times are seconds as numbers (`_s` suffix), counted from the start of the timed run for a trace and from server start for the endpoints.
- Every top-level object carries `"version"`. Adding a field is not a breaking change and does not bump it; clients ignore fields they do not know. Removing or changing the meaning of a field bumps it.
- Sequence ids (`seqs`, `id`) are request ids; a request keeps its id across preemption.

## Step object

One scheduler step: one forward pass over the batch and one sampled token per sequence in it.

```json
{
  "index": 12,
  "start_s": 0.4105,
  "end_s": 0.4198,
  "kind": "decode",
  "seqs": [0, 1, 3],
  "tokens": [
    {"id": 450, "text": " the"},
    {"id": 29892, "text": ","},
    {"id": 2, "text": ""}
  ],
  "finished": [3],
  "waiting": 2,
  "running": 2,
  "free_blocks": 1211
}
```

| Field | Meaning |
| --- | --- |
| `index` | Step number, from 0, in order |
| `start_s`, `end_s` | When the step started (before scheduling) and ended (tokens copied back to the host) |
| `kind` | `"prefill"` or `"decode"`; a step is all one or the other |
| `seqs` | The sequences in the batch, in batch row order |
| `tokens` | One entry per entry of `seqs`, same order: the sampled token id and the text it adds to its sequence. `text` is the incremental UTF-8 text, so it starts with a space where SentencePiece puts one (except the first token of a sequence, whose leading space decoding drops), and it is `""` when the token completes no character yet or is EOS |
| `finished` | Sequences that finished in this step (their last token is in `tokens`) |
| `waiting`, `running` | Queue sizes after the step |
| `free_blocks` | Free KV cache blocks after the step |

A prefill step samples the first output token of each sequence in it. A sequence that appears in a prefill step after it already had decode steps was preempted and is being recomputed; its token continues its output.

## Trace file

Written by `engine bench --trace out.json` (batched mode only, the timed run only, no warmup).

```json
{
  "format": "llmie-trace",
  "version": 1,
  "meta": {
    "engine_version": "0.1.0",
    "date": "2026-10-25T10:00:00Z",
    "gpu": "NVIDIA GeForce RTX 5060 Laptop GPU",
    "build": "Release",
    "model": "tinyllama",
    "max_batch": 16,
    "graphs": true,
    "kv_blocks": 12915,
    "block_size": 16,
    "num_requests": 24,
    "prompt_len": "32:128",
    "output_len": "32:160",
    "request_rate": 4,
    "temperature": 0,
    "top_k": 0,
    "top_p": 1,
    "seed": 0
  },
  "requests": [
    {"id": 0, "arrival_s": 0.0, "prompt_tokens": 77, "output_tokens": 131, "preemptions": 0,
     "first_token_s": 0.021, "finished_s": 1.402}
  ],
  "steps": [ ]
}
```

`requests` lists every timed request in id order; `steps` lists every step object in order. A request's `arrival_s` is when it was submitted, `first_token_s` when its first token reached the host and `finished_s` when its last one did; they match the per-request CSV of the same run.


`results/traces/chat_b16_rate4.json` is the recorded trace the web site replays (RTX 5060 Laptop GPU, release build, 751 steps over 6.4 s, up to 8 sequences decoding together, answers ending at EOS after 9 to 191 tokens), made with the 20 chat prompts of `scripts/prompts.json`:

```
build/release/bin/engine bench --model models/tinyllama --prompts scripts/prompts.json --max-batch 16 --graphs --num-requests 24 --output-len 64:192 --stop-at-eos --request-rate 4 --temperature 0.7 --top-p 0.9 --seed 0 --warmup 1 --out /tmp/chat_run.csv --trace results/traces/chat_b16_rate4.json
```

`meta.prompts` names the prompts file (`"random"` for the random-token workload of `bench`).

## HTTP endpoints (`engine serve`)

```
build/release/bin/engine serve --model models/tinyllama --graphs
```

Listens on `127.0.0.1:8000` once the model is loaded (`--host`, `--port`); `--max-batch` (default 16) and `--graphs` are those of `bench`, and `--temperature`, `--top-k`, `--top-p` and `--max-tokens` set what a request gets when it does not say. One engine thread runs the scheduler; HTTP worker threads hand it requests and read tokens through queues, so requests join the batch between steps.

### `GET /health`

`200` with `{"version": 1, "status": "ok"}`. The server only listens once the model is loaded, so anything else means it is not up.

### `GET /metrics`

A snapshot; the dashboard polls it about once a second.

```json
{
  "version": 1,
  "time_s": 512.3,
  "meta": {"engine_version": "0.1.0", "gpu": "...", "model": "tinyllama", "max_batch": 16, "graphs": true},
  "requests": {"waiting": 2, "running": 9, "finished_total": 341, "preemptions_total": 3},
  "kv": {"blocks_total": 12915, "blocks_free": 11020, "block_size": 16},
  "tokens": {"output_total": 48211, "output_per_s": 812.4},
  "latency": {
    "ttft_s": {"p50": 0.031, "p99": 0.210},
    "tpot_s": {"p50": 0.0112, "p99": 0.0164}
  },
  "steps_total": 6120
}
```

`output_per_s` covers the last 10 seconds (or the time since start when that is shorter); the latency percentiles cover the requests that finished in the last 60 seconds (`null` when there are none, and `tpot_s` also ignores one-token answers). TTFT counts from when the HTTP request was received. `finished_total` and the latency cover answers that reached their end (EOS, a stop string or `max_tokens`), not requests whose client went away. `output_total` counts every token sampled, including those of a cancelled request still running. `preemptions_total` counts prefill steps of sequences that already had output. `blocks_total` includes the null block 0, so an idle server has `blocks_free` = `blocks_total` - 1.

### `GET /v1/steps` (Server-Sent Events)

One `event: step` per scheduler step, `data:` holding a step object exactly as above (with `start_s`, `end_s` from server start). The dashboard's batch timeline reads this stream, the same way replay mode reads `steps` of a trace. A client that falls behind may miss steps; `index` shows the gap. Each subscriber has a queue of 256 steps that drops its oldest entry, so a slow client never holds up the engine; one that stops reading long enough for a socket write to time out (5 s) is disconnected. While no step arrives the server sends a `: keepalive` comment every second, which SSE clients ignore. Every running sequence is in every decode step; a sequence that disappears from the decode steps without being in `finished` was preempted (or, once `Scheduler::cancel` exists, cancelled).

### `POST /v1/chat/completions`

OpenAI-compatible. Request fields used: `messages` (roles `system`, `user`, `assistant`, string `content`), `max_tokens`, `temperature`, `top_p`, `seed`, `stream`, `stop` (string or list), and the extension `top_k`. `model` is accepted and ignored, as are fields not listed here, and a field set to `null` counts as missing. Without `max_tokens` the answer stops at `--max-tokens` (256) or the end of the context; a `max_tokens` that does not fit the context is refused. Without `seed` the seed is random. The prompt is built with the TinyLlama chat template with no BOS, as HF's `apply_chat_template` and `engine generate` do. A malformed request gets `400` with `{"error": {"message": "...", "type": "invalid_request_error"}}`.

With `"stream": true` the response is `text/event-stream`: a first chunk with `{"role": "assistant"}` (its `engine` field has only `seq_id`, since no token exists yet), one chunk per token with its `text` as `content` (no chunk for a token that adds no text), a last chunk with an empty delta and `finish_reason` `"stop"` (EOS or a stop string) or `"length"`, then `data: [DONE]`.

```
data: {"id":"chatcmpl-341","object":"chat.completion.chunk","created":1793000000,"model":"tinyllama","choices":[{"index":0,"delta":{"content":" the"},"finish_reason":null}],"engine":{"seq_id":341,"step":6118}}
```

The `engine` field is an extension: the sequence id and step index of the token, so the playground can highlight its own sequence in the timeline. Without `stream` the response is one `chat.completion` object with `usage` (prompt, completion and total tokens). Text matching a stop string is not sent, and text that could still become one is held back until it is decided. Greedy output comes from the paged runner, so it equals `engine generate` except where two logits are nearly tied (see `tests/gpu/test_serve.cpp`).

A client that disconnects cancels its request: no more of its tokens go anywhere. Freeing its blocks at once needs `Scheduler::cancel`, which is a Part C stub for now (`src/sched/scheduler.cpp`), so today the sequence keeps decoding with its tokens discarded and frees its blocks when it reaches `max_tokens` or EOS; the same goes for a sequence ended by a stop string. The server prints one warning about it.

### CORS

The server answers `OPTIONS` preflights and sends `Access-Control-Allow-Origin` on every endpoint for the origins given with `--cors-origin`, repeatable (default: `http://localhost:3000` only; the site's Vercel domain is not set yet, so pass it with `--cors-origin https://<domain>`, which replaces the default). A preflight that carries `Access-Control-Request-Private-Network: true` is answered with `Access-Control-Allow-Private-Network: true` so a page served over HTTPS can reach a server on localhost where the browser allows it. A request whose `Origin` is not on the list gets `403` and starts no work, which keeps other web sites from spending the GPU with a simple `text/plain` POST the browser would not preflight; requests with no `Origin` (curl, scripts) are served. Origins must be a bare scheme and host, and `*` is refused. The server has no authentication: do not expose it publicly or tunnel it.
