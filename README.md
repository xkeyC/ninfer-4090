# NInfer-4090

NInfer-4090 runs **Qwen3.8-27B** on one 24 GB NVIDIA GeForce RTX 4090. It is an `sm_89` port of
[NInfer-3090](https://github.com/Don-Chad/ninfer-3090), which derives from
[Neroued/ninfer](https://github.com/Neroued/ninfer), a specialized C++20/CUDA inference engine.
The engine loads the official groupwise `.ninfer` artifact, serves OpenAI- and
Anthropic-compatible APIs, and supports paged KV, compatible-prefix reuse, CUDA Graphs, MTP
speculative decoding, reasoning-effort control, and ReplaySSM state transactions.

This fork targets `sm_89` and Linux. Blackwell-only NVFP4/W4A4 execution is unavailable; the
engine uses the same groupwise-int path as the 3090 base. The Windows path and the
Qwen3.6-35B-A3B target are inherited but untested on the RTX 4090.

## Long-context multi-session cache

This fork adds a process-local host block cache for long agent conversations that outlive GPU KV
residency. `--host-prefix-cache-mib N` partitions a retained snapshot into metadata, cumulative GDN
state/checkpoints, and 64-token KV page groups. Byte-identical immutable blocks are stored once
across conversation branches, while logical manifests remain available even when one copy is
restored into a GPU lane. Restore streams block spans directly to pinned staging instead of first
copying the full snapshot; a later spill transfers and hashes only new or changed device blocks.

Each block records its recall count and last-recall time. When the byte budget is full, eviction
chooses the coldest unpinned block using recency plus a logarithmic frequency bonus and removes all
dependent manifests atomically. This avoids the old all-or-nothing behavior where a single large
session could exceed the host budget or destroy every reusable prefix on eviction.

KV-affinity admission complements the block store. With `--kv-affinity-burst 5` (the default), a
queued request matching the current GPU KV owner may pass cold work for at most five contended
admissions; the scheduler then rotates to the oldest competing conversation. The bound advances
only under real contention, and `--kv-affinity-grace-ms 1500` gives the current conversation a
short window to submit its next turn before paying a multi-GiB owner switch.

An RTX 4090 capacity test with three 90K-token branches produced three logical manifests backed by
2,909 unique blocks using 2.67 GB of host RAM. A resident continuation completed in 1.68 seconds;
an evicted 90K branch restored and answered in 10.27 seconds, with no capture or restore failures.

### Four-Agent gradual 200K rotation

The long-run cache test on 2026-08-23 used Qwen3.8-27B-Uncensored with `rk4v4-e8`, MTP3, a
253,952-token shared KV pool, two lanes, a 20 GiB host cache, affinity burst 5, and 1500 ms grace.
Four independent Agent histories ran sequential round-robin. Each started at 1,389 tokens, returned
the complete assistant content, reasoning, and tool calls, then added about 1,065–1,097 tokens of
tool/workspace state per turn. An Agent left the rotation only after its measured prompt reached
200K. The run completed 627 requests over 166 rounds in 63.70 minutes. A separate server restart
then measured a true empty-cache 200,037-token request as the cold baseline.

| 200K request | Prompt | Cached | Hit rate | Queue | Host restore | TTFT | TTFT vs cold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Empty-cache baseline | 200,037 | 0 | 0.00% | 0.000 s | 0 s | 149.951 s | 1.00× |
| Agent 0, resident | 200,900 | 199,803 | 99.45% | 0.002 s | 0 s | 1.800 s | 83.33× |
| Agent 1, restored | 201,127 | 194,074 | 96.49% | 1.064 s | 0.410 s | 10.627 s | 14.11× |
| Agent 2, restored | 200,128 | 196,479 | 98.18% | 1.043 s | 0.434 s | 6.890 s | 21.76× |
| Agent 3, restored | 200,766 | 199,669 | 99.45% | 1.047 s | 0.458 s | 4.363 s | 34.37× |

The median final TTFT was 4.363 seconds, 34.37× faster than the measured cold request. The full
continuation population separates scheduling delay from data movement as follows; restore
percentiles include only the 534 requests that actually transferred a Host entry, while resident
requests report zero restore time.

| Long-run metric | Result |
|---|---:|
| Continuations / weighted cache hit | 623 / 98.028% |
| Recompute ≤1.5K | 500 / 623 (80.26%) |
| Recompute 1.5K–10K | 118 / 623 (18.94%) |
| Recompute 10K–50K | 4 / 623 (0.64%) |
| Recompute >50K | 1 / 623 (0.16%) |
| Continuations with zero cached tokens | 0 |
| TTFT P50 / P95 | 3.884 s / 8.985 s |
| Queue P50 / P95 | 1.332 s / 1.483 s |
| Host restore P50 / P95 | 0.308 s / 0.479 s |
| Host restore payload / effective bandwidth | 1.188 TB / 7.592 GB/s |
| Captures / cumulative capture wall | 542 / 513.504 s |
| Block/manifest evictions | 516 |
| Cache drops / capture failures / restore failures | 0 / 0 / 0 |

The old approximately 122K per-lane wall was crossed without a reset: 128,644- and 130,095-token
requests reused 127,579 and 129,030 tokens. Once the 20 GiB budget filled, ordinary pressure caused
temporary rollback by a few 1K turns and then rebuilt the deep frontier. One request at 137,352
tokens retained only 8,439 tokens and took 89.881 seconds; this was the sole >50K rollback and the
same Agent returned to a recent deep frontier on following turns. Queue P50 was over four times
restore P50 in this deliberately strict alternation, and cumulative capture wall was the larger
remaining cache data-path cost.

See [Serving](docs/serving.md) for flags and metrics and
[Concurrent inference architecture](docs/maintainer/concurrent-inference-architecture.md) for the
ownership and fairness contracts.

## Measured results on the RTX 4090

Conditions: single request, greedy decoding, CUDA Graphs on, INT8 KV, `--prefill-chunk 1024`,
official 16.96 GiB Qwen3.8-27B artifact. The code-generation decode row and the prefill rows
are measured from the `ninfer-serve` `/metrics` counters (computed prefill only); the other
decode rows use the `ninfer` CLI.

| Test | Result |
|---|---|
| Decode, code generation, MTP3 | **148.6 tok/s** at 81.0% draft acceptance |
| Decode, bench corpus, MTP3 | 106.5 tok/s at 48.7% acceptance |
| Decode, no speculation | 50.5 tok/s |
| Decode at 128K depth, no speculation | 39.6 tok/s |
| 64K needle-in-a-haystack | exact answer, 1,849 tok/s prefill |
| 128K needle-in-a-haystack | exact answer, 1,561 tok/s prefill |
| Vision, chart reading | 3 of 3 oracle facts, 22 ms vision tower |
| Ops test suite | 78 of 78 runnable tests pass on `sm_89` |

MTP acceptance, and with it the decoded rate, tracks how predictable the output is: structured
code accepts about 81% of draft tokens, the mixed bench corpus about 49%.

The shipping default has since moved from INT8 KV to the E8 4-bit KV mode, which serves the
model's full native 262,144-token context on this card. Retrieval stays exact through 260K
(single-needle, 5-needle, and exact-code-detail probes), MTP acceptance at depth is unchanged,
and the costs against the INT8 numbers above are a 5.7% decode tax and 1-2% of prefill; see
[Quick start](#text-only-full-262k-native-context-e8-4-bit-kv-default) for the measured deltas.

For scale: llama.cpp on the same card decodes the Qwen3.8-27B `UD-Q4_K_XL` GGUF at about
46 tok/s in a 144K-context configuration where the MTP buffers do not fit. The upstream engine
on an RTX 5090 measures 172 tok/s on the same code-generation prompts with a 400 W power cap
(the upstream README quotes about 200), so this card lands within 14% of it under MTP.

### Depth sweep against llama.cpp

Both engines were measured on the same card. llama.cpp build 10358 ran `llama bench` on the
`UD-Q4_K_XL` GGUF (16.68 GiB) with q8_0 KV cache, flash attention, and `-ub 1024 -b 4096`,
which matches its deployed configuration, on 2026-08-15. The NInfer side was re-measured on
2026-08-17 on the deployed E8 262K configuration through the `/metrics` counters; the
llama.cpp configuration did not change between the dates. Two caveats: the artifacts differ
by about 2% in size, and `llama bench` is a bare kernel loop while the NInfer numbers
include the full server path.

Marginal rates at depth:

| Depth | llama.cpp pp2048 | llama.cpp tg32 | NInfer decode, no speculation |
|---:|---:|---:|---:|
| 0 | 3,024 tok/s | 45.9 tok/s | 50.4 tok/s |
| 32K | 2,327 | 42.0 | - |
| 64K | 1,866 | 38.6 | - |
| 128K | 1,336 | 33.1 | 42.1 |
| 256K | no entry | no entry | 36.6 |

Wall time to prefill one full prompt (llama.cpp integrated from the marginal rates, NInfer
measured):

| Prompt | llama.cpp | NInfer |
|---:|---:|---:|
| 32K | 12.5 s (2,630 tok/s) | 14.5 s (2,027 tok/s) |
| 64K | 28.3 s (2,317 tok/s) | 31.7 s (1,857 tok/s) |
| 128K | 70.4 s (1,862 tok/s) | 74.5 s (1,581 tok/s) |
| 192K | no entry | 127.9 s (1,381 tok/s) |
| 256K | no entry | 191.7 s (1,228 tok/s) |

The llama.cpp prefill lead narrows with depth. Server-measured, it prefills a 64K prompt in
28.7 s against 31.7 s (a 10% lead) and a 128K prompt in 71.6 s against 74.5 s (4%); the
server path costs llama.cpp 2-4% over the bare-loop estimates above. Everything past its
144K ceiling is NInfer-only. Decode inverts the shallow picture. NInfer leads by 10%
shallow and by 27% at 128K without speculation, and the MTP3 gap grows with depth:

| Workload | llama.cpp `draft-mtp` | NInfer MTP3 (E8) |
|---|---:|---:|
| Code, shallow | 118.8 tok/s at 85.9% acceptance | 142.9 tok/s at 78.0% |
| Prose, 64K depth | 55.5 tok/s at 45.3% | 86.1 tok/s at 42.3% |
| Prose, 128K depth | 42.3 tok/s at 45.4% | 77.5 tok/s at 41.6% |
| Prose, 256K depth | no entry | 65.4 tok/s at 41.1% |
| Code, 256K depth | no entry | 91.2 tok/s at 72.1% |

The NInfer rows in this table use the 2026-08-17 generated corpora; acceptance on them runs
a few points below the 2026-08-15 payloads (code 78% against 81%), which accounts for the
difference from the headline 148.6 tok/s. The llama.cpp MTP rows required a reduced
131,584-token context; the draft buffers push VRAM
to 23.8 of 24 GiB, and the deployed 144K llama.cpp configuration cannot fit them at all.
NInfer serves 172,032 tokens with MTP in the same VRAM at INT8 KV, and the full native
262,144 with the E8 4-bit KV default. Acceptance matches per content type, so the decode gap
is engine time, not draft quality.

Full configurations, method, and raw numbers:
[NInfer against llama.cpp](docs/llamacpp-comparison.md).

## Quick start (Linux)

Requirements: an RTX 4090, a recent NVIDIA driver, Docker with the NVIDIA Container Toolkit.

Build the image and download the model once:

```bash
docker build --tag ninfer-4090:sm89 .
NINFER_MODEL_DIR="$PWD/models" bash scripts/download-qwen38.sh
```

Then start one of the three profiles. The API is available at `http://127.0.0.1:8080/v1`.

The profiles as written run one generation slot. `--max-concurrency 2` is measured
and worthwhile on the 4090: the second lane costs about 390 MiB (state pools plus a
doubled CUDA-graph allowance) while the KV page pool stays shared, so a lone session
still uses the full context; single-stream decode is unregressed and two sessions
decode batched at roughly 1.5x aggregate throughput, each lane keeping its own
resident prefix. Prefill still serializes across lanes, so a deep cold prefill
delays the other lane's first token.

Add `--turn-checkpoints 32` when clients edit conversation history (agent memory
updates, message rewrites, regenerated turns): the server then re-prefills from
the nearest retained turn boundary instead of from zero. The ring costs host
memory only, about 4.6 GiB per slot at 32 entries. See
[docs/turn-checkpoint-ring.md](docs/turn-checkpoint-ring.md).

Add `--host-prefix-cache-mib 20480` to preserve involuntarily evicted sessions in a process-local,
byte-bounded host block cache. Immutable metadata, GDN checkpoints, and 64-token KV page groups are
deduplicated across session branches; block recall time and frequency choose victims under memory
pressure. A later compatible prompt restores the cached continuation automatically and prefills
only its new suffix, without `/slots` client calls. DFlash is not supported.

Extra requests beyond the slots wait in the admission queue, and the queue deadline
defaults to 30 seconds. A deep prefill can hold a slot longer than that, so
parallel agent clients would fail with `request_queue_timeout`. The
`--pending-timeout-ms 600000` line raises the deadline to 10 minutes. On a
streaming request the timeout arrives as an in-band SSE error event after HTTP 200;
a client that does not parse error events sees a stream that ends without a
`finish_reason`. See [docs/serving.md](docs/serving.md) for the full queue
contract.

### Text-only, full 262K native context (E8 4-bit KV, default)

The E8 Conway-Sloane lattice KV mode (`rk4v4-e8`, ported from
[UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090); see
[the fork comparison](docs/udp-fork-comparison.md)) fits the model's entire native
262,144-token context on 24 GB with 1.4 GiB to spare:

```bash
docker run --rm --gpus all --publish 8080:8080 \
  --volume "$PWD/models:/workspace/models:ro" \
  ninfer-4090:sm89 \
  ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 262144 --kv-capacity 262144 \
  --max-concurrency 1 --max-pending-requests 16 \
  --pending-timeout-ms 600000 \
  --prefill-chunk 1024 --kv-dtype rk4v4-e8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --preserve-thinking
```

Measured against INT8 KV on this build: identical MTP acceptance at 111K depth
(78.8% vs 78.4%), a 5.7% decode tax (126.6 vs 134.2 tok/s on a shallow greedy code
probe), prefill within 1-2% at matched depth, and exact single-needle, 5-needle, and
code-detail retrieval through 260K tokens.

### Text-only, 168K context (INT8 KV, maximum precision)

```bash
docker run --rm --gpus all --publish 8080:8080 \
  --volume "$PWD/models:/workspace/models:ro" \
  ninfer-4090:sm89 \
  ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 172032 --kv-capacity 172032 \
  --max-concurrency 1 --max-pending-requests 16 \
  --pending-timeout-ms 600000 \
  --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --preserve-thinking
```

### With vision, full 262K context (E8 4-bit KV)

The vision scratchpad defaults to 8192 tokens (`--vision-max-tokens`, ported from
the same fork as the E8 KV modes) instead of the former hardcoded 32768. The
smaller scratchpad frees about 1.5 GiB, so the full native context fits next to
vision on 4-bit keys:

```bash
docker run --rm --gpus all --publish 8080:8080 \
  --volume "$PWD/models:/workspace/models:ro" \
  ninfer-4090:sm89 \
  ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 262144 --kv-capacity 262144 \
  --max-concurrency 1 --max-pending-requests 16 \
  --pending-timeout-ms 600000 \
  --prefill-chunk 1024 --kv-dtype rk4v4-e8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --preserve-thinking
```

The scratchpad bounds one image/video item at a time; Vision items are encoded sequentially and
reuse the same workspace. Before patch construction, the processor automatically downsizes every
item whose aligned grid would exceed the limit. The cap is independent per item, so appending a new
image never changes an earlier image's grid or invalidates an otherwise reusable cached prefix.
Impossible items (for example, a video's mandatory temporal grids alone exceeding the limit) still
fail as `media_budget_exceeded` before reaching the encoder. Each additional 1024 tokens of
scratchpad costs about 62 MiB of VRAM. Independent aggregate safeguards for media count, retained
raw patches, Vision attention work, and total prompt length still apply to extreme histories.

Expanded media tokens remain ordinary prompt tokens and therefore continue to occupy context KV
until the client removes or summarizes that history. The encoded bytes and float patch payload are
transient: after Vision prefill only the token/KV state plus compact media identity metadata remain,
and a compatible cached prefix skips Vision execution for historical images.

### The tradeoff

KV precision, vision, and maximum context trade against each other on a 24 GB card:

| Profile | KV mode | Context | KV runtime | Startup slack |
|---|---|---:|---:|---:|
| Text-only, MTP3 | `rk4v4-e8` | 262144 (256K) | 5.08 GiB | 1.37 GiB |
| Text-only, MTP3 | `rk2v4-e8` | 262144 (256K) | 4.01 GiB | 2.43 GiB |
| Text-only, MTP3 | `int8` | 172032 (168K) | 6.31 GiB | 136 MiB |
| With `--vision`, MTP3 | `rk4v4-e8` | 262144 (256K) | 5.41 GiB | 780 MiB |
| With `--vision` (32K scratchpad), MTP3 | `rk2v4-e8` | 262144 (256K) | 5.85 GiB | 329 MiB |
| With `--vision` (32K scratchpad), MTP3 | `rk4v4-e8` | 212992 (208K) | 6.06 GiB | 108 MiB |
| With `--vision` (32K scratchpad), MTP3 | `int8` | 98304 (96K) | - | ~1 GiB |

262,144 is the model's own context limit, so `rk2v4-e8` (2-bit keys, 96.2% cosine)
buys no additional context over `rk4v4-e8` in the text-only profile - only slack.
That slack is what pays for vision. With the former hardcoded 32,768-token vision
scratchpad, vision cost about 2.1 GiB (1.83 GiB of runtime buffers plus a
0.28 GiB tower): INT8 could only afford it at 96K, 4-bit keys topped out at
212992, and only 2-bit keys fit the full 262,144. The default 8192-token
scratchpad cuts the cost to about 0.6 GiB, and the full native 262,144 now fits
alongside vision on 4-bit keys with 780 MiB of slack. The vision
modes answer a two-swatch color oracle exactly at temperature 0, including with
the image buried under 52,700 tokens of text on `rk2v4-e8`. `rk2v4-e8` also passes
the text retrieval gates (single-needle at 260K, 5-needle at 118K, exact code
details at 168K) at a 10% decode tax (120.5 tok/s on the shallow code probe). The
INT8 text-only ceiling is near 176K: 172032 starts, and 196608 is rejected at
startup with a byte-exact deficit. The server validates memory before it listens,
so an oversized context fails fast instead of at request time.

For a native build, follow the [Linux build guide](docs/rtx-3090-linux.md) with
`CMAKE_CUDA_ARCHITECTURES=89` (the default in this fork). The build requires CUDA 12.8 or newer,
GCC 13, and CMake 3.28 or newer; the Docker image builds with CUDA 13.1.

## What this fork changes

- **`sm_89` retarget.** The CMake architecture pin, the runtime compute-capability check, and the
  NVFP4 stub gate now select `sm_89`. Most SM86 kernel schedules run unmodified on Ada; the
  INT8 attention prefill schedule is retuned (below).
- **Ada-retuned INT8 attention prefill.** The SM120 schedule spills registers on Ada and pays the
  consumer half-rate penalty for f32-accumulate HMMA. Arch-gated for `sm_89`: the full
  128-register budget, eight paired producer warps over `Bc` column halves with one named-barrier
  exchange per key tile, byte-permute V dequantization (bit-identical), and fp16-accumulated PV
  tiles folded into the fp32 running accumulator each tile. The kernel gains 30% at 64K depth
  (109 to 143 TFLOP/s on the `d256-h24-kv4` INT8 append shape); serve prefill gains 5-7% at
  88K-128K. Needle-in-a-haystack retrieval stays exact at both depths and all 84 suite tests
  pass, which bounds the fp16-accumulation numerics change.
- **Causal-tile partitioned key-block traversal.** Interior key blocks (wholly below the causal
  diagonal for the whole CTA tile) run a separate instantiation of the key-block body: KV stages
  with unconditional copies and the softmax drops its masking selects; boundary blocks keep the
  exact masked path. The idea comes from the
  [UDPSendToFailed fork](https://github.com/UDPSendToFailed/ninfer-4090) (c5f70526),
  re-implemented inside the retuned schedule above. Kernel: 144 to 165 TFLOP/s at 32K-224K
  context on the INT8 append shape (-12 to -13% latency), register count unchanged, bit-exact.
  End-to-end this is bounded by the attention wall share of this hybrid-GDN model: about +1%
  serve prefill at 51K on INT8 KV, within noise on the E8 modes, whose staging time is dominated
  by lattice decode rather than the removed guards.
- **`/v1/models` reports `context_window`.** Clients without access to a llama.cpp `/props` or a
  vLLM `max_model_len` can size prompts from the models payload.
- **llama.cpp-compatible `timings` on chat completions.** Responses and final stream chunks carry
  a top-level `timings` block (`prompt_n`/`predicted_n`, per-second rates, `ttft_ms`, `queue_ms`,
  `cache_restore_ms`, `cache_n`, `draft_n`/`draft_n_accepted`), so proxies such as llama-swap show
  per-request prefill and decode rates, MTP draft acceptance, and prefix-cache hits. Contributed by the
  [shantanusingh16 fork](https://github.com/shantanusingh16/ninfer-4090) of this repository.
- **`GET /metrics`.** Prometheus counters under llama.cpp-compatible names
  (`llamacpp:prompt_tokens_total`, `llamacpp:prompt_seconds_total`,
  `llamacpp:tokens_predicted_total`, `llamacpp:tokens_predicted_seconds_total`,
  `llamacpp:requests_processing`, `llamacpp:requests_deferred`), so existing scrapers read this
  server without changes. Prompt tokens count only computed prefill; prefix-cache hits are
  excluded, as in llama.cpp. Additional `ninfer:` series report request totals, prefix-cache
  hits, MTP draft/acceptance totals, and host-prefix-cache captures/hits/drops/evictions,
  capture/restore failures, transferred bytes and seconds, plus its live entry and byte gauges.
- **`GET /slots`.** A llama.cpp-shaped slot table read from the engine's real lane state: busy
  slots report their request's prompt and reused-prefix sizes, idle retained slots report the
  resident session's depth and its identifying `session_digest`. Truthful per-slot attribution
  holds at any `--max-concurrency`.
- **Slot session save/restore.** `--slot-save-path DIR` (off by default) enables llama.cpp-style
  `POST /slots/{id}?action=save|restore|erase`: one idle slot's complete resident session -
  paged Text and MTP KV, GDN linear-attention state, turn checkpoint, and prefix identity -
  moves to or from disk, and a restored slot reuses the cache across server restarts instead of
  re-prefilling (a 6.9k-token session restores in about 0.1 s against a multi-second reprefill).
  Sessions are identified by a stable `session_digest`; chat completions carry `id_slot` and the
  digest next to `timings`, and `save`/`erase` accept an `if_digest` precondition checked
  atomically, so a client always persists exactly the session it means. Restore extends the
  saved frontier (or its turn checkpoint); the GDN state cannot rewind further, and the DFlash
  backend is not supported. Details in [docs/serving.md](docs/serving.md).
- **Reuse-aware lane choice.** When prefix reuse ties (typically zero for a fresh session),
  admission picks the lane whose occupation costs least to replace - an empty lane before any
  retained session, then the shallowest - so a burst request no longer evicts a deep resident
  session while a free lane exists.
- **Turn checkpoint ring.** `--turn-checkpoints N` (off by default) keeps up to N past turn
  checkpoints per slot in host memory. A prompt that rewrites the middle of its history -
  an edited message, an updated agent memory block, a regenerated earlier turn - restores at
  the deepest checkpoint below the edit instead of re-prefilling from zero; generation after
  the restore is greedy-identical to a cold prefill. One checkpoint holds the GDN
  linear-attention state (about 147 MiB of host memory on Qwen3.8-27B); the attention KV
  needs no copy. Slot snapshots carry the ring across restarts (format version 2, written
  only when the ring is non-empty, so existing files stay readable everywhere). The
  recommended value is 32. Details in
  [docs/turn-checkpoint-ring.md](docs/turn-checkpoint-ring.md).
- **Auto-save on eviction.** `--auto-save-evicted` (off by default, requires
  `--slot-save-path`) spills an involuntarily evicted session - checkpoint ring included -
  back to the slot file it was last saved to or restored from, before the eviction destroys
  it. Rotating more sessions than slots then loses nothing: the next restore recovers the
  session at its latest frontier. Explicit `erase` never auto-saves.
- **Automatic host prefix block cache.** `--host-prefix-cache-mib N` (off by default) captures an
  involuntarily evicted session whether or not the client used `/slots`. Logical session manifests
  share identical immutable blocks, remain matchable while restored into a lane, and return only
  changed/new blocks on the next capture. Admission restores the deepest compatible frontier and
  follows the ordinary suffix-prefill path. Under pressure, cold blocks are selected by recall
  time plus a logarithmic recall-count bonus; dependent unpinned manifests are removed atomically.
  DFlash is not supported.
- **KV-affinity admission.** With the host cache enabled, `--kv-affinity-burst 5` lets work matching
  a resident KV owner pass older cold work at most five contended admissions before rotating to
  the oldest competing session. `--kv-affinity-grace-ms 1500` briefly waits for the just-finished
  conversation's next turn. Both are configurable; a burst of `0` disables affinity scheduling.
- **NVFP4-A4 test gating.** The A4 activation tests skip on hardware without FP4 tensor cores
  instead of aborting. The full remaining suite passes on the RTX 4090.
- **E8 lattice KV quantization (ported).** The `rk8v4`/`rk4v4`/`rk4v4-e8`/`rk2v4-e8` KV modes
  and the 262K-to-1M visible-keys envelope lift from the
  [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) sibling fork,
  merged under this fork's retuned `sm_89` attention prefill schedule. The E8 codec verifies
  bit-exactly against the upstream microbenchmark (96.155% / 98.678% cosine); their 1 GiB
  CUDA-graph allowance bump was deliberately not taken (it would evict the INT8 168K profile).
  Method and measurements in [docs/udp-fork-comparison.md](docs/udp-fork-comparison.md).
- **Configurable vision scratchpad (ported).** `--vision-max-tokens` comes from the same fork
  and sizes the vision encode workspace (default 8192 tokens, formerly hardcoded 32768). This
  fork keeps the processor in lockstep and automatically fits each image/video grid into the
  reusable per-item workspace. Physical per-item minima and the separate aggregate processor
  safeguards still report `media_budget_exceeded` when exceeded.

## Known limits on the RTX 4090

- Prefill trails llama.cpp by 16-24% on full 32K-128K prompts under matched conditions (see
  the depth sweep above). The rate is flat across `--prefill-chunk` 1024 to 2688, so the
  chunk size is not the lever. With the attention schedule retuned, the remaining gap sits in
  the custom quantized GEMMs, which run about 10% below cuBLAS on Ada. Decode is where this
  engine leads.
- Keep `--prefill-chunk` at 2688 or below. This fork carries measured `sm_89` cooperative
  residency tables (the former hard abort above chunk 1024 is fixed), and chunks through 2688
  stay on split-K. Larger chunks route to the unsplit schedule, which is marginally less
  accurate at its onset (about 1e-5 relative).
- `--max-concurrency 2` is measured on the 4090 (see Quick start); higher lane counts are
  untested here, and the published cohort results in the
  [3090 base](https://github.com/Don-Chad/ninfer-3090) do not transfer directly.
- Prefill is strictly serialized across lanes with no chunk-level interleaving, and decode
  starves while any prefill runs: a short request submitted behind a 31k-token cold prefill
  measured a 13.5 s first token. Concurrency pays off for decode and for per-lane resident
  prefixes, not for prefill fairness.
- The limits of the base engine apply: one process, one GPU, one model, bounded FIFO admission,
  no multi-GPU execution, no weight offload.

## Artifact

| Model | Artifact | Size |
|---|---|---:|
| Qwen3.8-27B | [official NInfer groupwise artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 16.96 GiB |

The artifact is architecture-independent; the model card's RTX 5090 requirement describes the
upstream engine, not the file. Verify the download against the SHA-256 published on the card.

## Reasoning effort

Qwen3.8-27B has three trained reasoning depths plus an off switch. OpenAI Chat Completions
accepts a top-level `reasoning_effort` field (`low`, `medium`, `xhigh`) and a top-level
`enable_thinking` boolean; hidden reasoning returns separately as `message.reasoning_content`.
The `chat_template_kwargs` request field of llama.cpp is not supported and is rejected. For the
CLI, pass `--reasoning-effort` or `--no-thinking`. Sampling defaults come from the model card and
switch with the thinking mode.

## Serving APIs

OpenAI Chat Completions, OpenAI Responses with streaming and local continuation state, Anthropic
Messages, prompt-rendered function tools with parsed tool calls, compatible-prefix reuse, and
JSONL request logs. See [HTTP serving](docs/serving.md) and [CLI usage](docs/cli.md).

## Upstream and credits

- [Neroued/ninfer](https://github.com/Neroued/ninfer) - the engine, developed for the RTX 5090
  (`sm_120a`).
- [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) - the SM86 compatibility layer,
  ReplaySSM integration, and Qwen3.8 runtime support this fork builds on. Its
  [v0.6.1 release notes](RELEASE_NOTES_0.6.1.md) describe the inherited state.
- [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) - a sibling
  RTX 4090 port from the same 3090 base. The rotated and E8-lattice KV-cache quantization
  modes (`rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8`), the E8 codecs, and the 1M visible-keys
  envelope are their work, cherry-picked here with authorship preserved. The full 262K
  default profile exists because of it; see
  [the fork comparison](docs/udp-fork-comparison.md).
- [jram4/ninfer-4090](https://github.com/jram4/ninfer-4090) - an earlier RTX 4090 port of a July
  2026 snapshot. Its Ada dispatch tuning targets a kernel organization that upstream has since
  replaced, so this fork starts from the current 3090 base instead.

## License

Apache License 2.0. See [LICENSE](LICENSE).
