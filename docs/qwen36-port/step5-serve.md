# Qwen3.6 port, step 5, first part: serving on the 4090, and what broke below full residency

Branch `qwen36-step4`, 2026-10-13, VM 221 (RTX 4090, 24 GB; the user handed the box to this work at 02:05 UTC and the
Flash-Next service on it was stopped).  Tree `/models/work/qwen36-step4`, binary `build/unbiased-strata`, logs under
`/models/work/serve/`.  The 1080 Ti (VM 211) is off limits: another session's service runs there.

## The serve command

```
./build/unbiased-strata --model /models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --pack /models/work/packs/qwen36-q4 \
    --mtp /models/work/packs/qwen36-mtp-draft-full --ctx 262144 --threads 8 --host 127.0.0.1 --port 18090 -- --kv q4_0
```

`--mtp` takes a Qwen3.6 draft pack (`tools/mtp_qwen36_pack.py`); `qwen36-mtp-draft` carries a `draft_vocab.bin`
subset, `-full` has none.  Everything after `--` goes to the engine: `--vram-cap MiB`, `--ram-cap GiB`, `--kv
fp16|q4_0|int8`, `--expert-cache N|auto`, `--pcie-frac F`, `--prompt-cache N`, `--vram-reserve-mib`.

## Faults found in order, all fixed on the branch

| # | Symptom | Cause | Fix |
| --- | --- | --- | --- |
| 1 | `unbiased-strata: missing file: .../qwen36-mtp-draft-full/draft_vocab.bin` | the front end required the Flash-Next draft vocabulary subset | optional in the front end as it already was in the engine (3a806b5) |
| 2 | first request: 220 tokens generated, then `saving a conversation checkpoint failed` (`conversation snapshot: invalid model geometry`, then `invalid indexer running-state target`) | the prompt-cache checkpoint and the snapshot rejected a geometry with no indexer (idx heads / key dim 0) and no hyper-connection rate, and demanded the indexer's tail / dead / block-position / pooled buffers | zeros allowed where `has_indexer()` is false and `hc` is 1, no indexer buffers or block positions for dense attention, pooled rows 0 (3a806b5).  Two-turn conversation: 1 then 2 checkpoints, prefix reused |
| 3 | Q4 at 262,144: `2, 3! 5, 7, 11, !!!!!!...` (token 0 forever); IQ3 under any `--vram-cap` the same; both files fine at 8K / 32K; IQ3 at 256K fine | **the pool's native-pack phases iterated the canonical Q2_0 widths (FF 640, H 2560) instead of the layer's (512, 2048)**: every expert miss computed on the CPU wrote 512 rows past its output row.  The runner's and drafter's tests all ran with every expert resident, so the miss path had never executed on Qwen3.6 | `run_split_multi_native` at `NativeFmt::n_ff / n_embd`; `native_fmt` refuses wider layers; `pool_native_widths_test` (22d72e6) |

How fault 3 was isolated (each row one serve, the prime-numbers prompt, greedy, `bangs` = count of `!` in 120-200
tokens):

| Run | file | ctx | KV | slots resident | misses on | result |
| --- | --- | --- | --- | --- | --- | --- |
| e1 | Q4 | 8192 | fp16 | 10,240 / 10,240 | - | correct, 362 tok/s |
| e3 | Q4 | 262,144 | q4_0 | 9,980 | CPU + DMA | `2, 3, 5, 7, !!!` |
| e2 | Q4 | 262,144 | q4_0, `--vram-cap 20480` | 8,132 | CPU + DMA | `!!!` from the first token |
| e4 | Q4 | 262,144 | fp16 | 8,120 | CPU + DMA | `!!!` |
| e5 / e6 | Q4 | 8192 / 32,768 | q4_0 | 10,240 | - | correct, 354 / 356 tok/s |
| t1 | IQ3 | 262,144 | q4_0, `--vram-cap 15360` | 10,212 | CPU + DMA | `!!!` |
| t2 | Q4 | 262,144 | q4_0, `--vram-reserve-mib 1200` | 9,838 | CPU + DMA | `!!!` (so not VRAM pressure) |
| t3 | Q4 | 8192 | fp16, `--expert-cache 8000` | 9,587 | CPU + DMA | `!!!` (the cheap reproduction) |
| t4 | Q4 | 8192 | as t3, `--pcie-frac 1.0` | 9,587 | DMA only (GPU computes) | **correct**, 305 tok/s |
| t5 | Q4 | 8192 | as t3, `--pcie-frac 0.0` | 9,587 | CPU only | `!!!` |
| t6 / t7 / t8 | Q4 ggml dots only / IQ3 / one worker | 8192 | as t5 | 9,587 | CPU only | `!!!` all three: not the kernels |
| fix | Q4 | 8192 | as t5 | 9,587 | CPU only | correct, 328 tok/s |
| fix | Q4 | 262,144 | q4_0 | 9,980 | CPU + DMA | correct, 322 tok/s |
| fix | IQ3 | 262,144 | q4_0, `--vram-cap 11264` | 5,762 (56%) | CPU + DMA | correct, 169 tok/s; 63,526 VRAM hits of 73,553 lookups |

| 4 | Pascal host (VM 211, step5-1080ti, REGINA-106): with CPU-computed misses iq3-test20 51/64 top-1, KL 0.84; Q4 52-54/64, KL 0.25; DMA misses correct; deterministic; `--spec 1` changes it | **the multi-token CPU dispatch addressed token t's activation at t x 2560 and its output row at i x 2560** (the canonical widths pulled in by `using namespace`), while the window publishes at the layer's 2048: every token after a group's first used another token's activation.  Named by the Codex review of 22d72e6; the 4090's serve looked right because its text was checked, not its logits | the dispatch at the layer's width with refusals (the commit after 22d72e6); after it on VM 211 all three IQ3 prompts 64/64 (KL 1.4e-3 / 4.8e-3 / 2.0e-3), q4-test20 62/64 KL 6e-3, q4-prime 64/64 KL 8e-4, CPU misses forced.  The ladder points taken with the old binary are rerun |
| 5 | the 4090 sweep's `--ram-cap 12` points: "ArenaExpertSource: the expert arena needs 18.32 GiB ... memory limit is 12.00 GiB ... --mmap-experts" | the plan chose RAM copies but the arena source still pinned the whole file | the engine picks the mapped tier with profile-ranked RAM copies itself when the arena exceeds the limit or available RAM (same commit) |
| 6 | planner (Codex review): a full-head draft pack charged 515 MiB the drafter never allocates; k8v4 undercounted the drafter's INT8 K/V by 60 MiB; a lowered context was printed but not applied | | all three in the same commit; multi-GPU helpers refused for this geometry |

| 7 | second Codex review of fa69e82: the front end kept validating and advertising the requested `--ctx` when the engine's plan lowered it; the automatic mapped tier ignored an explicit `--shared-expert-arena` and the multi-GPU residency rules; the plan's session term keeps the RoPE table at the requested context | | the front end takes the context from the engine's `READY <n>` line and bounds requests by it; the automatic choice is skipped with a shared arena or the helpers.  The RoPE table at a lowered context is left as a conservative overcount (at most 32 MiB from 262,144 to 131,072), noted here rather than threaded through the plan |

## What the 4090 serves now

| file | ctx | fixed set | experts resident | VRAM free after load | short answer decode | drafts accepted |
| --- | --- | --- | --- | --- | --- | --- |
| IQ3_XXS, `-full` draft | 262,144 | 4,331 MiB (weights 1,596, drafter 831, KV 1,440, session 162, reserve 300) | 10,240 / 10,240 (10.35 GiB) | 8,758 MiB | 319-359 tok/s | 157 / 181, 223 / 237 |
| Q4_K_XL, subset draft | 262,144 | 4,823 MiB (weights 2,038, drafter 882, KV 1,440, session 162, reserve 300) | 9,980 / 10,240 | 434 MiB | 322 tok/s | 149 / 152 |

Also checked through the HTTP front end: streaming (SSE chunks, `reasoning_content` deltas), a two-turn
conversation with checkpoint reuse, a tool call (`<tool_call><function=read>` emitted; the earlier "!!!" inside its
arguments was fault 3).  The response's `model` field is now the GGUF's base name (was the Flash-Next constant).

## Observations for the measurements

- The RAM tier reports `expert copies in RAM: 0` on this host: with 62 GB the whole expert arena is pinned host
  memory (`cudaHostRegister PORTABLE`, 18.3 GiB for Q4), and a miss is served from it by DMA or the CPU; `ram_reads`
  and `file_reads` on `/health` stay 0.  On a 12 GiB host the arena cannot hold the file and the mapped-from-disk
  tier is what the ladder measures (`--ram-cap` emulates it here).
- Checkpoint reuse: the second turn reused 27 of 278 prompt tokens.  Cause and the opt-in that lifts it: the A5
  section below, check (d).
- `expert cache ... policy is` prints an empty policy name for native packs (cosmetic).
- At 262,144 the Q4 fixed set leaves 19.3 GiB for experts; 9,980 slots fit; the planner's "10240 of 10240 (100%)"
  line and the cache's "9980 slots" disagree by the 700 MiB runtime reserve the cache keeps; the plan line should
  say what the cache will actually hold.

## A5 on the 4090: the serving checks, after the streaming fix (REGINA-110)

**The fault.**  With `"stream": true` the answer arrived as one `content` delta before the final chunk; only
`reasoning_content` streamed.  Not the parser: `run()` in `frontend/main.cpp` cleared the content of every partial
parse together with its tool calls, a rule that came with 46c46ec so that partial tool-call JSON (closed and
normalized by the parser on each token) could never change text already sent.  That holds for a call's arguments,
not for the answer.  c3b41cc: reasoning and content are published as they arrive, tool calls once from the complete
parse (`StreamParse`); a partial parse whose content does not extend what was sent publishes no content until it
does; a non-streaming request is parsed once instead of after every token.  **A tool call still arrives whole**, in
one delta (name, id and full arguments) when the message ends; the text before a call streams.
`make test-chat-parser` now feeds seven responses a character at a time (reasoning on and off, tools declared and
not called, text before a call, one and two calls) and requires the pieces to add up to the whole parse, the answer
to arrive in many deltas and each call in one; it passes on the IQ3_XXS and Q4_K_XL files.  The ticket's request on
the final binary: 200 content chunks of 203 (was 1 of 3).

**The checks.**  VM 221, binary sha256 `46c5747f23dcbdeb` (c3b41cc), `--mtp qwen36-mtp-draft --ctx 262144 --threads
8 ... -- --kv q4_0`, greedy, client and server on the VM (`/models/work/serve/a5-20261013/`: `a5.py`, `serve.sh`,
one `.json` and `.log` per configuration).  All nine checks pass in all three configurations.

| check | Q4_K_XL, no cap (9,980 slots) | Q4_K_XL, `--vram-cap 11264` (2,910 slots) | IQ3_XXS, `--vram-cap 6144` (809 slots) |
| --- | --- | --- | --- |
| (a) chat, not streamed, thinking off: "capital of France" | pass, 0.15 s | pass, 0.54 s | pass, 0.54 s |
| (a) chat, not streamed, thinking on: 17 x 23 = 391 | pass, 314 tokens at 313 tok/s | pass, 314 at 76 | pass, 410 at 70 |
| (b) stream, thinking on: reasoning deltas, first, largest gap | 1,087, 0.081 s, 14.1 ms | 467, 0.293 s, 92.8 ms | 418, 0.348 s, 78.6 ms |
| (b) the same request: content deltas, first, largest gap | 89, 3.616 s, 13.8 ms | 82, 5.803 s, 61.6 ms | 53, 6.181 s, 82.3 ms |
| (b) the same request: tokens, decode, largest gap between any two deltas | 1,179 at 303 tok/s, 14.1 ms | 552 at 83, 92.8 ms | 474 at 72, 82.3 ms |
| (b) stream, thinking off (the ticket's request): content deltas, first, largest gap | 212 of 213 tokens, 0.076 s, 11.5 ms | 203 of 204, 0.330 s, 53.8 ms | 209 of 210, 0.345 s, 64.2 ms |
| (c) tool round trip, not streamed: call, result, answer | pass: `get_weather {"city":"Paris"}`, `tool_calls`, then "17 C with light rain", `stop` | pass | pass |
| (c) tool round trip, streamed | pass: reasoning 658 deltas, the call in 1 delta at 2.40 s; the answer in 14 content deltas | pass: 517 reasoning deltas, call in 1 at 5.40 s; 15 content deltas | pass: 173, call in 1 at 2.86 s; 18 content deltas |
| (c) tools declared, not called, streamed, thinking on: content deltas, first, largest gap | 73, 2.597 s, 14.0 ms | 74, 10.318 s, 62.2 ms | 81, 12.442 s, 81.1 ms |
| (c) the same, thinking off | 76, 0.032 s, 11.4 ms | 69, 0.067 s, 41.7 ms | 66, 0.093 s, 41.9 ms |
| (d) three-turn chat, tokens reused of the prompt per turn | 0 / 23, 18 / 129, 124 / 172 | 0 / 23, 18 / 133, 128 / 176 | 0 / 23, 18 / 178, 173 / 246 |

"First" is from the request to that delta, so with thinking on the first content delta comes after the reasoning.
Deltas come in bursts of an accepted draft window (median gap 0.05-0.1 ms); the largest gap is one verify round,
longer under a cap where it waits on expert misses.  No keepalive comment was needed in any run.

**(d) Why the reused prefix is short, and what lifts it.**  The engine keeps a checkpoint at the last
`<|im_start|>` of each prompt and one at the end of each reply.  The next request reuses the second only if it
renders the reply as it was generated, and in a plain chat Qwen3.6's template does not: a reply before the last
user message is rendered without its `<think>` block (with thinking off, without the empty `<think>\n\n</think>`
the generation prompt carried), whether or not the client sends `reasoning_content` back.  The tokens leave the
generated ones right after the assistant header, and the reuse is the previous prompt less its 5-token header: 18
of 129, 124 of 172, and the 27 of 278 above.  Each turn rereads the last reply and the new message, never the older
history.  Not an engine fault.  Two cases reuse the reply too:

- a tool loop: a reply after the last user message keeps its reasoning in the template.  pi's three requests (check
  e): 0 of 1,604, then 1,867 of 1,948 and 2,047 of 2,118;
- `"chat_template_kwargs": {"preserve_thinking": true}` with `reasoning_content` sent back, which the front end now
  passes to the template (3aa9eed; binary `e4d8ec0c07473e13`): four turns reuse 0 / 22, 551 / 569, 1,010 / 1,027,
  1,365 / 1,382.  Opt-in, because the earlier reasoning then stays in the context the model reads.  Not every turn
  hits: in two of eight such turns the re-rendered reply did not match the generated tokens (578 of 1,330; 18 of
  1,138) and the request fell back to the header checkpoint; which replies re-tokenize differently was not traced.

**(e) pi.**  pi 0.85.1 on the Mac, its own agent directory (`PI_CODING_AGENT_DIR`, the user's `~/.pi` untouched),
provider `openai-completions` at the engine through an SSH tunnel, Q4_K_XL no cap, `pi -p --mode json`: it wrote
`primes.py` with its `write` tool, ran it with `bash`, and reported `2 3 5 ... 71`.  Three requests, two tool calls,
80 thinking deltas and 82 text deltas seen by pi; nothing dropped by the parser.

**Tests.**  The Qwen3.6 ctest subset on the free card (`ctest -R "qwen36|vram_plan_test|pool_native_widths_test"`):
21 of 21.  `make test-frontend` and `make test-chat-parser` pass.  The engine is untouched by these two commits, so
no logits comparison was owed.

**Seen, then chased.**  The same greedy request does not always generate the same tokens on one server (check b,
thinking on, Q4 no cap: 632 tokens in one pass, 1,179 in the next).  Traced in the next section: not a fault.
Images (mmproj) were out of scope.

## The same greedy request, different tokens: which paths differ, and by how much

2026-10-13, VM 221, binary `11da5388afdfabcb`.  The engine was driven on its own stdin protocol (`unbiased-strata
--engine --serve ...` with the front end's arguments, `GEN <max_new> <ids>`), so the record is token ids.  Two
prompts from A5, rendered by the front end: `think` (26 tokens, "What is 2+2? Answer, then write three sentences
about addition.", thinking on) and `nothink` (25 tokens, the lighthouse request, thinking off), `max_new` 3,000.
Material in `/models/work/serve/nd-20261013/` (`nd.py` the driver, `cmp.py`, `trdiff.py`, `acc.py`, one `.jsonl` of
token ids and one engine log per run, 23 GB of logit dumps).

**Verdict: benign.**  No state leaks between requests.  The output is a pure function of the request and of which
of three numerically different, individually correct paths each number went through; two things move a request
between those paths from one pass to the next:

1. **Where an expert is computed.**  A resident expert and one read over PCIe (DMA) run the GPU kernel (llama.cpp's
   CUDA arithmetic: activations as Q8_1); a miss computed on the CPU runs ggml's CPU kernel (activations as Q8_K).
   The two round differently, as llama.cpp's own CPU and CUDA back ends do.  The adaptive tier (`--adapt-every`, 4
   rounds) swaps routed-but-missing experts into VRAM and keeps them across requests, so the second pass of a
   request finds experts on the card that the first computed on the CPU.  It settles: once the cache stops
   changing the output stops changing.
2. **How the prompt's state was built.**  A part of at most `--short-read` (64) tokens and everything a checkpoint
   or the live session supplies went through the verify windows; a longer part read from token 0 goes through the
   batched prompt path, which is different arithmetic (step4-runner.md: the two agree to 2.1e-2 on the fixture).
   Restoring a checkpoint is exact; what differs is a state made by decode against the same state re-read in a batch.

With misses present a third input reaches the output through (1): the lookup drafter's policy picks window sizes
from measured round times, the adaptive tier counts rounds, and the PCIe share of a layer's misses is a fraction
of the misses in that window.  With `--suffix-draft 0` a fresh engine gave the same tokens with and without the
tracing below (which slows every window); with the default it gave 557, 554 and 555 tokens for `think`.  With
every expert resident the window shape does not reach the output (the window is bitwise independent of its shape).

**Which configurations repeat.**  Ten requests of each prompt, alternating, on one engine unless noted:

| configuration | misses run on | distinct outputs, `think` / `nothink` | tokens generated |
| --- | --- | --- | --- |
| IQ3_XXS, 262,144, q4_0, no cap (10,240 resident) | - | 1 / 1 | 480 / 197 |
| the same, `think` alone 10 times (21 of 26 reused) and with `--prompt-cache 0` | - | 1, the same tokens | 480 |
| Q4_K_XL, 8,192, fp16 (10,240 resident) | - | 1 / 1 | 664 / 218 |
| Q4, 8,192, `--expert-cache 8000 --pcie-frac 1.0`, adaptive tier on or off | GPU (DMA) | 1 / 1, **the all-resident tokens** | 664 / 218 |
| Q4, 8,192, `--expert-cache 8000 --pcie-frac 0.0 --adapt-every 0` | CPU, fixed set | 1 / 1 | 632 / 191 |
| the same with the adaptive tier | CPU, changing set | 7 / 6, the same from the 7th request | 511, 557, 639, 633, 1,451, 478, 479 x 4 |
| Q4, 262,144, q4_0, no cap (9,980 slots): the A5 configuration | CPU (0.05% of lookups) | 3 / 3, the same from the 3rd | 557, 620, 553 x 8 / 221, 213, 214 x 8 |
| the same, `--prompt-cache 0` | CPU | the tokens of the row above | |
| the same, `--adapt-every 0` | CPU, fixed set | 1 / 1 | 558 / 211 |
| the same, `--suffix-draft 0` | CPU | 2 / 3 | 550, 551 x 9 |
| the same, `--mtp-max-t 1 --suffix-draft 0` (6 requests) | CPU | 4 / 2 | 561, 552, 1,014, 552 x 3 |
| the same, a new engine for every request | CPU | 1 / 1 | 557 / 196 |

Every configuration with one path is bit-repeatable; the only ones that are not have CPU misses and a cache that
is still changing.  The checkpoint cache, the drafter and the lookup drafter are not sources.  The second cause,
with every expert resident (IQ3 no cap; Q4 at 8,192), one engine against a new engine reading from token 0:

| request | state from | tokens | first difference |
| --- | --- | --- | --- |
| 258-token prompt, three times | 0 reused, then 253 reused + 5 | identical, and identical to a new engine | - |
| `think`, its 480 / 664 generated tokens and a second user turn (529 / 713) | the live session (505 / 690 reused) or the header checkpoint (524 / 708) | identical to each other | |
| the same on a new engine | the batched prompt path, all of it | IQ3 554 against 495, Q4 437 against 401 | token 191 / 140 |
| a 73-token second turn | 21 reused + 52 through the windows, or 68 reused | identical to each other | |
| the same on a new engine | 73 read from 0 | IQ3 890 against 494, Q4 465 against 438 | token 61 / 61 |

**The margins where they part, and the accuracy of every state.**  `STRATA_SERVE_LOGIT_TRACE=FILE` (new, off by
default) writes the head's five largest logits, the window and its misses per emitted token, and
`STRATA_SERVE_LOGIT_DUMP=PREFIX` the whole row (the last window's rows past the stop token included);
`tools/qwen36_teacher_llama.cpp` grew `--prompt-ids` and `--force-ids`, so llama.cpp (F16 K/V, the 4090) is
teacher-forced along each run's own tokens and every position of every run is scored, with the path that run took.  Margins are logit differences between the two tokens chosen.

| first difference | this pass | the other pass | llama.cpp |
| --- | --- | --- | --- |
| Q4 no cap, `think`, request 1 against 2, token 249 | 0.03 for A | 0.23 for B | 0.16 for B |
| request 1 against 3, token 245 | 0.10 for A | 0.78 for B | 0.09 for B |
| request 3 against 4, token 531 | 0.007 for A | 0.20 for B | 0.31 for B (its own pick is a third token, 0.12 above B) |
| IQ3 no cap, session state against a batched re-read, token 191 | 0.27 | 0.34 | 0.03 |
| Q4 8,192, the same, token 140 | 0.08 | 0.03 | 0.012 |
| IQ3 / Q4 second turn, windows against batched, token 61 | 0.09 / 0.11 | 0.37 / 0.08 | 0.32 / 0.03 |

Over the common prefix two passes differ by a mean of 0.12-0.13 logits on the reference's 20 largest (KL 1.4e-3
to 2.1e-3 between them), the size of each pass's distance from llama.cpp.  Every state against llama.cpp, the
whole run and its first 64 positions (the rule: 62 of 64, mean KL below 1e-2):

| run, path | positions | top-1 | mean KL | max KL | first 64: top-1, mean KL |
| --- | --- | --- | --- | --- | --- |
| Q4 8,192 fp16, all resident (serve) | 664 | 660 (99.4%) | 8.3e-4 | 0.077 | 64, 5.6e-4 |
| Q4 8,192, CPU misses, `--expert-cache 8000`, fixed (0.7% of lookups) | 632 | 624 (98.7%) | 8.9e-4 | 0.053 | 64, 5.5e-4 |
| Q4 8,192, CPU misses, `--expert-cache 4000`, fixed (33% of lookups) | 547 | 542 (99.1%) | 8.9e-4 | 0.046 | 64, 1.3e-3 |
| Q4 no cap 262,144 q4_0, requests 1 / 2 / 3 / 4 of `think` | 555 / 630 / 557 / 553 | 98.4 / 98.4 / 98.7 / 98.6% | 3.4e-3 / 3.3e-3 / 4.0e-3 / 3.8e-3 | 0.08 / 0.30 / 0.16 / 0.13 | 63, 1.8e-3 to 2.1e-3 |
| IQ3 no cap 262,144 q4_0, second turn: session state / batched re-read | 495 / 554 | 98.8 / 99.3% | 3.4e-3 / 4.7e-3 | 0.11 / 0.10 | 64, 1.4e-3 / 64, 1.0e-3 |
| IQ3, 73-token second turn: windows / batched | 494 / 890 | 98.2 / 98.5% | 4.1e-3 / 4.0e-3 | 0.15 / 0.14 | 62, 3.5e-3 / 63, 1.4e-3 |
| Q4 8,192, second turn: session state / batched re-read | 401 / 437 | 99.8 / 99.3% | 9.4e-4 / 8.1e-4 | 0.023 / 0.033 | 64, 2.5e-4 / 64, 2.0e-4 |
| Q4 8,192, 73-token second turn: windows / batched | 438 / 465 | 99.3 / 99.4% | 9.0e-4 / 1.5e-3 | 0.021 / 0.12 | 64, 8.3e-4 / 63, 6.2e-4 |
| `strata --teacher-force`, 400 tokens of the all-resident run: resident / DMA / CPU 8000 / CPU 4000 / CPU 4000 fixed | 400 | 393 / 393 / 396 / 396 / 395 | 9.2e-4 / 9.2e-4 / 9.6e-4 / 7.9e-4 / 8.8e-4 | 0.028 to 0.036 | 63 or 64, 3.6e-4 to 5.9e-4 |

Every top-1 disagreement with llama.cpp in these runs (113 in 9,912 positions) sits where llama.cpp's own margin is
under 0.75 logits; one position in 20 to 25 has a top-2 gap under 0.5, so a greedy run of 500 tokens meets some
twenty places where a pass can turn.  The teacher-forced resident and DMA dumps are the same bytes (`cmp`).  No
rising drift: Spearman(position, KL) is between 0.00 and +0.13 in every run but the IQ3 batched re-read (+0.20, KL
3.2e-3 in its first half and 6.1e-3 in its second).
The 262,144 rows carry the q4_0 K/V against llama.cpp's F16 (mean KL 3-5e-3 where the fp16 rows have 8e-4 to
1.5e-3); the K/V format is the likely reason (not isolated here), and it is in every pass alike.

**For a run that must repeat.**  `--pcie-frac 1.0` (every miss on the GPU: the all-resident tokens whatever the
cache holds, at the DMA path's cost), or `--adapt-every 0 --suffix-draft 0` (the miss set fixed, no timed policy).
A conversation that continues on one engine still differs from the same prompt read from token 0 on another.

**Not checked.**  The 1080 Ti (VM 211, off limits); a capped Q4 or IQ3 with thousands of misses per request beyond
the `--expert-cache 4000` rows; sampled (temperature above 0) requests; requests cancelled mid-generation followed
by the same prompt; the q4_0 K/V's own share of the 262,144 rows' KL.  `--teacher-force` with `--mtp` committed
the drafter's guesses as if they were the oracle's tokens (70 of 400 top-1 on a run that scores 394 without it): it
is now refused.
