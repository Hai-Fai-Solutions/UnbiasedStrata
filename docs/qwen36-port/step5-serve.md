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
- Checkpoint reuse: the second turn reused 27 of 278 prompt tokens; the reuse boundary is a measurement for the
  ladder (a long conversation, the `tokens reused` figure per turn).
- `expert cache ... policy is` prints an empty policy name for native packs (cosmetic).
- At 262,144 the Q4 fixed set leaves 19.3 GiB for experts; 9,980 slots fit; the planner's "10240 of 10240 (100%)"
  line and the cache's "9980 slots" disagree by the 700 MiB runtime reserve the cache keeps; the plan line should
  say what the cache will actually hold.
