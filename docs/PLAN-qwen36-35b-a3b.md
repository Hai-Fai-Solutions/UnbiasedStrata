# Plan: UnbiasedStrata on Qwen3.6-35B-A3B, any single NVIDIA card from Pascal up, 256K context

Written 2026-10-11, revised the same day three times: after reading
[Project Maya](https://github.com/mw00/project-maya); after the target widened from one card to a range;
after the llama.cpp-backend fallback was removed. Status: proposed, nothing started.

Engine base: this repository's Strata snapshot (`6f32ec0`, engine 0.1.39). Maya is a method reference only.
llama.cpp appears in two roles only: the external speed baseline (bare `llama-server`, never inside the
binary) and the numerical oracle for parity. There is no fallback path.

## 0. Target and what "adapt" means

**Target:** one NVIDIA GPU, compute capability 6.1 or newer (GTX 10 series through RTX 50), **6 GB to 32 GB**
of VRAM, Linux, x86-64 with AVX2. The engine sizes itself at start: fixed weights first, then the KV cache
for the requested context, then every remaining byte of VRAM becomes expert-cache slots. The same binary
and the same model files on every card; only speed differs. Q4 is the default quant. Context 262,144 where
it fits, never below 131,072 without saying so. **Multi-GPU is out of scope**; the front end already
refuses `--gpus`, and that stays.

The Strata engine in `src/` runs one architecture, Qwen3.8-Flash-Next (`qwen4_exp_text`). Qwen3.6-35B-A3B
(`qwen3_5_moe_text`) is the same family one generation earlier, with the extras removed. From the two
models' `config.json`:

| | Qwen3.8-Flash-Next (what the engine runs) | Qwen3.6-35B-A3B (the target) |
| --- | --- | --- |
| Hidden / layers | 2560 / 48 | 2048 / 40 |
| Layer pattern | 3 linear attention then 1 full attention (`full_attention_interval` 4) | the same |
| Linear attention (GDN) | 16 k-heads, 48 v-heads, 128-dim, conv 4 | 16 k-heads, **32** v-heads, 128-dim, conv 4 |
| Full attention | 24 heads, 2 KV heads, head_dim 256, partial rotary 0.25, output gate, **plus a sparse indexer (QSA)** | 16 heads, 2 KV heads, head_dim 256, partial rotary 0.25, output gate, **dense** |
| MoE | 512 experts, top-10, intermediate 640, shared 640 | 256 experts, top-8, intermediate 512, shared 512 |
| Residual | hyper-connections (`hc` 4 streams) | plain residual |
| PLE / n-gram table | yes (28.8 GB shard 2) | none |
| MTP | 1 draft layer | 1 draft layer |
| Vocabulary | 248,320 | 248,320, the same tokenizer |
| Context | 262K | 262K native |

Everything Qwen3.6 has, the engine has a kernel for; what it lacks, the engine treats as optional (PLE) or
needs a bypass for (hyper-connections, the QSA indexer). Maya's port study (`docs/GLM5-FLASH.md` in that
repo) crossed a larger *kernel* gap for GLM-5.3-Flash in about a week: study 2026-09-29, verified
streaming runner 2026-10-01, v1.0.0 on 2026-10-07. The Qwen3.6 kernel gap is smaller (no MLA, no sparse
indexer to write, no manifold hyper-connections, a SiLU-for-sigmoid change in the GDN gate). The
*plumbing* gap is not small: `docs/qwen36-port/geometry-inventory.md` (154 rows, 2026-10-11) shows the
hc = 4 gated residual and the QSA indexer assumed structurally in every block, the output head, the
verify window, the MTP drafter and prefill; prefill refusing any geometry but 2560/4/320/top-10; native
packs decoding only through the verify window; and `ModelGeometry` reading only the expert counts and
rope keys from the file. Expect Maya-scale work, with the risk in plumbing rather than kernels.

**The deliverable:** this tree's engine running Qwen3.6-35B-A3B natively, by Maya's method, behind the
existing front end (`frontend/`: OpenAI API, tokenizer, chat template, tool-call parser, vision), which is
model-agnostic and stays. The payoff is what llama.cpp's `--n-cpu-moe` cannot do: hot experts cached in
VRAM by usage across all layers, the CPU computing the misses concurrently with the GPU, instead of whole
layers' experts on the CPU serially. On a 6 GB card that is the difference between a cache holding the
hottest 7% of experts and a layer cut holding none.

Hypotheses:

- **H1.** Under llama.cpp `--n-cpu-moe N` the GPU is idle most of each decode step, and the idle share
  grows with N, so it is worst on the smallest cards. (The baseline sweep measures it.)
- **H2.** The idle is CPU expert matmul, not PCIe transfer.
- **H3.** 256K context fits at Q4 on every card of 11 GB or more; 128K fits on 8 GB; 6 GB needs the KV
  streamed to RAM or the Q3 arm, and the engine must say which it chose.
- **H4.** Qwen3.6's routing is skewed enough that a VRAM expert cache serves well above its slot fraction
  of routes at every budget from 6 GB up. The replay measures the whole curve before any CUDA.
- **H5.** The ported engine's decode scales as `1 / max(cpu, gpu)` rather than `1 / (cpu + gpu)`, and beats
  the llama.cpp baseline at every VRAM budget, by the most at the small end.
- **H6.** With 12 GiB of RAM beside the 1080 Ti, Q4 reaches the disk tier and the profile-ranked VRAM
  and RAM tiers keep disk reads under one expert per token on a warm conversation; IQ3 never reaches the
  disk. The replay with three tiers predicts this; the engine's tier counters confirm it.

H4 and H5 are not a go/no-go for building the engine; the port is the deliverable regardless. They decide
how the README describes the result and where the expert-cache tuning effort goes.

**Measured on 2026-10-11 (Phase 0; details in `~/Code/regina/analysis/qwen36-35b-single-gpu-ladder/`):**

| | Q4_K_XL at 11 GB | IQ3_XXS at 11 GB |
| --- | --- | --- |
| llama.cpp, 256K, fit edge (`--n-cpu-moe` 32 / 18) | 8.8 tok/s decode, 129 prefill, GPU busy 15% | 24.6 tok/s decode, 434 prefill, GPU busy 41% |
| Route trace, top 40% of experts per layer | 76% of routes (215K routed tokens) | |
| Replay, LFU with 3 spares per layer, VRAM slots per layer | 76 slots: 82.6% hit, 42.5 RAM and 13.2 disk reads per token | 156 slots: 96.0% hit, 12.7 RAM and 0 disk reads |
| Replay, Belady ceiling | 91.0% hit, 6.6 disk reads | 98.2% hit |
| Predicted decode, overlap / serial (gpu 17 ms, CPU lane 0.15 ms per expert, disk 1.3 ms; parameters, not measurements) | 42.5 / 24.7 tok/s | 58.8 / 52.9 tok/s |

Readings: H1 holds (the GPU idles 85% of each Q4 step, 59% at IQ3). H4 holds (83% hit from 30% of the
experts). With 12 GiB of RAM the Q4 arm is disk-limited (13 disk reads per token, about 17 ms, as long
as the whole GPU step), so on this rung IQ3 is the quant that reaches the GPU ceiling and Q4 needs either
16 GiB of RAM or 12 GB of VRAM. Prefill hit rates are linear in slots (a prompt touches nearly every
expert per layer per ubatch), so prompts stream experts over PCIe, as Strata already does. The overlap
column saturates at `1000 / gpu_ms`; the GPU step itself is the next thing to measure properly (nvprof,
in the sweep).

## 1. The card ladder

Real cards in the homelab cover the range, with Pascal at three sizes:

| Budget | Card | Where | Arch | Role |
| --- | --- | --- | --- | --- |
| 6 GB | GTX 1060 | g7 (192.168.50.185), laptop | sm_61 | the floor; regina's `g7/README.md` has llama.cpp numbers here already |
| 8 GB | GTX 1070 Ti | host .4 (`llama-qwen36-35b-mtp.service`, live) | sm_61 | second Pascal point |
| 11 GB | GTX 1080 Ti | .57 at PCI `2a:00`, in **VM 211 `gpu-qwen36`** at 192.168.50.43 (cloned from 210 on 2026-10-11: 8 vCPU, 12 GiB, 100 GB on `local-lvm`, Debian 13, driver 550, CUDA 12.4, gcc-13) | sm_61 | the primary measurement card, and the development card until a pod is rented |
| 24 GB, rented | RTX 3090 or A6000 on RunPod | on demand; this fork was already checked on a RunPod 3090 container (README) | sm_86 | the fast-compile development card for Phase 2 steps 1 to 5 and the 24 and 32 GB rungs; rented when step 2 starts, released between sessions. The user has approved spending on this |
| 24 GB | RTX 4090 | VM 221 (192.168.50.151) on pve2 | sm_89 | **in use by another agent's benchmark**; the 24 GB rung waits for it |
| 12, 16, 32 GB | RTX A6000 (48 GB) **capped** | VM 220 (192.168.50.8) on .57 | sm_86 | **in use by another agent**; the emulated rungs wait for it |

The cap is Maya's `STRATA_GLM_VRAM_GB` idea: the engine limits its own total VRAM so the expert pool gets
what a card of that size would leave it. It is built in Phase 2 step 2; the 1080 Ti emulates the 6 and 8 GB
rungs with it (the real 1060 and 1070 Ti then check launch failures, FP16 paths and PCIe), and the A6000
emulates 12, 16 and 32 GB whenever it is free.

Consequence of the 1080 Ti being the only local card: **the Pascal build comes first, not last.** Every
parity harness and the streaming runner are brought up under `STRATA_EXPERIMENTAL_SM60=ON` on sm_61 from
the start, so Pascal surprises surface in the first week. The rented sm_86 pod runs the same harnesses in
parallel for fast iteration; a kernel passes when it passes on both.

Constraints across the ladder:

| | |
| --- | --- |
| CUDA | one toolkit that compiles sm_61 and sm_120 both: **CUDA 12.8** (12.x is the last with Pascal; 13 dropped it). Upstream saw RTX 50 crashes with 12.8-built engines (issues #220, #224), so RTX 50 is "built, not measured" until a card exists here. Fat binary `CUDA_ARCHS=61;70;75;86;89;120` |
| Pascal specifics | dp4a exists on sm_61 (not sm_60, which gets the shim); no tensor cores; no `__nanosleep`; FP16 math at 1/64 rate on GP102/GP104/GP106, so every kernel path on these cards must be FP32 or dp4a int8. Strata refuses sm < 75 unless `STRATA_EXPERIMENTAL_SM60=ON`; that flag becomes the default build (Phase 2, step 6). Nobody has run this engine on Pascal; Maya requires Volta or newer |
| Build verification | `cuobjdump --list-elf \| grep sm_` after every build; two past incidents here were wrong-arch binaries |
| Profilers | current Nsight Systems: Turing and newer only. Nsight Compute: no Pascal since 2019.5. On Pascal use the `nsys` bundled with CUDA 12.x (2023.x) or `nvprof`; on Turing and newer current `nsys`. Same analysis script either way |
| .57 host | Ryzen 7 1700 (Zen 1, AVX2, no AVX-512), 62 GiB usable DDR4-2133. Checked 2026-10-11: VM 220 shrunk to 36 GiB by another agent, host 18 GiB available before VM 211, about 6 GiB after. The 1080 Ti's link is **Gen3 x8** (`LnkSta: 8GT/s, Width x8`), so the June Gen1 problem is gone. VM 220 shares the memory bandwidth and the 8 physical cores (VM 220 has 12 vCPU, VM 211 has 8: oversubscribed when both are busy), so each run's metadata records VM 220's load and measurement windows prefer its idle hours |
| Stale pointer | `docs/NEW-GPU-VM.md` says "both GTX 1080 Ti"; slot 29:00 now holds the A6000 |

## 2. Models, the memory hierarchy, and the budget table the engine must reproduce at start

**The hierarchy on the primary rung is three tiers, by the user's decision:** 11 GB of VRAM in the 1080 Ti,
about 12 GiB of VM RAM (roughly 9 to 10 GB usable for a pinned expert tier after the OS, the process and
the KV), then the VM's NVMe disk for the rest. Strata already has the pieces: the VRAM expert cache, the
pinned RAM tier, and the low-RAM modes (`--mmap-experts` serving misses from the model file through the
page cache; `--resident-experts` copying exactly the non-VRAM experts into RAM when they fit). At Q4 they
do not all fit in 12 GiB, so the mapped mode and the disk are in play; at IQ3_XXS the whole expert set fits
VRAM plus RAM. Every per-budget number below is therefore a pair: where the experts live, and how many
reads per token reach the disk.

| Arm | File | Size | Where it already is |
| --- | --- | --- | --- |
| **Q4, default** | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (unsloth, repo commit `a483e9e`) | ~21 GB | not on hand; download once on the Mac, SHA-256 against the repo, copy over LAN |
| Q4 + MTP | `Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf` | 22.66 GB | `.4:/mnt/models/qwen3.6/MTP/`, Mac `~/models/qwen3.6/MTP/` |
| Q3 | `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` | 13.21 GB | `.4`, `.151:/models`, g7 `/home/rk/models` |

K-quants for the default because most experts run on the CPU on most of the ladder, and K-quant dot
products are the fast AVX2 path. The MTP file is required for the engine: Strata's draft protocol is built
around the model's own MTP layer, and the draft head's tensors ride in that GGUF. So the shipped default is
the Q4 MTP file (or Q4_K_XL plus a separately packed head, if unsloth publishes the head on its own);
Phase 0 settles which.

Fixed VRAM before any expert slot (estimates from regina's measurements, replaced by Phase 0 numbers):

| Item | Q4 | IQ3_XXS |
| --- | --- | --- |
| Non-expert weights on the GPU (token embedding stays in RAM) | ~3.1 GB | ~1.9 GB |
| CUDA context and hidden overhead | ~0.5 GB | ~0.5 GB |
| KV, q4_0, 10 attention layers: 131K / 262K | 0.72 / 1.44 GB | 0.72 / 1.44 GB |
| Compute and prompt buffers at 262K | ~0.8 GB | ~0.8 GB |
| **Fixed total at 262K** | **~5.8 GB** | **~4.6 GB** |
| Expert weight, total / per expert | ~17.4 GB / ~1.7 MiB | ~10.5 GB / ~1.0 MiB |

What each budget leaves for experts at Q4, 262K (the llama.cpp layer cut on the right is the baseline's
equivalent, for the comparison only):

| Budget | Expert slots (of 10,240) | Share | `--n-cpu-moe` equivalent |
| --- | --- | --- | --- |
| 6 GB | 0 at 262K; ~700 (7%) at 131K with KV streamed; Q3 arm: ~1,300 (13%) | | 40 (all) |
| 8 GB | ~1,300 (13%) | | ~37 |
| 11 GB, 12 GiB RAM, NVMe | ~3,100 (30%) in VRAM; ~5,500 (54%) pinned in RAM; ~1,600 (16%) on disk at Q4. At IQ3: all off-VRAM experts in RAM, no disk tier | | ~32 (llama.cpp's CPU experts at Q4 exceed 12 GiB: its baseline on this rung is page-cache bound too) |
| 12 GB | ~3,700 (36%) | | ~31 |
| 16 GB | ~6,100 (60%) | | ~26 |
| 24 GB | all, with ~0.8 GB to spare | | 0 |
| 32 GB | all; the spare VRAM goes to f16 KV or a bigger prompt chunk | | 0 |

The sizing rule the engine implements (Phase 2, step 2): take the fixed total for the requested context;
if it does not fit, first stream the KV to RAM above a resident window (Strata's `--kv-resident`), then
drop the context toward 131,072, then refuse with the exact numbers and the suggestion of the Q3 file.
Never silently pick a shorter context. Then the RAM tier: free RAM minus a headroom for the OS and the KV
stream, filled with the next experts by profile rank; whatever remains stays mapped from the GGUF on disk
and is read on a miss. The start-up log states all three counts.

CPU-side ceiling for the misses: ~12 MiB of expert weight per CPU-computed layer-token at Q4. On the .57
host (DDR4-2133, ~30 GB/s) that is ~60 to 70 tok/s before compute; Zen 1's 8 cores will be the lower
bound. Prior Pascal data: GTX 1060 6 GB, IQ3_XXS, N = 32: 16 tok/s at 8K depth, 11 at 32K, 355 tok/s
prefill, all llama.cpp.

## 3. Phase 0: environment, baseline, and the replay

Human steps (Proxmox state is the user's):

1. (Decided: VM 220 stays up; host RAM already freed for a 12 GiB VM.)
2. (Done 2026-10-11.) VM 211 `gpu-qwen36`, a full clone of VM 210: **8 vCPU, 12 GiB RAM, balloon 0,
   100 GB on `local-lvm` (NVMe)**, `--hostpci0 0000:2a:00,pcie=1`. No `rombar=0`: that was needed only
   for the boot-VGA card at 29:00 (homelab ticket 002); the 1080 Ti at 2a:00 never needed it. VM 220
   keeps running. The disk is the third tier, so `fio` its read throughput and 4K random latency once,
   into `phase0-notes.md`.
3. Inside the VM: pin the kernel/driver pair, install `build-essential cmake nvtop linux-perf`, CUDA 12.8
   (the template has 12.4; 12.8 needs a 570-series driver, still a Pascal-capable branch), confirm
   `nvidia-smi`, check `LnkSta`, set `perf_event_paranoid` to 1.
4. Copy the GGUFs over LAN; `sha256sum` each.

Agent steps (fresh sonnet agent; brief = this section plus the hosts):

5. Build bare `llama-server` at the submodule's pinned commit `3cf0325`, `-DGGML_CUDA=ON`, arches
   `61;86;89`, static; verify with `cuobjdump`. Also the latest release (v0.6.0) for its MTP arm. One
   build per rung, each verified for its own arch. This binary is the baseline and the oracle, nothing
   else.
6. Profiler check on the 1080 Ti: CUDA 12.x `nsys status -e`, a 10-second `nsys profile -t cuda` of
   `llama-bench`, `nvprof --print-gpu-trace` as fallback. Record which works.
7. Smoke test at 256K on the 1080 Ti with `llama-server`: `-ngl 99 --n-cpu-moe 32 -fa on -c 262144
   -ctk q4_0 -ctv q4_0 -b 2048 -ub 2048 --parallel 1 --cache-ram 0 -lv 4 -t 8`. Save the `-lv 4` buffer
   lines; they replace the estimates in Section 2.

**The baseline sweep (sonnet to drive, haiku to monitor).** Regina's harness (`scripts/sweep_ncpumoe.py`,
`plot_ncpumoe_sweep.py`): the N sweep with VRAM (`nvidia-smi` and `-lv 4`), host RAM (`RssFile`), prefill
and decode from `timings`, one real launch per point. One env spec per card; the A6000 budgets via
`CUDA_VISIBLE_DEVICES` plus a side process holding the surplus VRAM. Regina's hygiene: 30 s settle, drop
caches, warmup discarded, `--cache-ram 0`, `--parallel 1`. Workload per point: the Opus system prompt
(~27K tokens) + "implement Minesweeper", `n_predict 800`, plus one 3K-prefill / 1200-decode request.
Reasoning off, `temp 0`, `seed 42`.

Instruments, logged per run (the same three serve the engine later):

- **Level 0, always:** `nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw
  --format=csv -lms 100`. `utilization.gpu` is the GPU-busy share at 100 ms resolution. nvtop in a tmux
  pane is for the human, not the record.
- **Level 1, per point:** a decode-phase kernel timeline (`nsys` or `nvprof`). A script computes busy
  fraction, gap histogram, memcpy vs kernel share (the H2 test), per-token period, profiler overhead.
- **Level 2:** `perf stat` on the server PID, so gaps match CPU time in ggml matmul threads.

| # | Where | Arm | N values | ctx | Question |
| --- | --- | --- | --- | --- | --- |
| B1 | 1080 Ti | Q4 | fit edge, +2, +4, +8, 40 | 262144 | H1, H3 on the primary card; with 12 GiB of RAM the CPU experts page from disk, so `RssFile` and disk reads are recorded per point |
| B1b | 1060, 1070 Ti, 4090, A6000 at 12/16/32 | Q4 | the fit edge for that budget, and 40 | 262144, or 131072 where 262K does not fit | the baseline row of the ladder table |
| B2 | 1080 Ti | Q4 | fixed N | 262144 | depth 8K, 32K, 128K |
| B3 | 1080 Ti, 1060 | Q3 | fit edge, +4, +8, 40 | 262144 | K-quant vs i-quant CPU cost; the 6 GB default |
| B4 | 1080 Ti | Q4 + MTP | fixed N | 262144 | multi-token verify on CPU experts (llama.cpp's MTP) |

B1, B1b and B3 are the baseline. B2 and B4 run when a card is idle. File-based drivers under `tmux`,
`DRIVER_DONE` markers. B1 takes 2 to 4 hours at 256K context.

**E0, the route trace and cache replay (sonnet agent; any GPU host, even the Mac).** From Maya; it
answers H4 for the whole ladder before any CUDA and it produces the expert profile the engine loads:

8. Record routes: a small patch to llama.cpp's `examples/eval-callback` writing one line per token and MoE
   layer with the 8 selected expert ids (`ffn_moe_topk`). Corpus: regina's coding tasks, the Opus system
   prompt, a few long chats; ~200K routed tokens.
9. Replay: adapt Maya's `tools/glm_tier_replay.py` (MIT; models LFU residents + spares + the CPU-lane
   pick) to 40 layers × 256 experts **and three tiers** (VRAM slots, RAM slots, disk), and sweep the VRAM
   slot count from 0 to 10,240 landing on each budget from Section 2, at Q4 and IQ3, with the RAM tier
   fixed at what 12 GiB leaves. Policies: LFU, LRU, Belady ceiling.
10. Output: `phase0-replay.md` with VRAM hits, RAM hits and disk reads per token **vs slots**, the seven
    budgets marked, and the predicted engine decode ceiling per budget against the baseline, using the
    measured disk latency from step 2 for the disk term. The trace also seeds `expert-profile.bin`.

Exit criteria: the baseline ladder row with the 1080 Ti's full N sweep; a working profiler on Pascal; the
hit-rate curve with the budgets marked; Section 2's estimates replaced by measured buffer sizes.

## 4. Phase 1: images and the baseline write-up

Under `~/Code/regina/analysis/qwen36-35b-single-gpu-ladder/`:

1. `gpu-busy-vs-N.png`: Level-1 busy fraction and Level-0 utilization vs N on the 1080 Ti, both arms. H1.
2. `timeline-N<k>.png`: a 200 ms strip of one decode step, kernels vs memcpys vs gaps. H2.
3. `ladder-baseline.png`: decode and prefill vs VRAM budget (6, 8, 11, 12, 16, 24, 32), llama.cpp at the
   fit edge. The engine's points go on the same axes in Phase 2 step 5. The headline image.
4. `vram-vs-N.png`: VRAM vs N with the budget lines and the 262K/131K KV points.
5. `dmon-strip-N<k>.png`: the raw 100 ms trace over prefill then decode.
6. `replay-hitrate-vs-slots.png`: hit rate vs slots per policy, budgets marked. H4.

Analysis reads the PNGs and CSVs. The write-up answers with numbers: busy fraction and its slope in N;
memcpy vs gap; the default quant per budget; the baseline ladder row; the simulated hit rate per budget
and the predicted engine gain, largest-to-smallest card. A fresh agent reviews it, given the data but not
the conclusion. This write-up is what the engine must beat; it does not decide whether the engine is built.

## 5. Phase 2: the engine port, Maya's recipe on a smaller gap

Where: VM 211, the 1080 Ti, under the Pascal build from the first step (see Section 1). The A6000 and
the 4090 are taken; if either frees up it becomes the fast-compile card, but nothing waits for it. Each step has a parity harness in Strata's negative-test discipline (every wrong reading must
differ materially before the right one is judged), as Maya's section 6 did. Steps 1 to 4 need no GPU time
beyond seconds of parity runs, so they proceed while the Phase 0 sweeps hold the cards.

1. **Oracle and fixture (sonnet, no GPU).** Found on 2026-10-11 (`docs/qwen36-port/upstream-tooling.md`):
   upstream never published its Python oracles (`ref/` holds only `load.py`; `gdn.py`, `moe.py`,
   `qsa.py`, `docs/semantics.md` are cited but absent), most parity tests carry a host C++ reference
   inside the test file, two embed generated `.inc` vectors, and the ggml-oracle tests link a llama.cpp
   build through `STRATA_ORACLE_SOURCE_DIR`. So: `tools/` and `tests/` restored from the import commit `12a7825`
   (done, commit c00942d: development tooling only, the binary still needs no Python), then write `ref/qwen36.py` new on `ref/load.py`, following the submodule's
   `qwen35moe.cpp` graph with dumps named by its `cb()` calls; `tools/qwen36_synth_gguf.py` from
   `test_iq_pack.py`'s `write_gguf()`; invariant tests (top-8 renormalization, GDN state shapes, the
   output gate). Cross-check with the submodule's `llama-eval-callback` on the same synthetic GGUF.
   One model difference already known: Flash-Next's GDN output norm gates with sigmoid(z), `qwen35moe`
   with SiLU(z); `gdn.cu` and `gdn_parity` encode the sigmoid reading, so that is the first negative
   test. `make_profile.py` hardcodes 48 layers and 512 experts; the MTP tools hardcode Flash-Next tensor
   names; `iq_pack.py`'s `FORM` table has no `qwen35moe` entry. Dependencies: numpy, regex, gguf-py; no
   torch, no GPU.
2. **Geometry, metadata, sizing (default model).** Per the inventory: `ModelGeometry` is a set of
   Flash-Next defaults; only `qwen4exp.expert_count`, `expert_used_count` and the rope keys are read, and
   `check_architecture` rejects anything but `qwen4exp`. So: read every field of `ModelGeometry` from
   the `qwen35moe.*` keys (hidden, layers, head counts, `ssm.*`, `n_ff`, interval); add `hc = 1` as "no
   gated residual" with a plain-residual path and the `attn_norm` / `post_attention_norm` /
   `output_norm` tensors the engine never references today; fix the one leak in the native expert path
   (`expert_layout.cpp:284` passes compile-time 2560/640 to `native_fmt`); make `--ple-gguf` optional at
   the command line (the kernels already allow no PLE); accept llama.cpp's fused `ffn_gate_up_exps` and
   `attn_qkv` forms or split them at pack time; the draft geometry follows the target; the draft vocab
   size comes from the pack. **The VRAM planner:** free VRAM minus the fixed set for the requested
   context; KV streaming engaged when the fixed set does not fit (`--kv-resident`); the context lowered
   toward 131,072 only after that; a refusal with numbers below it; the remainder becomes expert slots;
   then the RAM tier from free RAM minus headroom, the rest mapped from disk (the existing
   `--mmap-experts` source, with per-layer prefetch hints as Maya's section 7 added); `--vram-cap` and
   `--ram-cap` for emulation. This step is what makes "6 GB to 32 GB" one binary, and it is where the
   engine's tier counters (VRAM, RAM, disk reads per token) are exposed in the log and on `/health`.
3. **Kernels and the block (default model).** In order of risk, each with a parity harness:
   - the GDN mixer: head pairing `h % h_k` already handles 32 v-heads against 16 k-heads; the gate
     changes from sigmoid(z) to SiLU(z) in four kernels (`gdn.cu`, `fused_gdn.cu`,
     `native_gdn_preprocess.cu`, the prefill GDN kernels), with the sigmoid reading as the negative test;
   - the full-attention layer as *dense* attention: the decode kernel reads KV through an index list and
     does not care how the list was built, so a "select every cell" producer replaces the indexer; the
     kernels fixed at 12 query heads per KV head (Qwen3.6 has 8) and the short-context kernel fixed at
     24 heads become parameters; `native_flash_attn.cu` (specialized from llama.cpp's `fattn-vec`) is
     the fallback for prompts;
   - the router at top-8 over 256 (generic already; parity at the selection boundary) and the shared
     expert at 512 (its Q8_K rule needs a multiple of 256, which 512 satisfies);
   - the block without hyper-connections: `layer.cpp` *and* `verify.cpp`, because native packs decode
     only through the verify window, and that window requires the fused hc kernels (fixed at
     2560/4/320), top-10 and the native indexer today;
   - prefill: `Prefill::init` refuses anything but 2560/4/320/10 and its GDN, split-q, route and MoE
     kernels carry literals (48 v-heads, 24 heads, 1280, 640, top-10); parameterize or fork them.
   Then the layer composition test: one GDN+MoE and one attention+MoE layer against the oracle's dumps.
   The canonical Q2_0 CPU expert kernel bakes Flash-Next's shapes; Qwen3.6 rides the native pack path
   (per-layer runtime geometry, ggml dot products, Unsloth's Q4_K/Q5_K types already included), as the
   Coder with its 256 experts per layer does.
4. **Pack and runner.** `iq_pack.py` taught `qwen35moe` and its BF16-preserve list; the MTP head: the
   engine's drafter reads its own Q8_0 directory with vLLM-style names, canonical Q2_0 experts at
   2560/640 and the default geometry, and Qwen3.6's single `nextn.eh_proj` over the concatenated
   embedding and hidden state does not match the two separate projections it expects, so the drafter
   needs a Qwen3.6 pack format and a small graph change (Maya's `mtp_gguf.py` is the model);
   `make_profile.py` (hardcoded 48 × 512 today) from the E0 trace;
   the streaming runner held to the oracle at every position of a short sequence; then the real Q4 file
   end to end, greedy, compared with llama.cpp's output distribution (not token-for-token; the README's
   known-problems entry says why).
5. **Serve and measure on the 1080 Ti at 11 GB, then capped to 8 and 6; the 12, 16, 24 and 32 rungs when a card frees up.** CPU lane and PCIe
   share calibrated as Maya's `--calibrate` does, against this engine's own session; the Phase 0
   harness and instruments; the engine's points onto `ladder-baseline.png`. The H5 measurement. The
   front end's `make test-chat-parser MODEL=<gguf>` against Qwen3.6's template (the homelab services
   use a patched `qwen3.6-enhanced.jinja`); the Qwen3-Coder any-argument-order patch checked for
   misfires on this model's tool-call grammar.
6. **The other Pascal cards and the fat binary.** The sm_61 build has been the working build since step
   1; this step makes `STRATA_EXPERIMENTAL_SM60=ON` the default for the fat binary
   (`CUDA_ARCHS=61;70;75;86;89;120`, CUDA 12.8) and runs the 1070 Ti and the 1060 against B1b. The
   sm_75+ paths (tensor-core prompt attention, thread-block clusters) stay compiled in for the cards that
   have them and are checked on the 4090 or A6000 when one is free.
7. **README/UPSTREAM.** The supported model and its checksums, the ladder table with the engine against
   the baseline, the sizing rule in words, the Pascal caveats (CUDA 12.x, FP32 paths, no Nsight Compute).

**The base is this repository's `src/` and `include/`: Strata at `6f32ec0`, engine 0.1.39.** Maya's engine
is not merged, cherry-picked or vendored: it forked from Strata around engine 0.1.27 and has rewritten the
expert tiers, the mixers and the server since, so its C++ does not apply to this tree without a port of its
own. Taken from Maya (MIT, by reference to a pinned commit): the method and the Python-side tooling; the
list of places Strata's geometry was hardcoded, each re-found and fixed here; the route-log format,
`glm_tier_replay.py` and `glm_expert_prior.py` adapted as standalone scripts; the VRAM cap and the
CPU-lane calibration as *designs*, implemented against this engine. Left out: the Python installer, the
dashboard, the Anthropic API, all multi-GPU work. One binary, no Python at run time, no network, one GPU.

## 6. Acceptance

Per budget, at the default quant, the engine must meet:

| Criterion | Threshold | Basis |
| --- | --- | --- |
| A1 context | 262,144 on 11 GB and up; 131,072 on 8 GB; on 6 GB whichever the sizing rule picks, stated at start | the stated goal and floor |
| A2 decode | beats the baseline's fit-edge decode on the same rung; absolute floors ≥ 15 tok/s at 8K depth and ≥ 10 at 32K on 11 GB, ≥ 10 / 7 on 6 GB, ≥ 60 / 45 on 24 GB | the 1060 did 16/11 at IQ3 under llama.cpp; the 4090 did 100+ |
| A3 prefill | ≥ 300 tok/s on Pascal, ≥ 2,000 on Ada | the 1060 did 355 |
| A4 stability | 7 consecutive requests, no OOM, ≥ 100 MiB VRAM headroom | g7's "+78 MiB per request" |
| A5 serving | chat, streaming, reasoning, tool calls and images through the front end, pi writing and running a program through it, on the 1060, the 1080 Ti and the 4090 with no flags but the model | the README's existing status bar |

If A2's "beats the baseline" fails on a rung after the cache tuning of step 5, the README reports the
rung's numbers against the baseline as they are. The engine still ships for that rung; the comparison is
the honest part of the result, not a gate.

## 7. Agents: who does what, and how cheaply

Rules:

- `ark` is human-only. If a phase should run under ark, the plan says so and the user starts it.
- One fresh agent per phase boundary, briefed with this file, the phase section, the hosts, the run
  directory, and the previous phase's write-up (not its transcript).
- Fork (same context) only to continue an in-progress phase after a wakeup.
- Nothing an agent runs changes Proxmox state. Agents do not touch the live `.4` service except by
  starting a second server on another port with the production unit stopped, and restoring it.
- Unattended sweeps: a file-based driver under tmux, polled on a `/loop` or wakeup every 15 to 30
  minutes, `DRIVER_DONE` markers.

Tiering:

| Work | Model | Why |
| --- | --- | --- |
| `-lv 4` logs to CSV; `dmon.csv` parsing; nsys/nvprof export to busy-fraction and gap histograms; checksums; marker checks; replay runs at given slot counts; B1b single points on a card with a ready script | **haiku** | mechanical, re-runnable |
| Env specs, drivers, plots; llama.cpp builds per arch; the eval-callback route-dump patch; the replay adaptation; the oracle and synthetic fixture (Phase 2 step 1) | **sonnet** | code with a known shape and an example to copy |
| The Phase 1 write-up; geometry, the VRAM planner, kernels, pack, runner, the Pascal build (Phase 2 steps 2 to 7); diff review before the user ships | **default (Fable)** | judgment and engine code |
| Independent review of the Phase 1 write-up and of each parity harness before it is trusted | fresh **default** agent, given data but not the conclusion | an agent handed a hypothesis tends to confirm it |

Safe parallelism: the 1080 Ti sweep, the B1b points on the other cards, E0 on the Mac, and Phase 2 steps
1 to 4 on the A6000 are independent of each other. If the user wants a multi-agent fan-out for Phase 2,
they say "use a workflow".

## 8. Order of work and checkpoints

1. User: VM 220 decision; the 1080 Ti VM; GGUFs copied to every rung.
2. Agent (sonnet): `llama-server` builds per arch, profiler check, smoke test → `phase0-notes.md`.
   **Checkpoint: the first `-lv 4` table on the 1080 Ti; Section 2's estimates replaced.**
3. Agent (sonnet, then haiku): E0 → `phase0-replay.md` and the expert profile. **Checkpoint: the hit-rate
   curve with the seven budgets marked.** In parallel, Phase 2 step 1 on the Mac.
4. Agents: B1 on the 1080 Ti; B1b on the 1060, 1070 Ti, 4090 and capped A6000; B3. **Checkpoint: the
   baseline ladder row.** In parallel, Phase 2 steps 2 to 4 on the A6000.
5. Agent (default): plots and the baseline write-up; separate reviewer. **Checkpoint: the number to beat
   on every rung.**
6. Agent (default): Phase 2 step 5 across the capped budgets. **Checkpoint: the engine's points on the
   ladder plot against the baseline, and pi served through the front end on the 4090.**
7. Agent (default): step 6 on the three Pascal cards. **Checkpoint: `unbiased-strata --model <gguf>`
   serving pi on the 1060 and the 1080 Ti with no flags but the model; A1 to A5 on every rung.**
8. Step 7, the README. B2 and B4 whenever a card is idle.

Measurements land in `~/Code/regina/analysis/qwen36-35b-single-gpu-ladder/`, infra notes in
`~/homelab/investigations/<date>-vm-1080ti-qwen36.md` with a LOG.md entry, the model-support outcome and
the ladder table in this repo's README, and the port study in `docs/QWEN36-35B-A3B.md`. Tickets: one
`homelab` for the VM, one `regina` for the harness, one `UnbiasedStrata`-prefixed ticket per Phase 2
step, filed when the work starts.
