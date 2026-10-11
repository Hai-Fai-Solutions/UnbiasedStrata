# The Pascal (sm_61) build of the engine on the GTX 1080 Ti

Phase 2 step 2, the one item that needed the GPU (2026-10-11, branch `qwen36-step2`, VM 211: GTX 1080 Ti sm_61,
8 vCPU of a Ryzen 7 1700, 12 GiB, CUDA 12.4, Debian's gcc 14.2 as the default compiler and gcc-13 / g++-13 as the
host compiler for nvcc). The configure line is the one in `step2-notes.md` section 4.

## 1. Where the tree uses what Pascal lacks (read before the build)

Pascal GP102 is compute capability 6.1: `__dp4a` is native; no tensor cores (`mma.sync`, `ldmatrix`, tf32), no
`cp.async` (sm_80), no `__nanosleep` (sm_70), no thread-block clusters (sm_90), no `__match_any_sync` (sm_70),
no `__reduce_*_sync` (sm_80), fp16 arithmetic at 1/64 rate. The tree was searched for every one of those before
the build (`grep -rl` over `src/` and `include/`, `.cu` / `.cuh` / `.hpp` / `.cpp`):

| Feature | Files | Guard already in the file |
| --- | --- | --- |
| `mma.sync` f16 (m16n8k16 sm_80, m16n8k8 sm_75), `cp.async` | `src/kernels/cuda/qsa_prompt_attn.cu` | `STRATA_PA_SM80` (`__CUDA_ARCH__ >= 800`); `mma16816` is `__trap()` below sm_75 and `qsa_prompt_attn_batch` refuses `cc < 70` at run time, prefill takes `qsa_decode_attn_batch` instead. The Volta m8n8k4 kernel is under `STRATA_EXPERIMENTAL_SM60 && 700 <= arch < 750`, so it is not compiled for sm_61 |
| `mma.sync` tf32, `ldmatrix` | `src/kernels/cuda/native_qsa_score.cu` | `__CUDA_ARCH__ >= 800` around `load_a` / `load_b` / `mma`; `score_kernel` has an FP32-FMA path under `__CUDA_ARCH__ < 800` |
| `cvt.rna.tf32`, `mma.sync` tf32, `cp.async` | `src/kernels/cuda/qsa_select.cu` | `STRATA_SEL_SM80`; `qsa_block_scores_tc` refuses the device, the warp kernel runs |
| thread-block clusters (`cudaLaunchKernelEx`, cluster barriers) | `src/kernels/cuda/qsa_select.cu`, `src/kernels/cuda/sampler.cu` | `STRATA_SEL_CLUSTER` / `STRATA_AM_CLUSTER` (`__CUDA_ARCH__ >= 900`): a trap that `qsa_block_topk_cluster` / `sample_greedy_cluster` never launch below sm_90 |
| `__match_any_sync` | `src/kernels/cuda/qsa_select.cu` | inside the sm_80 region above (checked at the build) |
| `cp.async` | `src/kernels/cuda/fused_gr.cu` | `__CUDA_ARCH__ >= 800 && !__HIPCC__` |
| `cp.async` in the GDN recurrence | `src/prefill/kernels.cu`, `src/prefill/gdn_rec_parity.cu` | `STRATA_GDN_CP_ASYNC` (`__CUDA_ARCH__ < 800`: plain copies, and `gdn_keyhead_ok` never launches the key-head kernel there) |
| `mma.sync` / `ldmatrix` / `cp.async` in the fused prompt MoE | `src/prefill/moe_fused.cu`, `src/prefill/moe_fused_iq.cu` | `__CUDA_ARCH__ >= 800` at the two sites of each; their tests exit 77 below sm_80 |
| `__nanosleep` | `include/strata/kernels/dp4a.hpp` (`strata_spin_pause`), `src/prefill/gdn_rec_parity.cu` (`nap`) | `__CUDA_ARCH__ < 700`: the loop spins |
| `__dp4a` | `include/strata/kernels/dp4a.hpp` | `STRATA_DP4A`: the byte-loop fallback only below 6.1; sm_61 uses the instruction |
| half arithmetic (`__hfma2` and kin) | none found outside `qsa_prompt_attn.cu`'s mma path | conversions only elsewhere (the 1/64-rate risk of the design doc's section 7 item 2 is a throughput item, not a build item) |
| bf16 intrinsics (`__nv_bfloat16`, `__float2bfloat16`), fp16 atomics, `__reduce_*_sync` | none | |
| `STRATA_EXPERIMENTAL_SM60` itself | `src/core/device.cu` (the cc check), `src/kernels/cuda/qsa_decode_attn.cu` (the sm_60 decode kernel), `src/kernels/cuda/qsa_prompt_attn.cu` (the Volta kernel), `src/prefill/gemm.cu` (the FP32 GEMM default) | the flag the configure line passes |

So upstream's experimental build already guards every sm_75+ site this tree has; the question the build answers is
whether any site was missed for sm_61 specifically (the guards were written for sm_60 and sm_70, and nobody has run
the build at 61).

## 2. The build

Nothing needed a guard. The tree as it stood at `8b7525e` configured and built for sm_61 unchanged:

| Phase | Command | Result | Time (8 Zen 1 vCPU) |
| --- | --- | --- | --- |
| configure | the `cmake -S . -B build ...` line of `step2-notes.md` section 4 | `Strata: CUDA enabled, arch 61`; `NVIDIA 12.4.131 with host compiler GNU 13.3.0` | 4 s |
| `unbiased-strata` | `cmake --build build -j 8 --target unbiased-strata` | 38,221,376 B; `cuobjdump --list-elf` shows `sm_61` only | 173 s |
| every other target | `cmake --build build -j 8 -- -k` | 93 targets built, 0 errors (`logs/all.log`) | 43 s on top of the engine's libraries |

No compile error for sm_61 at all, so the table of section 1 stands: every `mma.sync` / `ldmatrix` / tf32 /
`cp.async` / cluster / `__nanosleep` / `__match_any_sync` site is behind a `__CUDA_ARCH__` guard with a fallback or
a trap that its host code never launches below the architecture. The 102 warnings of the engine build are the same
on every target: 46 `-Wdeprecated-enum-enum-conversion` from ggml's `ggml_scale_mode | ggml_scale_flag`, nvcc's
`#611-D` partially overridden `ExpertSource::prefetch` (twice each for `FileExpertSource` and `ArenaExpertSource`),
`#177-D` the unused `MAGIC` of `strata::prefill::fused`, a handful of unused variables (`w`, `width`, `n_bid`) and
unused functions in the vendored jinja. None is architecture-specific.

The VM has no `gguf` module for its system Python, but `tools/qwen36_synth_gguf.py` and `tools/iq_pack.py` run
there with the vendored `third_party/llama.cpp/gguf-py` on `PYTHONPATH` (numpy 2.2.4 is installed).

## 3. The parity tests and ctest on the 1080 Ti

`ctest --output-on-failure --timeout 900` in `build/`: 75 tests, 52 s. First run: 7 failed, 2 skipped. Two of the
failures were the branch's own tests that had never been run (both CUDA-gated in CMake, so the Mac had no binary),
fixed on the branch; the final run (`logs/ctest-final.log`) is **69 passed, 4 failed, 2 skipped**.

| Test | 1080 Ti | First line of the failure | Pascal-specific? |
| --- | --- | --- | --- |
| `qwen36_synth_fixtures` then `model_geometry_test` | failed, then Not Run; **pass after the fix** | `CMake Error at cmake/qwen36_synth_fixtures.cmake:11 (list): list index: 1 out of range (-1, 0)` | no: cmake 3.31 drops the empty element of `"toy.gguf;"` with CMP0007 unset (cmake 4 on the Mac does not). Fix: `cmake_policy(SET CMP0007 NEW)` in the script (commit `cd2a73c`). `model_geometry_test --real` also passes on the VM against both real files |
| `expert_layout_test` | 3 checks failed; **all pass after the fix** | `qwen36-q4_k_xl.txt: per-layer formats and blobs (... total 19636158464 B) FAIL` and twice `at Flash-Next's 2560 x 640 refused by its numbers: native experts: expert geometry is not whole blocks FAIL` | no: the test's Q4 total constant was 524,288 B above the pack's own header (19,635,634,176 B), and n_ff 640 is not whole 256-value blocks for Q4_K / IQ2_S rows, so `native_fmt` refuses before the blob message the test looked for (commit `5798a46`) |
| `kv_hybrid_parity` | **FAIL** at `[5/5]` (steps 1 to 4 pass, max error 7.8e-08 / 4.4e-03 vs true fp32) | `[5/5] qsa_prompt_attn mode 3: FAIL (refused the hybrid pools)` | **yes**: `qsa_prompt_attn_batch` returns false for `cc < 70` (`qsa_prompt_attn.cu:1433`), by design (the prompt path takes the decode kernel on Pascal, design doc section 7 item 3). The test counts a refusal as PASS on HIP and FAIL on CUDA; it needs the same "refused below sm_70: the old kernel runs" branch. Step 3 item, with the dense attention work |
| `platform_memory_test` | FAIL | `lock_resident: ok=0 locked=0 MiB (mlock failed (raise ulimit -l))` | no: the VM's `ulimit -l` hard limit is 8,192 KiB; every `file_cache_keeps` case before it passes |
| `ple_parity` | FAIL | `ple_parity: required PLE table is missing or incompatible: ../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf (cannot open ...)` | no: the test hardcodes a Flash-Next shard path relative to the build directory; the VM has no Flash-Next file |
| `expert_multi_test` | FAIL | `strata: this CPU cannot run the expert kernel: missing AVX512F AVX512BW AVX512VL AVX512-VNNI AVX512-VBMI.` | no: the canonical Q2_0 CPU expert kernel needs AVX-512; the Ryzen 7 1700 has AVX2 only (`STRATA_PORTABLE=ON`). Qwen3.6 rides the native pack path, which does not use this kernel |
| `prefill_fused_moe_test`, `prefill_fused_iq_test` | Skipped (exit 77) | | expected: the fused prompt MoE needs sm_80 (`CMakeLists.txt`, SKIP_RETURN_CODE 77) |

Everything else passed, including every kernel parity on the card: `iq_parity`, `iq_multi_parity`, `mmvq_multi_parity`,
`s2_gemv_*`, `s_gemv_*`, `dequant_s2_parity`, `gdn_parity`, `gdn_rec_parity`, `gr_parity`, `qsa_parity`,
`qsa_topk_parity`, `qsa_topk_active_parity`, `decode_cluster_parity`, `sampler_parity` (all three), `rope_parity`,
`quantize_act_parity`, `route_window_parity`, `router_top10_parity`, `router_dot_parity` (and `_avx`),
`elementwise_parity`, `bf16_gemv_parity`, `gemm_bf16_parity`, `kv_q4_parity`, `kv_q8_parity`, `kv_stream_parity`,
`cvec_parity`, `native_grouped_parity`, `shared_expert_parity`, `s2_expert_grouped_parity`, the nine
`native_expert_parity_*` cases (including `refuses_q6_K`), `cuda_device_selftest`, `ple_reader_selftest`, the
expert cache / source / profile tests, `vram_plan_test`, `gguf_*`. So on the 1080 Ti the numerics of every kernel
that has a parity test agree with their references; the one Pascal-specific result is a refusal, not a wrong number.

## 4. The pack and the gate

The pack is the layout half (`tools/iq_pack.py --layout-only`): one `native_experts.txt`, made on the Mac in 0.35 s
and copied to `/home/rk/packs/qwen36-iq3/` (2,669 B). The Q4_K_XL layout was made on the VM itself
(`PYTHONPATH=third_party/llama.cpp/gguf-py`, 0.6 s, 2,727 B, `/home/rk/packs/qwen36-q4/`), byte-identical in its
layer lines to `tests/data/native_experts/qwen36-q4_k_xl.txt`.

The gate, exactly the notes' command on the IQ3_XXS file (`logs/gate.log`), exit 1 after 0.13 s:

    strata generate: model geometry (/home/rk/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf, from the file's metadata):
      arch qwen35moe
      n_embd 2048
      n_layers 40
      nextn_layers 0
      qsa_interval 4
      n_qsa_layers 10
      n_gdn_layers 30
      ssm_state_size 128
      ssm_k_heads 16
      ssm_v_heads 32
      ssm_d_conv 4
      ssm_conv_channels 8192
      ssm_value_dim 4096
      n_head 16
      n_head_kv 2
      head_dim 256
      n_rot 64
      rope_sections 11,11,10,0
      rope_base 10000000.000000
      rms_eps 0.000001
      idx_q_heads 0
      idx_key_dim 0
      hc 1
      hc_lr 0
      n_expert 256
      n_expert_used 8
      n_ff 512
      n_ff_shexp 512
      has_ple no
      has_indexer no
    strata generate: layer 0's experts are IQ2_S/IQ3_XXS (ggml types 22/18), which this engine has no GPU kernels for

The Q4_K_XL file with its layout pack (`logs/gate-q4.log`): the same geometry block (nextn_layers 1), then

    strata generate: layer 0's experts are Q4_K/Q5_K (ggml types 12/13), which this engine has no GPU kernels for

**So the step 2 gate message (`qwen35moe_not_yet`, printed after the VRAM plan) is not reached by either real file.**
The refusal is loud, specific and not a crash, but it is an earlier one: `generate.cpp` checks
`native_expert_supported()` for every layer right after `expert_layout_load`, before the plan and the gate, and that
whitelist's *down* types (`STRATA_D_FMTS` in `src/kernels/cuda/iq_kernels.cu`) are IQ4_NL, IQ4_XS, Q5_0, Q5_1,
Q8_0 and type 42 only; Q4_K / Q5_K / IQ2_S / IQ3_XXS are gate/up types (`STRATA_GU_FMTS`) and Q6_K is in neither
(`native_expert_parity_refuses_q6_K` pins that). The IQ3_XXS file's down projections are IQ3_XXS on 37 layers and
IQ4_XS on 34/38/39; the Q4_K_XL's are Q5_K on 37 and Q6_K on 34/38/39. The plan's step 3 line "Unsloth's
Q4_K/Q5_K types already included" holds for gate/up only. The findings for the next steps:

1. the grouped *down* kernel needs IQ3_XXS (the default file) and Q5_K / Q6_K (the Q4 file), or step 4's pack
   requantizes the down projections to a type the kernel has (Q8_0 / Q5_1, as the Flash-Next packs do) and the
   census' 10.6 / 18.7 GB expert totals move with it;
2. the step 2 gate is demonstrable on a real file only once the whitelist passes (or the plan+gate block is moved
   ahead of it); the message itself is exercised by `model_geometry_test --real` on both files on the VM, which
   prints it in full.

## 5. The Makefile's Pascal switch

`make build CUDA_ARCHS=61 BUILD_DIR=build-make` on the VM, with no compiler override at all (`logs/make-plain.log`):
the llama.cpp patch applied (`git apply` works outside a repository, so the rsynced tree without `.git` is fine and
the patch now sits in the VM copy of the submodule), CMake chose Debian's g++ 14.2 for the host code and nvcc 12.4
took g++-13 as its host compiler by itself (`NVIDIA 12.4.131 with host compiler GNU 13.3.0`), `Strata: CUDA enabled,
arch 61` with `STRATA_EXPERIMENTAL_SM60=ON` from the Makefile's switch, 0 errors, `built: build-make/unbiased-strata`
(38,627,048 B) in 181 s from configure to link; `cuobjdump --list-elf build-make/unbiased-strata` shows `sm_61` only.
