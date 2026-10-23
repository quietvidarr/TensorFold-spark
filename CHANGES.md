# Changes in this fork

Fork of [ashhart/TensorFold](https://github.com/ashhart/TensorFold) `609ca419` (0.6.5), Apache-2.0.
Modified by quietvidarr, 2026. EXL3 Qwen3.8 Flash Next (`qwen4_exp`), Qwen3.6-35B-A3B (`qwen3_5_moe`) and GLM-5.3 Flash Mosaic (`glm5_next`) on one DGX Spark (GB10).
Vision prefix cache from souf's recipe (sxuff/qwen38-flash-next-dgx-spark).
GLM patches 01-04 from sxuff's recipe (sxuff/glm53-flash-single-gb10), MIT.
One commit per change on `spark`. Full diff: `git diff 609ca419 spark`.

## Commits
- Vision prefix cache for image-history prompts
- EXL3 n-gram: parallel deduplicated reads ahead of the GPU wait
- EXL3 prompt experts on a prefill kernel
- EXL3 n-gram rows via a native reader
- Tune the EXL3 prompt expert kernel
- Flash: keep every conversation's prompt ends
- Prompt cache on disk
- Faster EXL3 expert kernels
- Prompt cache: host copies off the GPU path
- Flash: decode graphs once per slot size
- Prompt cache points at every message start, never dropped
- EXL3 prompt MoE kernel v2
- EXL3 prompt: exact deterministic sum
- Prefill prompt-path fusions
- EXL3 linear: 32-row passes decode each weight tile once
- Router: 17-32 rows in BLOCK_E 16 / BK 128 tiles
- Grouped experts: PF 3/4 instantiations
- MoE combine: optional bf16 rows
- Grouped experts: shared program body
- Grouped experts: plain launch bounds for the default
- Grouped experts: persistent loop calls the program body
- Prompt rows independent of how the prompt is split
- Flash: copy drafting gated by the MTP head's first draft
- Flash: hc_gemv for the fp16 decode matrices
- Flash: hc_gemv12 for the hyper-connection matrices
- Flash: hc12 decode windows up to 24 rows
- Flash: hc_gemv12 launch shapes (TF_FLASH_HC12_CFG)
- Flash: read ahead the next window's n-gram rows
- Flash: n-gram read-ahead off by default
- Flash: n-gram rows of every stream in one read
- Qwen3.6-35B-A3B EXL3 MoE support
- EXL3 prompt kernel: 256-column strips
- Qwen3.6: keep every conversation's prompt ends
- Qwen3.6: prompt cache on disk
- Qwen3.6: prefill steps under concurrency
- Qwen3.6: prompt cache host copies off the GPU path
- Qwen3.6: prompt cache never drops, keep_rank eviction
- Qwen3.6: prompt MoE v2 tests
- Qwen3.6: prompt sum mode tests
- Qwen3.6: one read-back per draft level
- Qwen3.6: defer the GDN replay in multi-stream decode
- Qwen3.6: one read-back per draft step
- Qwen3.6: graphs keyed on buffer rows
- Qwen3.6: MoE combine writes the bf16 rows
- Qwen3.6: conv windows read through a pointer
- Flash: prompt cache points inside one prefill pass
- Qwen3.6: graphs capture in thread_local mode
- Qwen3.6: prompt cache points inside one prefill pass
- GLM-5.3 Flash: sxuff's patches 01-04
- EXL3 prompt kernel v2: 2- and 3-bit trellises
- GLM: prompt kernel for prompt chunks, prompt cache on disk
- GLM: prompt EXL3 projections through the prompt GEMM
- GLM: 4096-row prompt chunks (TF_GLM_PREFILL_ROWS)
- Prompt store: release pinned host copies once the queue drains
- Prompt store: writer waits for an idle engine
- GLM: idle saver copies waiting kept states
- GLM: prompt plain F16 matmuls and latent absorb/expand on tensor cores
- GLM: fuse hc_post with the next hc_pre
- EXL3 prompt GEMM: K step 64
- GLM: faster KDA step kernel
- GLM: one-pass sparse latent attention for prompt chunks
- Prompt store: pool sums by one join, scans near a cap
- Prompt store: waiting points within TF_PROMPT_CACHE_DEFER_GIB
- Qwen3.6: a step's points within the defer cap

## Files changed
`*` = added, others modified.

- `src/tensorfold/cuda/exl3/`: experts.cpp, experts.cu, experts.py, experts_cb0.cu, experts_cb1.cu, experts_cb2.cu, experts_grouped.cuh, experts_prefill.cuh*, linear.cpp, linear.cu, linear.py, prefill.py, prompt_moe.cpp*, prompt_moe.cu*, prompt_moe.cuh*
- `src/tensorfold/cuda/`: geometry.py, http.py, server.py, keep_rank.py*, markers.py, memory_gate.py, moe.py, prompt_store.py*, sampling.py
- `src/tensorfold/cuda/kernels/`: attention.py, gdn.py, gdn_prefill.cu
- `src/tensorfold/`: cli.py, cli_args.py, engine/thinking_landing.py*
- `src/tensorfold/families/glm5_next/`: __init__.py
- `src/tensorfold/families/glm5_next/cuda/`: decode.py, disk_points.py*, engine.py, forward.py, glue.py, graphs.py, kda.cu, kda.py, latent.py, mtp.py, packed.py*, prefill.py*, split.py, weights.py
- `src/tensorfold/families/qwen3_5/cuda/`: b16.py, exl3_load.py, forward.py, glue.py, prefill.py, weights.py
- `src/tensorfold/families/qwen3_5_moe/`: __init__.py
- `src/tensorfold/families/qwen3_5_moe/cuda/`: __init__.py, decode.py, disk_points.py*, engine.py, exl3.py*, graphs.py, mtp.py, multi.py, weights.py
- `src/tensorfold/families/qwen4_exp/cuda/`: __init__.py, decode.py, disk_points.py*, engine.py, exl3.py, exl3_mm.py, exl3_pack.py, forward.py, gdn_io.cpp, gdn_io.cu, gdn_io.py, glue.py, graphs.py, hc_fused.py, hc_gemv.cu*, multi.py, multi_fill.py, multi_solo.py, ngram_read.cpp*, prefixes.py, state.py, vision_prefix.py*
- `tests/cuda/`: test_exl3_prompt_experts.py*, test_exl3_prompt_strips.py*, test_glm_packed_adapter_cpu.py*, test_glm_packed_loader_cpu.py*, test_glm_prompt_store_cpu.py*, test_glm_residual_fp32_cpu.py*, test_glm_split_and_policy.py, test_flashnext_prompt_cache.py, test_group_kernel_smem.py, test_prefill_p3.py*, test_prompt_store.py*, test_qwen27_prompt_end_cache.py, test_qwen36_exl3_moe.py*
- `tests/cpu/`: test_glm_mosaic_widths_cpu.py*, test_glm_tp1_contracts.py*
- `tests/fixtures/`: c_on_arithmetic.json*, v50_c_seed1.json*, v50_c_seed2.json*
- `serving/`*: sxuff's GLM serving files
- `tools/`: glm_packed_audit_cpu.py*, glm_packed_eager.py*
- `tests/`: test_answer_landing.py*, test_landing_integration.py*, test_landing_mtp.py*, test_noheader_landing.py*, test_thinking_landing.py*, test_exl3_ngram_pread.py*, test_flash_capacity_dispatch.py, test_flashnext_round_plan.py, test_flashnext_vision_prefix.py*, test_qwen36_kept_ends.py*, test_qwen36_prompt_store.py*
