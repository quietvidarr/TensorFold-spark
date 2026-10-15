# Changes in this fork

Fork of [ashhart/TensorFold](https://github.com/ashhart/TensorFold) `609ca419` (0.6.5), Apache-2.0.
Modified by quietvidarr, 2026. EXL3 Qwen3.8 Flash Next (`qwen4_exp`) and Qwen3.6-35B-A3B (`qwen3_5_moe`) on one DGX Spark (GB10).
Vision prefix cache from souf's recipe (sxuff/qwen38-flash-next-dgx-spark).
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

## Files changed
`*` = added, others modified.

- `src/tensorfold/cuda/exl3/`: experts.cpp, experts.cu, experts.py, experts_cb0.cu, experts_cb1.cu, experts_cb2.cu, experts_grouped.cuh, experts_prefill.cuh*, linear.cpp, linear.cu, linear.py, prefill.py, prompt_moe.cpp*, prompt_moe.cu*, prompt_moe.cuh*
- `src/tensorfold/cuda/`: keep_rank.py*, markers.py, moe.py, prompt_store.py*, sampling.py
- `src/tensorfold/cuda/kernels/`: attention.py, gdn.py, gdn_prefill.cu
- `src/tensorfold/families/qwen3_5/cuda/`: b16.py, exl3_load.py, forward.py, glue.py, prefill.py, weights.py
- `src/tensorfold/families/qwen3_5_moe/`: __init__.py
- `src/tensorfold/families/qwen3_5_moe/cuda/`: __init__.py, decode.py, disk_points.py*, engine.py, exl3.py*, graphs.py, mtp.py, multi.py, weights.py
- `src/tensorfold/families/qwen4_exp/cuda/`: __init__.py, decode.py, disk_points.py*, engine.py, exl3.py, exl3_mm.py, exl3_pack.py, forward.py, gdn_io.cpp, gdn_io.cu, gdn_io.py, glue.py, graphs.py, hc_fused.py, hc_gemv.cu*, multi.py, multi_fill.py, multi_solo.py, ngram_read.cpp*, prefixes.py, state.py, vision_prefix.py*
- `tests/cuda/`: test_exl3_prompt_experts.py*, test_exl3_prompt_strips.py*, test_group_kernel_smem.py, test_prefill_p3.py*, test_prompt_store.py*, test_qwen36_exl3_moe.py*
- `tests/`: test_exl3_ngram_pread.py*, test_flash_capacity_dispatch.py, test_flashnext_round_plan.py, test_flashnext_vision_prefix.py*, test_qwen36_kept_ends.py*, test_qwen36_prompt_store.py*
