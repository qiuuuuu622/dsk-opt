# 优化机制与实现边界

## fastokens

入口 `src/dsk_fastokens_entry.py`，在 tokenizer 构建前接入 fastokens 0.3.6；`__main__` guard 防止多进程 spawn 重入。`VLLM_USE_FASTOKENS=1`、`DYN_TOKENIZER=fastokens`、`DYN_TOKENIZER_FALLBACK=false`。首轮 backend/encode/decode 留证后移除诊断包装。脚本 83 对前 150 条的模板文本、token IDs、decode 做原生对照。Dynamo 的 MM-aware KV routing 会关闭，这是行为差异，不是仅替换 tokenizer 的无条件等价声明。

## Humming W4A8

`src/dsk_humming_worker.py` 为 40 个目标 MoE 层开启 Humming FP4 权重 + 每 token FP8 E4M3 激活，draft 仍原生 BF16。`DSK_HUMMING_A8=1`。这是激活量化，不能宣称输出 bitwise 等于原生 W4A16。使用激活 dtype 与实际 GEMM trace 核验，不能只凭环境变量判断命中。

## 输出收尾融合

旧适配：`src/vllm_dsv41_moe/{adapter,ops}.py`。完整融合：`src/dsk_humming_full_finalize.py`，worker：`src/dsk_humming_full_worker.py`。import 不改变原生函数；显式开关安装进程内 hook。

完整融合在 Humming grouped unpermute 时，将 routed sum、缩放、shared add 合并，维持各阶段 BF16 舍入顺序。通过 contextvars 限定单次 runner 调用；异步 shared experts 在 finalizer 处等待。不能提前读取有消费语义的 `SharedExperts.output`：先查看底层输出，到原边界消费一次。合并后的输出仍在原位置做 EP all-reduce。

限制为目标 E=384/H=5120、TP8/EP8 replicated rows、DP1、无 SP/PCP 扩展、无 DBO/额外变换、BF16、1..128 行。其他形状回退。需要 A8、旧 finalize、full finalize 三开关，GC 关闭。

真实模型 routed scale=1.0，所以本次真正省掉的是 shared add，不能宣称省掉一个原本不存在的 scale kernel。每卡 20 步 trace 的 BF16 add 从 860 降到 60，去掉 40 层×20 步的 800 次。脚本 93 用合成专家 GEMM 输出验证真实 runner/finalizer，8 卡共 160 eager、1600 changing-input graph checks，覆盖异步 shared、缩放和 129 行回退；不是完整 GEMM 精度验证。

## compact indexer

入口 `src/dsk_indexer_adapter.py`，内核 `src/indexer/vllm_dsv41_indexer/`。第 20 层发布候选，24/28/32/36 层消费。把候选整理为紧凑池，仅对候选 KV 位置打分并选 top512，避免扫全上下文。decode/verify 分支，候选 2048 block×8，最多 384 行；保留原生 fallback。

实际 paged cache 不是简单 contiguous：uint8 page64、内部 stride 132/132/1、page pitch116928。适配必须传真实 stride，早期连续内存假设已修正。支持 next_n1..6，实际 V2 runner 观察为 next_n1。5760 次 top-k set/graph 对照覆盖该合同。

## head↔row

入口 `src/dsk_headrow_adapter.py`，CUDA `src/headrow/vllm_dsv41_head_exchange/csrc/segment_exchange.cu`。TP8 每卡只有 8 个有效 query heads，原路径补至 64。通过两次融合 NVLink 交换将 head 分片变成 row 分片，让 FlashMLA 处理完整有效 heads，再反向交换。

覆盖 40 个目标 attention 层；draft/prefill 保持原生。最多 384 行，按块处理，动态 Q、metadata 和 caller output 指针须正确；每卡 workspace 约759 MiB。独立 CUDA `.so` 需要本机编译。已有 2400 graph checks 和 attention oracle 证据。

## GC-MoE 原生对照

`src/dsk_gc_adapter.py` / `dsk_gc_worker.py` 使用 GC 原生 `sm90_fp8_fp4_gc_moe`，不是 Humming 更名。TP replicated rows 按 EP8 切行；BF16 转 K128 block FP8，MXFP4 转原生 GC tile，后者可能重舍入。native GC 完成 routed reduction，shared 分支单独归约再融合收尾。

40 层共享 workspace，固定输出地址，每 rank 行数补至 next power of 2；padding route weight=0，避免子图缓存被动态 shape/output 指针撑爆。原生 GC role kernel 已有多角色协同，不能把本次端到端差异解释为单一 GEMM 或 green context 的纯收益。八卡正确性通过，但 GC 本次组合延迟明显落后于同 E1 环境 Humming。

## 来源

- https://github.com/qiuuuuu622/Experiment-Results
- https://github.com/qiuuuuu622/vllm_dsv41_flash_opt
- https://github.com/qiuuuuu622/tp8_MLA
- https://github.com/Bruce-Lee-LY/gc-moe （0.6.2 / 7902831）
- vLLM 0.32.0 / 5df63a7203cdb9328d824a27b682e77c7b9928be

保留代码内已有版权头、第三方 LICENSE/NOTICE；本仓库不重新授权第三方代码。具体测试时源码指纹见 handoff/source-manifest.json、evidence 下冻结记录和 EXPORT-MANIFEST.json。
