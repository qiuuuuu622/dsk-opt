# 复现边界与实际步骤

这是实验归档，不是已验证的全新机器一键部署包。历史绝对路径和分阶段失败/修复脚本刻意保留；不要把 scripts 按编号全部无脑执行，其中包含历史重启和进程终止控制器。

## 环境

8×H20，driver580.95.05；vLLM0.32.0 commit5df63a7、Dynamo1.5.0、Humming0.1.17、fastokens0.3.6。

|环境|Torch|Triton|CUTLASS|NCCL|
|---|---|---|---|---|
|E0 主环境|2.13.0+cu130|3.7.1|4.8.0|2.30.7|
|E1 隔离环境|2.11.0+cu130|3.7.1|4.8.1|2.28.9|

E1 vLLM CUDA 扩展从源码重编译，不能复用 E0 Torch ABI。GC0.6.2/7902831，ptxas13.4.92。E1 的 Torch 包约束 Triton3.6，实际按用户授权升级3.7.1；其他包约束警告也未全部消除。不声称 pip check 通过。

所有 E1 正式组实际使用 E0 兼容 HTTP 前端 + E1 GPU worker；E1 原生 Rust HTTP schema 曾拒绝业务 assistant content，因此混合前端是实验配置的一部分。Python Dynamo 兼容文件曾对齐主环境，详见 scripts/63_e1_align_dynamo_python.py 与环境证据。镜像 digest 未采集，GC 隔离目录现已缺失，重建需要恢复对应补丁与依赖。

## 目录与开关

原运行目录 `/root/dsk_opt_goal_20261116`；原 E1 `/root/gc_moe_isolated_20261116`；模型 `/model`。若更换路径，须同时调整 PYTHONPATH、group/env、evidence/build/discovery 路径及 E0/E1 Python。

候选配置见 `experiments/20261116/groups/P-E1-FAST-HUM-FULL-r2/scripts/`：

```
DSK_HUMMING_A8=1
DSK_HUMMING_FINALIZE=1
DSK_HUMMING_FULL_FINALIZE=1
DSK_COMPACT_INDEXER=1
DSK_HEADROW=1
DSK_GC_MOE=0
VLLM_USE_FASTOKENS=1
DYN_TOKENIZER=fastokens
DYN_TOKENIZER_FALLBACK=false
```

worker class 为 `dsk_humming_full_worker.Worker`，启动入口为 `src/dsk_fastokens_entry.py`。PYTHONPATH 需包含 src、src/indexer、src/headrow 及对应 E1 vLLM 源码。headrow build 模块产出 `.so`，`DSV41_HEAD_EXCHANGE_LIBRARY` 指向该文件；编译方式参考其 build.py 及脚本22/44。

1. 先恢复独立 E1 环境及兼容前端，核实版本/ABI/设备拓扑；构建记录在 scripts/31..63 和 evidence，不应把失败尝试当推荐安装流程。
2. 使用脚本83核验 fastokens；脚本93核验完整融合；47/68核验 strided indexer，43核验 headrow。检查每项实际退出码和证据。
3. 从目标组的 00_runtime_env.sh、01_worker.sh、02_frontend.sh 恢复实际启动参数，使用新 group/namespace/output 目录，避免覆盖历史结果。等待模型就绪并执行合成 preflight。
4. 正式数据文件为私有 `first150.plan.json.gz`。来源 ZIP SHA256：`46601d9dda2cdb0f6f581d99e1fa00d6aa1c11719829966d689606639f8fe2df`，本公开仓库不附带对话正文。使用 scripts/09_replay150.py，参数和逐条计划读取方式见代码及目标组03_replay.py。
5. 150 条发完后等待在途请求结束，保存完整内部响应；脚本33做结构质量检查，正式统计与 profile 分开。公开 JSONL 移除了响应正文，不能再直接用于脚本33。

## 服务共同参数

TP8+EP、Engram CPU offload、max-num-seqs64、max-cudagraph-capture-size64、max-num-batched-tokens4096、gpu-memory-utilization0.92、block-size64、mm-encoder-tp-mode=data、deepseek_v41 tokenizer、multimodal、DSpark5（probabilistic/block/adaptive）。模型服务名及完整命令见每组01_worker.sh。HTTP8000、worker管理8081、KV events20080。

默认关闭所有 DSK 优化可回退；实验 handoff/native-off 保存回退脚本，但最终回退服务实启验证未执行。开关默认值、import 不修改原生函数已做审计，不能代替重新启动后的实际路径验证。
