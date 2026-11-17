# DSV4.1 优化 Goal：首轮算子准入与基线复测

- 日期：2026-11-16
- 运行结果：`completed`
- 完成范围：仅本次首轮GC/Humming算子验证、主模型精度审计、原生路径150条复测。
- 整体Goal：仍在进行；Indexer/head-row算子已验证，服务接入、优化A/B、组合和稳定性测试尚未完成。进度见[execution.json](execution.json)及[第二阶段报告](STAGE2.md)。

## 结论与核心结果

尚无优化端到端收益结论。运行时确认8rank×40主模型层×W13/W2共640项配置均为FP4 E2M1权重、BF16激活，input_quant_mode=none，即W4A16。三个关键入口SHA与此前官方0.32.0审计一致。本轮服务仅增加加载后只读配置记录，不启用任何优化。

GC隔离环境：routed/shared各9档常规输入+4组边界输入，共26个八卡case通过。Routed三次独立启动各1000步变化输入Graph、shared一次1000步通过，全部rank输出无陈旧输入，固定输入重放逐位一致。Routed相对未量化BF16的最小余弦0.977039；相对处理后参考的最大diff1.80467e-5、最大relative L2约0.006008。Shared相对处理后参考最小余弦0.99998277。两种参考精度口径不同，不能据此宣称模型质量无损。

Humming grouped输出：8卡各192项，共1536项同精度逐位一致，最大绝对差0。覆盖top3/top6、12档M=1至512、尾界无效路由、普通/scale+shared、eager/变化输入Graph；未测完整GEMM、跨层调度或整模型优化质量。

原生路径复测：150/150成功、零失败；0.45发送QPS，含排空完成QPS0.427527，output tok/s539.846，输出189408 tokens、均值1262.72，elapsed350.855秒。所有请求usage完整、finish_reason=stop；发送顺序/唯一性/分位数独立复算通过。Prefix命中率最终统计待单独提取，不用于本轮优化归因。

| 指标 | mean | P50 | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| TTFT (ms) | 3159.380 | 1141.453 | 9895.374 | 11898.125 | 13666.322 |
| TPOT (ms/token) | 8.008 | 6.875 | 10.836 | 15.340 | 31.423 |
| E2E (ms) | 12912.550 | 9202.587 | 26706.630 | 36322.180 | 63180.715 |

上一轮原生结果也未启用优化，两轮存在采样、调度与运行波动；不能把本轮变快当作优化收益。后续要求重复成对A/B。

## 复现

完整Goal顺序与准入规则见[PLAN.md](PLAN.md)。远端 `/root/dsk_opt_goal_20261116`。

1. 01尝试安装DeepGEMM2.8.2失败；02保存基线身份；03通过归档的纯Python量化函数运行GC测试，未加载DeepGEMM二进制。
2. 暂停原worker311625/frontend317210，确认8卡无计算进程。04因ptxas13.2不支持PTX9.4失败；05在独立tools/cuda134安装13.4.92；06 eager10步全部rank通过。
3. 07/08执行三轮routed Graph1000步、shared Graph1000步、routed常规/边界通过；shared测试导入失败。
4. 15/16/17补齐测试目录导入，shared9档计算检查通过但最终审计仍尝试导入DeepGEMM二进制，运行报失败；18/19/20将测试依赖审计明确记录为量化源码、binary=null，重跑shared常规/边界完整通过。未修改GC kernel及参考计算。
5. 10/11在主环境依赖的独立venv中执行Humming输出1536项对照，通过。
6. 12/13/14启动只读审计worker/frontend；21等待模型注册与8rank配置记录后自动执行09回放；24独立复算通过。

服务完整命令见[scripts/13_start_audit_worker.sh](scripts/13_start_audit_worker.sh)及[scripts/14_start_audit_frontend.sh](scripts/14_start_audit_frontend.sh)。TP8、EP、DP1、PP1、Engram CPU offload、max-num-seqs64、max-cudagraph-capture-size64、batched-tokens4096、memory-util0.92、block64、mm encoder data、deepseek_v41 tokenizer、多模态、DSpark5(probabilistic/block/adaptive)均与已验证配置一致。新增worker-cls仅记录加载后的配置。HTTP127.0.0.1:8000，KV events20080，worker TCP40405。该历史批次PID worker331824/frontend331823已停止；当前服务身份见execution.json及各group的service-pids.json。

数据仍用既有first150.plan.json.gz，原顺序、0.45QPS，331.111秒停止发送并排空；max_tokens16384、temperature0.8、top_p0.95、seed未设置。没有额外业务预热；新进程冷请求缓存，编译缓存沿用现有主环境。09补充finish_reason，可选保存响应本轮未开启。

```bash
/root/dsk_opt_goal_20261116/.venv-e0/bin/python /root/dsk_opt_goal_20261116/scripts/09_replay150.py --plan /root/bench_dsk_first150_20261116/first150.plan.json.gz --base-url http://127.0.0.1:8000/v1 --model sf-finance-dsk41-dynamo-vllm-alyeast-prod --output /root/dsk_opt_goal_20261116/results/NEW_UNIQUE_NAME.jsonl
```

重跑需按同一冷缓存/启动约定使用独立输出路径，不得直接覆盖既有结果。22/23及25—27已执行八卡head-row/indexer算子验证；src/dsk_humming_worker.py的全关服务已通过150条回放。详情见STAGE2.md。

## 运行配置

- h20-dsk，容器interactive-su6z5ygi57wp-58895475cb-cz78m，8×H20 97871MiB，driver580.95.05；镜像digest未采集。
- 主环境vLLM0.32.0，commit5df63a7，Dynamo1.5.0，Torch2.13.0+cu130，Triton3.7.1，Humming0.1.17。主环境包清单前后无差异。
- GC0.6.2源码7902831，隔离Torch2.11.0+cu130、Triton3.6.0、CUTLASS4.8.1、NCCL2.28.9，实测使用独立ptxas13.4.92。完整差异见[环境报告](../../../gc-moe/environment/20261116-h20-dsk-isolated/README.md)。
- GC量化测试工具为本次归档的vLLM vendored纯Python math.py；无法取得仓库声明的deep-gemm2.8.2，因此无MegaMoE对照，不宣称原封不动执行完整依赖集。
- 模型/model，原量化权重未修改；数据ZIP SHA256：46601d9dda2cdb0f6f581d99e1fa00d6aa1c11719829966d689606639f8fe2df。
- 当前与上一轮KV容量均19,605,754 tokens、Available KV37.61GiB，最终Graph3.98GiB。

## 限制与证据

整体Goal未完成。无本轮W4A8模型结果，无Indexer/head-row完整模型服务结果，无GC完整vLLM服务兼容结果。Indexer/head-row已有八卡算子证据。A1补查每rank40主干+3draft层，全部仍为FP4权重/BF16激活。没有模型质量标注、独立短trace或真实shape直方图；算子正确性不能代替模型质量。

GC测试桥接与两次shared入口修复均仅作用于测试依赖/审计；失败日志保留。源码、量化工具与脚本身份以evidence归档为准。任何未准入优化不得据此进入性能推广。

| 证据 | 路径 |
|---|---|
| 完整顺序与阶段 | [PLAN.md](PLAN.md)、[execution.json](execution.json) |
| 原始回放 | [JSONL](results/A0-audit-replay150.jsonl)、[summary](results/A0-audit-replay150.summary.json)、[CSV](results/A0-percentiles.csv) |
| 独立复算 | [验证](evidence/A0-replay-validation.json) |
| 主模型实际精度 | [汇总](evidence/baseline-dtype-summary.json)、evidence/loaded-layer-configs/ |
| GC正确性 | [汇总](evidence/gc-correctness-summary.json)、[长重放](evidence/readiness-validation.json)、results/各rank日志 |
| Humming正确性 | [汇总](evidence/humming-finalize-summary.json)、results/humming-finalize/ |
| 环境与源文件 | evidence/baseline-freeze.json、evidence/quant-utils.sha256 |
| 启动与客户端日志 | logs/audit-worker.log、logs/audit-frontend.log、logs/A0-audit-replay150.log |

