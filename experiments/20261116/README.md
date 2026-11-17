# DSV4.1 / vLLM 0.32.0 优化实验总览

- 日期：2026-11-16
- 运行结果：`completed`
- 完成范围：下表已列出的正式批次；整体 goal 尚未完成，长稳态仍在进行。

## 结论与核心结果

已完成默认关闭的 Humming W4A8、输出收尾融合、compact indexer、head↔row，以及 GC 独立环境服务接入和同环境对照；另完成用户追加的 fastokens 两组测试。GC 目前没有显示端到端延迟优势。Humming 完整收尾已验证真实融合，但单轮 TPOT/E2E 存在取舍，最终候选已复验两轮，长稳态正在运行，不宣布最终获胜方案。

所有表中数据为各自一次前150条、0.45发送QPS的原始性能回放，含排空。完整分位数及定义见各组 summary/CSV，不能把跨环境与随机输出长度差异当纯内核收益。

|组|成功/请求|完成QPS|输出token/s|TTFT P95(ms)|TPOT P95(ms/token)|E2E P95(ms)|
|---|---:|---:|---:|---:|---:|---:|
|A1-off-r1|150/150|0.398|491.71|11473.49|12.772|31566.83|
|P-H1-a8-r1|150/150|0.401|535.73|10542.94|10.391|35404.52|
|P-H3-a8-finalize-r1|150/150|0.404|520.16|10788.41|11.252|33878.91|
|P-ALL-w4a8-stride-r1|150/150|0.420|517.93|10490.48|10.225|35981.10|
|P-ALL-w4a8-stride-r2|150/150|0.396|527.98|10494.92|10.585|34015.39|
|P-GC-ALL-w4a8-r4|150/150|0.391|500.26|15067.24|32.878|88212.71|
|P-FAST-GC-ALL-r2|150/150|0.399|479.50|14739.53|25.413|58313.98|
|P-FAST-HUM-ALL-r1|150/150|0.439|572.82|10262.54|9.984|37114.30|
|P-E1-FAST-HUM-ALL-r1|150/150|0.399|537.72|9995.78|9.647|34390.68|
|P-E1-FAST-HUM-FULL-r1|150/150|0.433|549.15|10266.27|10.947|28556.32|

|P-E1-FAST-HUM-FULL-r2|150/150|0.429|521.03|10324.94|10.043|30183.76|

## 复现

远端根目录 `/root/dsk_opt_goal_20261116`，主机 `h20-dsk`。每组 `groups/<组名>/scripts/00_runtime_env.sh`、`01_worker.sh`、`02_frontend.sh`、`03_replay.py`、`04_validate.py` 保存实际命令、开关、回放和校验；编号控制脚本在 `scripts/`。不要覆盖已有组结果，也不要绕过进程归属及正确性门禁启动。

模型 `/model`；TP8/EP、Engram CPU offload、max-num-seqs64、max-cudagraph-capture-size64、max-num-batched-tokens4096、memory-util0.92、block-size64、mm encoder data、deepseek_v41 tokenizer、多模态、DSpark5（probabilistic/block/adaptive）。HTTP8000、worker管理8081、KV events20080。每组新worker，记录合成预检查，正式150条原顺序0.45QPS、temperature0.8/top_p0.95/max_tokens16384，未固定采样seed。

数据为既有 `first150.plan.json.gz`，来源用户ZIP。回放代码 `scripts/09_replay150.py`，逐请求响应和usage在各组results中。失败启动和部分失败回放独立保存，不混入成功组。

## 运行配置

8×H20，driver580.95.05；容器镜像digest未采集。vLLM0.32.0源commit5df63a7、Dynamo1.5.0、Humming0.1.17、fastokens0.3.6。

- E0：Torch2.13.0+cu130、Triton3.7.1、CUTLASS4.8.0、NCCL2.30.7。
- E1：Torch2.11.0+cu130、Triton3.7.1（按用户要求升级）、CUTLASS4.8.1、NCCL2.28.9，vLLM CUDA扩展在E1源码构建。GC0.6.2/7902831，独立ptxas13.4.92。
- 两套环境的前端统一使用已验证E0兼容前端。Python Dynamo补丁与主环境对齐；E1 wheel Rust HTTP schema差异曾导致失败，保留证据。包元数据警告并非全部清零，不宣称pip check完全通过。
- 数据ZIP SHA256：46601d9dda2cdb0f6f581d99e1fa00d6aa1c11719829966d689606639f8fe2df。原权重未改写。

## 限制与证据

没有业务标准答案。按用户认可口径验证成对输出、DSML/工具格式、空输出、循环和截断；结构检查通过不等于业务精度无损。原生FP4/BF16、Humming每行FP8、GC K128 FP8合同不同；GC权重转换亦不能宣称无损重排。

完整收尾仅对安全的1..128行Humming grouped路径启用，大形状回退。当前模型runner scale=1，真实融合省掉的是shared add；八卡短trace每卡860→60次BF16 add，不能宣称省掉原本不存在的scale kernel。

命中证据包含8rank实际dtype、40层分支与独立3请求/20步trace；尚未取得整轮150条无扰动路径命中率，不冒充全程计数。长稳态、最终回退服务验证仍待完成。已知知识库全库校验有5项其他历史报告问题，不属于本次实验数据校验失败。

- [完整阶段记录](STAGE2.md)、[执行状态](execution.json)、[方案及用户调整](PLAN.md)
- [早期原始报告](README.initial-stage.md)，保留当时的数据和限制，不能用作最新状态。
- 各组 `evidence/validation.json`、`quality-gate.json`、`profile-diagnostic.json`、`four-paths.json`；完整融合另有 `full-fusion-proof.json`。
- 各组 `results/replay150.jsonl`、`replay150.summary.json`、`percentiles.csv` 和 `traces/` 保存原始证据。
