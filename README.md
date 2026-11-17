# DeepSeek V4.1 / vLLM 0.32.0：H20 TP8 优化实验

2026-11-16 在 8×H20 上进行的真实服务实验、核心适配代码与逐请求性能记录。当前综合候选为 **fastokens + Humming 原生 W4A8 + 输出收尾融合 + compact indexer + head↔row**。不是通用安装包，也未完成长稳验收。

| 配置 | 成功 | TTFT P95 (ms) | TPOT P95 (ms/token) | E2E P95 (ms) |
|---|---:|---:|---:|---:|
| 原生基线 A1-off-r1（E0） | 150/150 | 11473.49 | 12.772 | 31566.83 |
| fastokens + GC 全组合（E1） | 150/150 | 14739.53 | 25.413 | 58313.98 |
| fastokens + Humming，旧收尾（E1） | 150/150 | 9995.78 | 9.647 | 34390.68 |
| fastokens + Humming，完整收尾 r1（E1） | 150/150 | 10266.27 | 10.947 | 28556.32 |
| fastokens + Humming，完整收尾 r2（E1） | 150/150 | 10324.94 | 10.043 | 30183.76 |

完整收尾候选相对基线的 TTFT/TPOT 有改善；E2E 两轮较低，但请求 bootstrap 区间跨零，不能声称稳定收益。旧收尾同环境组的 TPOT P95 更低，因此完整收尾不是每项指标最优。GC 组合在本次实现和负载下延迟较差。

- [优化机制与代码入口](docs/MECHANISMS.md)
- [环境、复现与开关](docs/REPRODUCE.md)
- [实验总表与历史记录](experiments/20261116/README.md)
- [最终状态与公开导出范围](experiments/20261116/STATUS.md)
- [完整融合两轮统计](experiments/20261116/results/final-repeat-statistics.json)
- `src/`：适配器、Triton/CUDA 内核及 worker。
- `scripts/`：当时的构建、正确性、启动、回放、统计脚本，含历史绝对路径。
- `experiments/20261116/groups/`：成功和失败组独立保存；各组 results、evidence、scripts。

## 统计口径

固定原数据前 150 条、原顺序、0.45 QPS，发送后排空；temperature=0.8、top_p=0.95、max_tokens=16384，未固定采样 seed。TTFT 为首个非空内容/reasoning/tool 增量；TPOT=(E2E−TTFT)/(completion_tokens−1)，是逐请求均值，不是逐 token ITL。分位数采用线性插值。正式回放不启用 profiler；路径证据来自独立 3 请求/20 步诊断，不能当作整轮 150 条命中率。

每个正式组新启 worker；长稳重复批次保留 warm cache，不能混入冷启动 A/B。只有结构质量代理检查，没有业务标准答案。环境、量化合同和随机输出长度差异限制因果解释。

## 当前状态

长稳于 11 月 16 日按用户要求停止，30 分钟和 2 小时阶段均未完成；11 月 17 日确认记录仍在，GC 隔离环境目录已不存在。没有在本次整理中重启服务或重跑性能测试。

公开版保留逐请求数值、token usage、汇总和路径证据，移除对话/响应正文并匿名化业务请求 ID。原始 ZIP、权重、服务全文日志和 profiler 原始大文件不在仓库内；质量评估结果保留，但不能仅凭此公开版重算输出质量。

运行 `python3 tools/verify_records.py` 可复算已完成正式组分位数并校验发布文件哈希。
