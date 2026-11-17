# Goal 执行方案

目标：在固定前150条、0.45 QPS下验证四类优化的可插拔接入与收益。

执行顺序（按用户最新要求修订，Humming W4A8 优先）：
1. 已完成的原生 W4A16/BF16 激活审计与回放作为 A 组；不再扩展 BF16 激活优化测试。
2. 立即执行 H1：主模型 Humming W4A8，先逐 rank 核验40层W13/W2实际为FP8 E4M3激活、FP4 E2M1权重。禁止静默回退BF16；draft暂保留原生并明确披露。
3. H1按前150条/0.45QPS完成质量代理检查，再执行原始temperature0.8配置的A/B/BA性能重复；报告TTFT/TPOT/E2E、吞吐、输出长度与实际kernel命中。质量诊断参数与性能参数分别记录。
4. 执行 H3：W4A8+tiled输出融合，与W4A8单项比较；再验证W4A8+完整scale/shared收尾融合（H4）的正确性与性能。BF16输出融合H2只保留已经完成的控制证据，不再安排新轮次。
5. Humming阶段结论出来后，再继续Indexer、head↔row单项/组合与GC隔离服务同环境对照。已经完成的结果及源码编译产物保留。GC编译暂停，避免干扰Humming测量。
6. 测试获胜组合，完成30分钟及2小时稳定性，再恢复原服务并汇总。整体Goal范围保持不变，仅调整顺序和BF16控制组投入。

Humming正式性能比较固定：同一主环境、模型/数据/启动参数、前150条原顺序0.45QPS、原始temperature0.8/top_p0.95/max_tokens16384；每组至少3轮、重启统一缓存状态，禁止与源码编译或其他GPU任务并跑。W4A8内核有效性必须由运行时dtype配置和独立短trace证明，不能仅以环境开关为依据。

以下详细计划是预注册要求，并非执行结果。

### 测试计划：阶段0，冻结与路径审计

保存各组完整依赖、源码commit与本地修改、扩展二进制哈希、实际加载的CUDA/NCCL库、driver/ptxas、8卡拓扑、GPU时钟温度功耗、进程、环境变量、模型config、数据与脚本哈希。所有环境变量使用白名单，不保存凭据。

主环境E0，GC隔离环境E1。先记录必要差异，不将wheel版本当作二进制兼容证明。

记录每个MoE层W13/W2的a_dtype、b_dtype、scale粒度、实际experts类、专家映射和归约阶段；每条优化记录选择/回退次数、原因、shape范围。采集独立短trace验证实际kernel命中；性能测量关闭profiler与高频日志。

Gate：主环境无自定义优化模块调用；全部开关关闭时执行链恢复原生。当前已核对三个入口SHA与官方0.32一致，但W4A16/W4A8不能仅凭环境变量判定。

### 阶段1，环境兼容与分布式生命周期

1. E1导入torch、cutlass、gc_moe；pip check；验证实际模块路径与版本。
2. CUDA绑定符号/Green Context资源查询、SM分区粒度、workqueue能力；记录实际通信/计算SM数，不把请求值当实配。
3. 独占8卡后执行P2P、对称显存rendezvous、GC创建销毁、eager及外层CUDA Graph嵌套调用。
4. 变化输入A/B交替replay至少1000次；多层串行workspace复用、主模型/draft分离、重复启动/退出3次，无挂起、陈旧输出或资源泄漏。
5. E1若要加载vLLM，需在固定5df63a7源码上适配并重新编译Torch相关扩展及匹配的FlashMLA等依赖；禁止直接复制E0二进制。先完成原生完整模型与质量回归，再进入服务A/B。若无法实现，E1仅报告算子结果；可另立GC向E0移植分支，不混为同一实验。

### 阶段2，正确性（先于性能）

GC MoE：真实模型H5120、I2304、E384、topk6、EP8，主干每rank48专家；另核对DSpark专家数量及形状，未适配draft时显式保留原生。

覆盖均衡与倾斜路由、重复/无效专家、零token rank、非均匀token数、capacity边界、非整tile、最大支持行数；对比本地专家映射、routing weight、routed scale=1.5、SwiGLU clamp10、shared加法及归约位置。GC输入必须按所有权分配，不能将TP复制输入当成8份不同token；输出布局恢复与额外通信都属于被测链路。

MoE数值分两层：固定相同量化张量验证计算/路由/融合；再对BF16或高精度参考衡量量化误差。Humming每行FP8与GC K128 FP8不是相同精度配置，不能要求或宣称整个后端逐位一致。记录max_abs、relative L2、cosine、非有限值、原生重复运行噪声；输出融合保留BF16舍入边界。误差阈值在正式性能测试前冻结，不能看到结果后放宽。

Indexer：source20→consumer24/28/32/36；32 index heads、D128、top512、候选2048块×8；覆盖真实逐query因果长度、DSpark verify展开、rank独立物理页表、请求重排、空候选/重复候选、Graph padding及毒化旧缓冲。比较score与top-k集合；同logits tie规则单独检查。第一版仅decode/verify；P5c prefill另设实验，不用旧共享候选结果替代逐query验证。

Head-row：Q/O搬运逐位一致；attention按合理浮点容差；覆盖M=1/7/8/9/32/64/127/128/129/256/512及真实负载出现的行数，ratio0/1/2，尾部、空rank、动态窗口、延迟rank。metadata按接收rank本地KV映射重建，各rank一致选择路径。小M、未测形状与不支持拓扑回退原生。

Gate：任何非有限值、错误路由、陈旧数据、越界、死锁或重复归约均阻止性能准入。数值变化必须与质量验证一并报告。

### 阶段3，算子及完整层A/B

先在E0取真实请求shape直方图，再覆盖M=1/2/4/8/16/32/64/128/256/512与实测prefill/verify尺寸；context 2K/8K/16K/32K/64K/256K及实际出现的长上下文，避免盲目全笛卡尔积。GPU事件与host wall同时采样，8rank取完整调用的关键路径，报告P50/P95/P99、原始样本、CV与峰值显存。

编译、权重转换和workspace分配放在稳态计时外，但单独报告启动时间和常驻显存。每个shape预热后至少30计时窗口；低延迟算子每窗口多次调用并记录次数，AB/BA交替，重复至少3轮。固定输入、缓存策略和kernel启发式，不把冷缓存与热缓存结果混用。

GC比较必须包含输入量化、布局转换、EP dispatch/GEMM/combine、输出恢复、scale与shared收尾；Humming也测同一语义边界。GC-R保留原shared分支，GC-RS再融合shared；不重复加shared或collective。GC与primary-context对照仅在协议与launch几何一致时用于GC资源分区归因。

Head-row包含双向搬运、动态metadata准备与FlashMLA；Indexer包含source候选发布及四consumer完整链路，另报各阶段。不得用独立分段耗时相加代替完整链路测量。

### 阶段4，完整服务矩阵

以下是完整组别清单；执行优先级以上方修订顺序为准，未通过正确性/质量Gate不开展正式性能比较。

| 组别 | 环境 | 配置 | 目的 |
| --- | --- | --- | --- |
| A0 | E0 | 当前原生 | 复现150条基线与路径身份 |
| A1 | E0 | 集成代码存在、所有开关关闭 | 证明零行为改变 |
| H1 | E0 | 仅Humming A8 | 激活精度收益，先确认A0实际dtype |
| H2 | E0 | 仅tiled finalizer | 同精度输出收尾收益 |
| H3 | E0 | A8+tiled | 分离量化与二维kernel增益 |
| H4 | E0 | A8+完整shared输出融合 | 与H3比较完整融合 |
| I | E0 | 仅compact indexer | 稀疏选择收益 |
| R | E0 | 仅head-row | attention收益 |
| E1-N | E1 | 重新编译且通过验证的原生vLLM | 分离环境差异；不可缺省 |
| G1 | E1 | GC routed、原shared | 与E1-N比较 |
| G2 | E1 | GC routed+shared | 与G1比较 |
| H+I、H+R、H+I+R | E0 | 获胜Humming配置与单项/组合 | 检测相互影响 |
| G+I、G+R、G+I+R | E1 | 获胜GC配置与单项/组合 | 检测相互影响 |

Humming与GC为互斥MoE后端，不设置Humming+GC同层叠加组。主干和draft分别记录命中率。新写Async TP不纳入本轮。

共同服务参数沿用实际跑通配置：TP8、EP、DP1、PP1、Engram cpu_offload、max-num-batched-tokens4096、gpu-memory-utilization0.92、block-size64、max-num-seqs64、max-cudagraph-capture-size64、mm-encoder-tp-mode=data、tokenizer-mode=deepseek_v41、multimodal、DSpark5(probabilistic/block/adaptive)。Dynamo1.5.0 frontend使用dyn-chat-processor=vllm。完整原命令存于20261116-h20-dsk-first150/scripts/。

每个A/B均单独启动进程；缓存从同一状态开始；kernel预热规则相同。正式请求前不加入未记录的业务请求。重复运行采用A-B-B-A等顺序，至少各3轮；每轮前150条、原顺序0.45QPS、发送331.111秒后等待排空，保留max_tokens16384/temperature0.8/top_p0.95、工具与图片。先复现无额外业务预热的既有口径；稳态预热实验单独标记，并统一清理请求缓存。

若输出量变化明显，不将E2E缩短直接当吞吐优化：报告实际输入/输出token、finish_reason、DSpark接受率、运行/排队请求和KV占用。配置可支持时另加固定输出长度诊断组，不能替代原始业务回放。

### 阶段5，质量、长稳态与统计

质量：固定seed及相同请求的成对诊断，检查工具调用schema/参数、结构化格式、空输出/异常循环/截断、代表性图像问答及独立任务评分。保留原始0.8采样性能组，另设低随机性质量组；文本不同本身不等于错误。用户数据没有标准答案时，不能仅以150条HTTP成功率宣称模型质量无损。

长稳态：获胜组合在8卡独占下先30分钟，再2小时，监测错误/挂起/内存增长/回退频率；容量与饱和吞吐另做0.45/0.9/1.8 QPS等递增试验，触发错误或队列持续增长则停止该档并排空，不无限堆积。

正式报表：TTFT、TPOT、E2E的mean/P50/P90/P95/P99及有效样本数；成功/失败、发送QPS与含排空完成QPS、output tok/s、输入输出长度、显存、各优化命中率。TPOT=(E2E-TTFT)/(completion_tokens-1)，不把SSE块当token。排除role-only首块，缺失usage单独报告，失败请求不混进成功延迟且必须单列。

各轮分别报告，再给跨轮中位数及成对差异/置信区间；不只挑最好一轮，不将150条的P99当稳定尾延迟结论。只有同环境、同精度/布局合同的对照可归因于单项机制；E0与E1最终整体差异单独列出。

建议预注册工程准入目标：TTFT或TPOT主指标稳定改善至少5%，另一个主指标不回退超过3%，P99不得持续回退超过5%；都属于待冻结的工程标准而非既有实测结论。正确性及质量Gate优先于性能；不达标的形状回退原生。

### 停止、回退与交付

任意GPU非法访问、跨rank挂起、错误归约、非有限值：停止本组，保留失败证据，仅清理该组进程。关闭所有新开关回到原路径；量化/权重布局变更必须重启加载，不能在活跃请求中热切换。GC workspace先退役外层Graph再销毁。主环境最终按原脚本恢复并验证models接口及短请求。

每组保存README、编号脚本、配置、依赖锁、源码/扩展指纹、逐请求JSONL、finish_reason/usage、指标CSV、服务/客户端日志、正确性报告和独立短trace。任何失败仍归档；不覆盖刚才150条结果。

## 运行配置

硬件、driver及模型/data共用。软件实际清单及差异以evidence中的安装后文件为准。GC源码固定7902831；源码归档不含Git元数据，另记录完整commit和tar SHA，不能仅依赖wheel版本号溯源。

## 限制与证据

本计划尚未产生新优化的性能或质量结论。环境安装/import成功不等于8卡算子通过，也不等于vLLM与Torch2.11兼容。E1完整服务对照存在明确源码编译兼容Gate，未通过不得填端到端性能表。

知识库校验：本Result文件已通过规则检查；全库仍有5项先前存在的其他实验链接/frontmatter问题，未改动无关历史记录。
