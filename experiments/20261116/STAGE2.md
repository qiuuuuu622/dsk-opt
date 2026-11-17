# 阶段2：八卡正确性与全关集成控制

日期：2026-11-16。当前A1回放进行中，整体Goal未完成。

- Head-row：真实TP8、11档行数、eager/Graph，共176个rank-case，每case200次变化输入；搬运逐位一致、配置不一致拒绝测试通过。未添加旧实验的NCCL或multicast禁用环境变量。
- Head-row attention：ratio0/1/2，M8/64/128/512，32K上下文及变化长度；12形状×8rank均通过独立参考与原生对照，最大nRMSE0.00205412。计时仅1窗口×1次，不能作为正式收益；metadata准备在计时之外，正式完整层实验必须计入动态metadata。
- Indexer：真实8rank独立物理页表、六行verify、变化输入Graph、重复/空/部分候选、同分及NaN尾部检查通过。
- Indexer原生打分：每卡8形状/压力组合×4consumer，共256项；最大score nRMSE7.14136e-8，top-k集合100%一致。N6参考展开为逐query N1，仅用于数值比较，无性能结论。数值脚本的逻辑rank固定0，但每个物理GPU分别执行；真正各rank契约由独立distributed verify覆盖。
- A1-off-r1：同一套Humming可选接入代码存在，DSK_HUMMING_A8=0、DSK_HUMMING_FINALIZE=0。8rank各43层（target40+draft3）均确认W4A16；KV容量19,605,754 tokens，与A0相同。当前按原150条、0.45QPS执行，结果未完成时不得计算整轮结论。

证据：evidence/stage2-correctness-summary.json、results/headrow-tp8-verify.json、results/headrow-attention-pilot.json、results/indexer-tp8-verify.json、results/indexer-scores/、groups/A1-off-r1/。脚本22/23/25/26/27，测试源码固定保存在src/headrow与src/indexer。GC、W4A8及两项attention优化均未接入当前服务。


## A1 完成与质量基线启动

A1-off-r1：150/150 成功；P95 TTFT 11473.493 ms、TPOT 12.772 ms/token、E2E 31566.825 ms，输出吞吐 491.705 tok/s。8 rank target 配置与 A0 完全相同，主干与 draft 仍为 W4A16，所有优化关闭。跨基线波动不能记为优化收益。原始数据位于 groups/A1-off-r1。

cache_salt 探针未隔离缓存：a,a,b,b 对应 cached_tokens 0,2112,2112,2112。因此各组重启后回放，保持起始缓存状态一致。Q0-native-r1 正在原环境用 temperature=0、seed=1234、前150条0.45QPS保存输出，单独用于质量对照。没有业务标签，结构和成对输出检查不能证明业务准确率无损。


## 质量基线完成，开始同精度融合

Q0-native-r1完成150/150，输出190212 tokens，全部stop且usage完整。170个正文DSML调用，离线检查未发现未闭合标记、未知工具、重复参数、非字符串参数JSON回退、空输出、截断或重复尾片段。数据采样为temperature=0/seed1234，不能混入原0.8采样性能表。用户确认无标准答案，按成对输出与结构对照准入；仍需原生重复运行校准差异。

QH2-finalize-r1已启动：仅FINALIZE=1，A8=0；对应<=128 tokens的tiled finalizer，不包含shared专家融合。worker350032/frontend350031，完整命令、环境白名单和源码hash见group目录。质量结果未出。

GC E1服务兼容进展：固定5df63a7原始源码已上传；安装独立构建依赖并通过72包依赖检查。首次构建因无Git退出；Git仅下载解包至tools/git，未安装系统包。34脚本在CUDA_VISIBLE_DEVICES为空、MAX_JOBS=2下重试源码扩展构建，PID348775，尚未证明Torch2.11服务兼容。主环境二进制没有复制到E1。


## QH2结果与接入进展

QH2-finalize-r1完成150/150成功，172个DSML调用，无新增结构告警。8rank加载配置与Q0相同。完整文本精确匹配0/150，不能直接归因于优化：先运行Q0-native-r2并通过只读启动器审计引擎实际temperature/seed。QH2已停止，尚未完成质量准入或正式性能比较。

compact indexer初版接入代码为src/dsk_indexer_adapter.py和src/dsk_indexer_worker.py，默认关闭；原生source发布候选到单个串行工作区，支持FP8扁平decode/verify N1；其他形状、PCP/DCP、未知候选缓冲区回退原生。AST源码指纹及custom-op对象身份检查通过，未部署到模型服务。36/37脚本正在8卡验证池化绑定、4层候选复用和变化输入Graph；该参考用于接入生命周期而非替代此前native score算术比较。


Indexer池化工作区验证完成：8rank×9shape，rows=1/6/12/32/64/127/128/129/384；4consumer、20次A/B变化Graph，共5760次top-k集合一致，错误候选buffer及未发布回退通过。见evidence/indexer-pool-summary.json。

Q0-r2启动器首次失败已归档（未发送请求）；主入口保护修复后r2b已正常开始回放。只读审计确认effective SamplingParams的temperature=0、seed=1234、top_p=1（贪心规范化）、max_tokens=16384。继续比较原生重复运行，尚不将QH2的完整文本差异归因于优化。


## 用户修订：Humming W4A8 优先

BF16激活优化不再扩展，仅保留原生基线及已完成H2控制证据。当前QI已150/150完成并停止；GC编译进程组359035已SIGSTOP保留增量产物，消除CPU竞争。QH1-a8-r1启动（worker372474/frontend372473）：主干启用A8，输出融合关闭，draft原生；回放前必须核验8rank×40层×W13/W2共640项实际FP8 E4M3激活及FP4 E2M1权重，否则禁止发送请求。随后优先原生与W4A8性能A/B、W4A8+tiled以及完整shared收尾融合。整体Goal其他项目后移而非取消。

QH1加载后实际dtype门槛已通过：8rank×40主干层×W13/W2=640项均为FP8 E4M3激活、dynamic_token；FP4 E2M1权重。Draft48项仍为BF16激活。证据groups/QH1-a8-r1/evidence/pre-replay-dtype-audit.json。尚无该组端到端性能或质量结论。


### W4A8 正式单轮与组合启动（2026-11-16）

P-H1-a8-r1：150/150，0失败。相对 A1-off-r1，P95 TTFT -8.11%，TPOT -18.65%，E2E +12.16%；总输出 token 185443→200388（+8.06%），输出吞吐 491.705→535.726 token/s（+8.95%）。仅为单轮筛选，输出长度不同，不能证明整体 E2E 改善。计时结束后3条短请求采集八卡 trace，每卡捕获1600次 W4A8 Humming 内核；不是完整150条的命中率。原始 trace 保留远端 groups/P-H1-a8-r1/traces，解析证据和原始150条已归档本地。

P-H3-a8-finalize-r1 已启动，worker PID 383176，frontend PID 383175；沿用同一配置，开启 A8 + tiled finalize（不是 scale/shared 全融合），保留响应做同轮结构检查。GC 编译仍暂停，避免计时争用。


### 三项组合接入与准入（2026-11-16）

用户要求输出融合、compact indexer、head↔row一起测试，保留W4A8。P-I单项未启动并取消排期。P-H3已完成150/150，无结构异常，167个DSML调用；P95 TTFT10788.407ms、TPOT11.251665ms/token、E2E33878.907ms，吞吐520.159token/s。其短trace每卡确认_grouped输出收尾内核。

新增默认关闭的dsk_combined_worker与dsk_headrow_adapter：构造期集中分配对称内存；动态Q指针；逐次切分元数据并补齐；row planner跟随原生每步metadata生命周期；显式拷贝回调用方output。主模型40层开启，draft与prefill保持原生。每卡工作区约759MiB（另加indexer约27MiB），须计入KV容量变化。

八卡headrow adapter检查120个rank-case、2400次变输入Graph，最大nRMSE0.001703。初版两路径逐元素阈值有单元素超界，后以原阈值分别对照独立FP32参考；两路径均通过，Graph/eager一致。中间诊断版本因条件分支导致各rank通信次数不一致，已用all-reduce同步诊断分支并保留失败记录。最终八卡全部通过，未放宽FP32容差。详见evidence/headrow-adapter-validation-notes.json和results/headrow-adapter。

P-ALL-w4a8-r1 worker388990，前端388989，三项开关加A8全部开启；服务就绪后原采样固定前150条0.45QPS，同轮保存输出检查结构。所有性能结论待回放和实际kernel路径证据。


### Indexer 接入回退定位与修复（2026-11-16）

P-ALL-w4a8-r1完成150条并通过结构检查，但短trace中compact发布和打分均0，不能当作三项全部生效。W4A8、tiled finalize、head-row已实际观测。最初以为next_n6导致旧N1 guard回退，补齐B×next_n映射并通过72rank-shape/5760变输入Graph精确集合检查；随后运行metadata证实V2实际next_n1，B×6不是此次根因。

真正原因：FP8 index cache形状[pages,64,1,132]，page stride116928，页内stride132/132/1；整体noncontiguous。scorer已有真实page stride支持，但adapter的全连续资格检查错误拒绝。修复仅允许合法页内布局、足够且16字节对齐的page stride，保留真实stride，不复制大cache。八卡用padding字节165、page stride116928对照相同连续cache，72rank-shape、5760变输入Graph top-k集合精确通过。

P-ALL-w4a8-spec6-r1与r2都在正式150请求之前停止（r2有单独合成32输出token预检），无额外完整E2E。P-ALL-w4a8-stride-r1已启动且八卡compact分支进入，150条进行中。包含一条2119输入/32输出token的合成用户预检，独立记录，不计入150时间。不能把启动分支记录当作runtime kernel命中率，完整回放结束后单独短trace核验。

复核参考仓库Experiment-Results，HEAD/origin/main均548fe680b53febbfbea9d908ff3b49a6b06225bc。P8/P4 grouped共享候选契约不得直接套用逐query候选；当前保留逐query。P5c为独立prefill范围，本次仍聚焦decode/verify组合。模块微基准收益不外推服务。数据集原基线150条中位prompt44401tokens，25条至少64K。


### 三项实际命中组合最终单轮：P-ALL-w4a8-stride-r1

本轮completed：W4A8 + tiled output finalize + compact indexer decode/verify + head-row；不是scale/shared全融合，GC未加入。前150条固定顺序0.45QPS，原始temperature0.8配置，150成功0失败，elapsed357.327071秒，185070输出tokens，517.928853token/s。原生基线185443输出tokens、491.705183token/s，输出量接近，本轮输出吞吐+5.33%；这是固定到达负载结果，不是最大吞吐容量。

|指标|均值|P50|P90|P95|P99|
|---|---:|---:|---:|---:|---:|
|ttft_ms|2646.768977|893.885584|8134.997947|10490.478114|12673.062958|
|tpot_ms_per_token|5.771960|5.283011|8.348781|10.225381|13.331790|
|e2e_ms|9947.189242|6331.536679|21337.844276|35981.097934|49100.008458|

相对A1-off原生：P95 TTFT -8.57%，TPOT -19.94%，E2E +13.98%。相对H3（A8+finalize）：P95 TTFT -2.76%，TPOT -9.12%，E2E +6.21%。单轮随机采样输出的逐请求长度不同，不能宣称总体胜出或统计显著。原生与组合总输出量仅差-0.20%，但总量接近不代表逐请求尾部相同。

170个DSML调用，空输出/截断/工具JSON/DSML不平衡/未知工具/重复参数/重复尾部均0；这是结构代理检查，不是业务准确率。八卡post-timing短trace每卡：W4A8 1600次、tiled finalize860次、compact score80次、candidate publish20次、head-row exchange1600次。证明这些路径实际执行，不是完整150条的命中率。原始trace仍在远端该group/traces，解析证据、源码快照、原始150条与统计已归档。

环境：vLLM0.32.0、Torch2.13+cu130、Triton3.7.1、Humming0.1.17，主依赖未更换。相同gpu-memory-utilization0.92；headrow工作区796051456B/rank，indexer约27MiB，KV18881293tokens（基线19605754）。单独1条2119输入/32输出的合成用户预检，不计入150；启用但未运行的profiler配置与旧原生基线不同。GC编译暂停，timed150期间不采样。服务PID403670/前端403669保留，HTTP127.0.0.1:8000。


### 用户要求同配置复测：P-ALL-w4a8-stride-r2

已核对插件源码SHA与r1一致，fresh worker410048/前端410047，独立namespace与日志，重启清空上一轮cache；相同合成preflight与前150条0.45QPS、原采样，保留响应同轮结构检查。不重复完整质量批次和算子测试。结束后对照native与r1。

上轮E2E P95对应的排序边界请求已改变：native第142/143名为请求44/97；组合为请求142/91。请求142输出2468→6870 tokens，TPOT5.847→5.104ms，E2E15.126→35.725s。说明总输出token近似相同不能保证逐请求尾部长度相同；不能直接把P95+14%解释为每条请求慢14%。完整逐请求代数分解在groups/P-ALL-w4a8-stride-r1/results/e2e-paired-decomposition.json，非因果控制实验。


### 同配置复测完成：P-ALL-w4a8-stride-r2

状态completed，150/150成功、0失败；源码SHA与r1一致，fresh worker清空cache，0.45QPS、前150条原顺序及原采样，仍保留单独合成preflight。输出200051tokens（r1为185070，+8.09%），elapsed378.899545秒，527.978992token/s。170个DSML调用，全部结构异常标记0。未重复短trace，沿用r1同源码八卡kernel证明，并核验本轮八卡compact startup分支与开关；不能宣称本轮整150条测得命中率。

|指标|均值|P50|P90|P95|P99|
|---|---:|---:|---:|---:|---:|
|ttft_ms|2661.724652|908.526367|8068.621076|10494.921971|12683.087583|
|tpot_ms_per_token|5.703347|5.302085|8.547822|10.585234|13.334406|
|e2e_ms|10381.034272|6248.024700|20507.790951|34015.392314|55836.571986|

相对原生A1-off：P95 TTFT -8.53%、TPOT -17.12%、E2E +7.76%，本轮输出吞吐+7.38%。相对组合r1：P95 TTFT +0.04%、TPOT +3.52%、E2E -5.46%。两次组合的TTFT/TPOT低于基线，而E2E P95均高于基线；不能把+14%当作稳定退化幅度，也不能声称E2E尾延迟改善。生成长度及其逐请求分布改变，未做固定输出长度的因果隔离。

服务PID410048/前端410047保留。原始输出、percentiles.csv、相对native/r1的comparison JSON/CSV、验证报告、运行日志、脚本与源码快照均在groups/P-ALL-w4a8-stride-r2。


### 恢复 GC 隔离环境完整服务兼容工作

组合r2正式150条已结束，原编译进程组359035从T恢复，未重新从零构建。确认其子cicc/ptxas进程运行，进度至少187/412。主服务410048保留空闲，未启动新timed批次。隔离env默认ptxas由旧13.2路径改为此前已通过GC测试的13.4路径；当前build原本就显式覆盖为13.4，无编译参数变化。

已生成common runtime requirements和main版本约束（保留现有GC/build固定版本），只进行uv --dry-run解析，尚未安装、更未改变主环境。Dynamo1.5.0的vllm extra要求0.28.0，因此采用ai-dynamo基本包及显式vLLM运行依赖。主torchvision0.28绑定Torch2.13，E1需0.26（官方兼容表）；torchcodec0.17官方声明>=2.11兼容，可保持主版本，仍需实际import验证。证据在evidence/gc-runtime；完整服务兼容和同环境AB未完成。


### GC 隔离环境：24 jobs 构建与冻结运行依赖

旧8 jobs已排空后增量切换，当前构建PID417146；本次现场确认141/214且cicc持续占用CPU。公共147包及CUDA配套3包的no-deps dry-run均成功。保留实际主环境版本及已有GC必要差异，完整清单见evidence/gc-runtime/runtime-frozen-plan.json。主环境既有Pydantic约束冲突与缺失aisimulate均如实保留，不因此宣称运行兼容。

新增53安装、54独立导入检查、55构建成功后控制器（PID420954）。控制器验证构建PID/starttime并等待退出码0，随后才改隔离venv；未改主环境。CPU导入测试覆盖Torch/GC/多媒体/vLLM扩展/DSV41/Dynamo，不替代模型加载和8卡服务正确性。此时安装尚未执行、完整服务仍待验证。

正式压测入口已移除硬编码旧编译PID359035的检查，改为扫描本任务GC构建/安装进程及独立编译子进程。现场识别58个活跃相关进程并确认会拒绝计时准入，防止新PID编译器污染未来AB。仅更新后续运行脚本，不更改已完成组的脚本快照。


### GC 构建依赖修复与 E1 原生服务准备

24 jobs构建在DeepGEMM步骤失败：缺少elfutils/libdwfl.h；cgroup OOM计数0。仅下载并解包与已安装系统版本一致的libdw/libelf 0.190-1.1ubuntu0.1到tools/elfutils/sysroot，不修改系统包。增量重试进一步发现cusparse.h搜索路径缺失；头文件已存在隔离venv的nvidia/cu13/include，因此新增CPATH/LIBRARY_PATH后重试（PID423028，脚本58）。两次失败完整日志保留，安装控制器均未越过失败门禁。最新控制器PID423029只等待成功后安装。

E1-native-r1脚本准备完成，所有实验开关关闭，原生模型参数不变，PYTHONPATH显式指向同commit源码；尚未启动。运行检查改用0.32实际扩展名_C_stable_libtorch/_moe_C_stable_libtorch，并验证Torch/vLLM模块路径及Dynamo CLI help。源码采用setup.py egg_info生成元数据，禁止借用主环境二进制。

GC同commit CUDA扩展增量构建成功（58脚本，退出0）；原生_C_stable_libtorch约269MiB、_moe_C_stable_libtorch约46MiB已复制源码目录，其余FlashMLA/FlashAttention/DeepGEMM安装输出见构建日志。可选Rust build_rust提示无编译器，不能据构建退出0认定所有运行模块可用。依赖安装PID423765已启动，CPU导入/CLI检查随后执行；尚未开始E1模型加载或压测。


### GC运行合同审计和已编译扩展链接

根据四项组合r2的8rank实际模型元数据，全部320个主模型层实例均为E384/local48/H5120/I2304/SP1/skip_final_all_reduce=false，各rank另有3层draft。GC routed-only接入需要分片TP复制输入、恢复行顺序，并向MoERunner正确标记routed已归约，shared维持原生分支，禁止重复all-reduce。FP8 K128与Humming dynamic_token量化差异仍需单独质量验证。完整记录见evidence/gc-integration-contract.json，尚不是GC服务实现或路径证明。

脚本60在CUDA_VISIBLE_DEVICES为空时，以隔离Torch2.11.0+cu130分别torch.ops.load_library加载_C_stable_libtorch和_moe_C_stable_libtorch，二者退出0且无stderr。此证据只覆盖CPU侧动态链接与注册，不覆盖GPU执行、FlashMLA或完整服务。运行依赖安装PID423765仍活跃且有网络接收，安装前后freeze此时一致，未误判为卡死或重启。


### GC 直接组合接入与首轮启动

公共依赖切换复用预解析缓存后147包1.32秒解析并安装；CUDA配套3包安装完成。Dynamo同版本wheel与主环境存在11个Python差异及1个缺失测试文件，已备份wheel原件后对齐主环境已有补丁（不复制二进制）。vLLM源码共同2878个Python文件全部一致，仅生成_version.py不同。全部17项导入/CLI检查通过。E1-native启动在GPU清理查询超时处停止，未发正式请求；用户随后明确改为直接GC组合。

新增dsk_gc_adapter/worker：MXFP4原始权重经GC原生转换（可能包含指数范围规范化round-downscale），40主模型层使用GC原生FP8 K128算子；复制输入按rank分片并all-gather恢复，标记routed已归约；原shared保持正确归约，BF16 scale/shared收尾融合。8rank×10行数×2权重=160个原生GC逐位等价case及收尾逐位检查通过，两组权重共享workspace的20步Graph共320次检查通过，脚本65/66退出0。

P-GC-ALL-w4a8-r1 worker429206/frontend429205，8卡模型加载约61秒、37.66GiB/rank，40层GC元数据全部记录。首次Graph预热在compact indexer的Triton3.7 allocate_mbarrier API失败，正式请求0条；保留全部日志。终止本组后8个rank孤儿进程仍留在原worker PGID，按PGID、原日志PID和隔离库映射验证归属后TERM再KILL清理。临时单CTA等价兼容改动尚未运行验证，用户要求直接升Triton3.7后已撤回。

CuPy14.2.0已与主环境对齐。flashinfer-cubin0.7.0.post1在默认公开索引无对应版本，组合安装解析失败且未更改环境；隔离环境继续使用本地JIT生成匹配Torch2.11的FlashInfer内核，不复制E0的JIT二进制。Triton3.7.1升级正在执行，需短验证后恢复组合150。


### 用户要求 Triton3.7 后重启 GC 四项组合

Triton3.6.0已卸载并安装3.7.1（与E0一致），Torch仍2.11.0+cu130；constraints同步记录显式覆盖，原始indexer代码已恢复。升级后GC适配器160个case及320次双权重共享workspace Graph检查再次通过。Indexer测试首启遇到vLLM源码benchmarks包遮蔽测试输入模块，未执行GPU检查；改为精确文件导入后，8rank×9shape×4consumer×20steps共5760次top-k集合检查通过（退出0）。

P-GC-ALL-w4a8-r2已启动：worker437682/frontend437681/replay_launcher437683；目前正在CUDA Graph预热，尚未生成正式150结果。脚本73在短检查成功后启动本组；脚本74（PID441653）等待正式完成后计算分位数/结构代理、与E0基线及Humming组合比较并采集独立短trace。对比明确标记跨环境和FP8量化粒度差异，不宣称纯GC因果收益。


### Triton 3.7.1 后前端兼容修复（2026-11-16）

GC 隔离环境已升级 Triton 3.7.1，GC adapter 160 个直接 API 对照、320 个图更新检查，以及 compact indexer 5760 个集合检查通过。P-GC-ALL-w4a8-r2 完成模型加载并开始正式回放，但隔离 Dynamo Rust 前端对数据集部分历史 assistant content 返回 HTTP 400（反序列化错误），故本轮无效，保留原始结果。仅 Python 文件一致不能证明 Rust HTTP schema 相同。

P-GC-ALL-w4a8-r3 改为与原生基线完全相同的 E0 前端 Python/runtime，GPU worker 仍为 E1 Torch 2.11 + Triton 3.7.1，GC native W4A8/输出融合/compact indexer/head-row 均保留。没有改写测试数据。首次启动因上一 rank GPU 释放未完成而在启动服务之前被保护检查拒绝；确认释放后原地重试，保留 prelaunch-gpu-release-wait.json。当前 controller=447575, frontend=447578, worker=447579, replay_launcher=447580, finalizer=447653。正在初始化，尚未形成有效 150 条结果。脚本 75–77 已归档。


### GC 接入层图缓存修复与 r4（2026-11-16）

r3 在 53 条请求完成后 GC 内部 cuGraphInstantiate 报 CUDA_ERROR_OUT_OF_MEMORY，终止并保存为 failed。GC runtime 的 _child_key 包含参数地址与形状，而接入层每次动态分配输出；此为缓存增长的代码层候选原因，未取得失败时实际缓存计数。修复为复用共享固定输出缓冲区，每卡行数按 2 的幂补齐，补齐行路由权重为 0，原生 GC 内核不变。八卡 160 个直接原生 API 对照及 320 个图更新检查均 bitwise 通过。r4 保留四项优化，基线前端，固定 150 条/0.45 QPS。controller=462402, frontend=462405, worker=462406, replay_launcher=462407, finalizer=462472。截至 53 条成功，0 错误、无排队；不能提前视作完整结果。


### GC 原生 W4A8 四项组合最终结果（r4，2026-11-16）

Triton 3.7.1。GC native W4A8 + GC 输出 scale/shared 融合 + compact indexer + head↔row。150/150 成功，0 失败，原顺序 0.45 QPS，383.173 秒完成（含收尾），500.265 output token/s。

|指标|P50|P90|P95|P99|
|---|---:|---:|---:|---:|
|TTFT (s)|2.509|13.501|15.067|17.093|
|TPOT (ms/token)|16.077|28.536|32.878|39.718|
|E2E (s)|16.563|49.672|88.213|136.482|

相对 A1 原生 E0 基线，P95 TTFT +31.32%，TPOT +157.41%，E2E +179.45%；相对上轮 Humming 四项组合 r2，P95 TPOT +210.60%，E2E +159.33%。这只是跨环境完整配置的单轮对照，不能归因成纯 GC 因果收益/损失：E1 Torch2.11/CUTLASS4.8.1/NCCL2.28.9 与 E0 不同，量化粒度不同，随机采样生成长度也不同。两环境均 Triton3.7.1，r4 前端与 E0 相同。

质量代理检查：空输出、截断、异常循环后缀、DSML不平衡/未知工具/重复参数/JSON异常均为 0，解析到163次DSML调用；无业务标准答案，不能视作业务精度验证。

正式计时结束后的3请求 trace：8rank每卡原生 GC role kernel 2400 次（dispatch/routed/finalize 同名合计，不能作为纯compute次数），GC staging800、输出融合800、compact score80、candidate publish20、headrow1600。四条路径均观察到；不是整轮150条的命中率。原始 trace 留在 h20-dsk 对应组 traces，摘要和原始150条已本地归档。

已完成本轮150条请求。结论：此接入版本未获得端到端延迟收益，不应作为优选方案。服务保留：frontend462405/worker462406，端口8000，未继续发送正式压测。前置失败r2/r3已归档，不能混入分位数。GC图缓存根因仍为代码证据支持的推断；固定输出/行数分档后本轮未再OOM。后续若继续，应先分析现有trace的GC/通信/CPU图调度开销，再决定是否值得追加同环境对照，避免盲目增加端到端轮数。

归档补充：8卡原始trace共约16MiB亦已同步本地group/traces。KB refresh成功；全库validate仍为此前5项无关历史错误（4条Nov7缺失链接、1条Nov13技能副本frontmatter），本次结果公式/150条校验与输出检查已通过。


### fastokens 两组追加执行

用户要求已加入 PLAN.md 及 execution.json 的 goal 追加必做项。fastokens0.3.6 在 E0/E1 均已安装。150条原始会话含tools及chat_template_kwargs的模板、tokenIDs、解码一致性全部通过。启用 VLLM_USE_FASTOKENS=1、DYN_TOKENIZER=fastokens、DYN_TOKENIZER_FALLBACK=false；新入口记录首次真实backend/encode调用，非仅环境开关。r1在发请求之前因审计入口缺少__main__保护导致spawn重入而失败；已修复并保留失败证据。GC r2 worker473254/frontend473253，顺序控制器473249，完成后自动执行Humming r1。两组各150条，0.45QPS，仅与各自未开fastokens组作增量对照。


### fastokens + GC 完成

P-FAST-GC-ALL-r2：150/150成功、0失败，376.328秒，479.499output token/s，P95 TTFT14739.534ms、TPOT25.4127ms/token、E2E58313.978ms。全部结构代理检查通过（173个DSML调用）。八卡短trace确认原生GC和其他三项路径，fastokens实际backend/encode调用在前端及worker记录。前端报告fastokens模式会关闭MM-aware KV路由，图像token指标仍启用，此差异需披露。随机采样输出长度不同，单轮改善不能当作已隔离的纯分词加速因果收益。Humming组已启动，frontend480353/worker480354。


### Goal 追加：fastokens 两组测试完成（2026-11-16）

前150条原顺序、0.45QPS、原temperature0.8/top_p0.95，发完排空；每组150/150成功、0失败。开启fastokens0.3.6（Python vLLM及Dynamo后端），关闭自动回退。150条原始文本模板/token IDs/decode一致性通过；多模态仍由vLLM原渲染路径处理，fastokens会使Dynamo关闭MM-aware KV路由，此为开关伴随差异。两组均已完成实际backend/encode调用审计。

|组|指标|P50|P90|P95|P99|
|---|---|---:|---:|---:|---:|
|GC|TTFT(s)|1.942|12.639|14.740|16.693|
|GC|TPOT(ms/token)|12.907|22.676|25.413|32.881|
|GC|E2E(s)|13.086|38.162|58.314|105.115|
|Humming|TTFT(s)|0.747|7.925|10.263|12.490|
|Humming|TPOT(ms/token)|5.751|8.594|9.984|17.458|
|Humming|E2E(s)|6.392|19.728|37.114|60.074|

GC：376.328秒，180449输出tokens，479.499token/s；P95 TTFT/TPOT/E2E相对自身未开启fastokens组分别-2.17%/-22.70%/-33.89%。Humming：341.846秒，572.819token/s；P95分别-2.21%/-5.68%/+9.11%。因此Humming的E2E尾延迟没有随TTFT/TPOT一起改善。两组单轮随机采样输出长度不同，不能把变化全部归因于fastokens。GC和Humming环境Torch/CUTLASS/NCCL及量化粒度仍不同，不是纯后端因果对照。

质量代理：两组空输出、截断、异常循环后缀、工具JSON异常、DSML不平衡/未知工具/重复参数等标志全部0；GC173个DSML调用、Humming170个。没有业务标准答案，不能宣称业务精度无损。

正式计时后3请求8卡trace：GC每卡原生role kernel2400、finalize800、compact80、headrow1600；Humming每卡W4A8kernel1600、tiled finalize860、compact80、headrow1600。仅代表短诊断路径命中，不是150条全程命中率。Humming沿用原有tiled收尾融合，GC采用scale/shared收尾融合，未混称两者。原始结果、源码、脚本、日志和8卡trace均已本地归档。服务保留Humming frontend480353/worker480354，8000端口。

本次新增goal执行项均已完成；系统goal保持active，既有更广范围的未完成项没有被误标完成。


### 同E1环境服务对照完成：P-E1-FAST-HUM-ALL-r1

前150条、0.45QPS、原temperature0.8/top_p0.95，150/150成功，0失败。375.790秒、202071输出tokens、537.723token/s。E1 Torch2.11/Triton3.7.1/CUTLASS4.8.1/NCCL2.28.9，fastokens0.3.6，前端与已有GC/Humming组一致。E1实际版本保存在e1-effective-package-versions.json。

|指标|P50|P90|P95|P99|
|---|---:|---:|---:|---:|
|TTFT(s)|0.721|7.556|9.996|12.429|
|TPOT(ms/token)|5.382|8.497|9.647|14.251|
|E2E(s)|5.816|19.952|34.391|62.307|

同E1环境下，相比fastokens+GC四项组合，Humming的P95 TTFT/TPOT/E2E约低32.2%/62.0%/41.0%。Humming在E1和E0的P95相近。因此GC此前延迟差距不能仅由Torch/CUTLASS/NCCL环境差异解释；尚不能归因为纯GC kernel问题，因为量化粒度、输入分片/输出gather、收尾融合及其他适配路径不同，随机采样输出长度也不同。该对照补齐完整服务的同环境比较，不证明纯内核因果收益。

质量结构代理全部0异常，166次DSML调用；无业务标签。E1 finalizer八卡1536个eager/Graph算子对照通过。正式计时后3请求trace在8rank均观察到W4A8kernel1600、tiled finalizer860、compact80、headrow1600，不代表整轮150条命中率。

Goal完成审计：P0基线、P1 GC正确性/生命周期、P4同环境服务对照已有证据；P2仍缺Humming完整scale/shared收尾融合的服务接入（独立算子测试不代替服务），P5仍缺最终候选稳定性判定。按用户指示，单项冗余E2E已跳过，不重新增加。当前没有满足所有稳定性门槛的已验证获胜方案，不能将goal标记complete。服务保留E1 Humming frontend486402/worker486403，端口8000。


### Humming 完整收尾融合接入

新增默认关闭的 dsk_humming_full_finalize.py 与 Worker，开关DSK_HUMMING_FULL_FINALIZE=1。限定TP8/EP8、无DP/SP/DBO/输入输出变换、未归约HummingGroupedExperts、BF16和1..128行；其他路径保持原实现。shared输出可能异步计算，模块在最终unpermute前等待事件，读取私有_output保留缓存，并在原runner边界消费一次，跳过后续重复scale/add；EP最终allreduce位置不变。

93脚本使用真实MoERunner方法和native unpermute作对照（专家GEMM输出为合成数据，不冒充完整模型验证），8rank×5行数×2异步模式×2scale=160个配置，1600次变输入Graph检查，原生/融合BF16和allreduce后逐位一致；129行原生回退也覆盖。退出0。

真实服务P-E1-FAST-HUM-FULL-r1已启动，八卡各40个主模型runner均记录完整融合分支，全部shared独立流等待正确。实际runner scale=1.0，因此该模型实际融合的是unpermute与shared加法，不存在可宣称省去的独立scale kernel。0.45QPS前150条正在运行，尚不能标记性能/稳定性完成。


### 完整收尾融合真实服务验证完成

P-E1-FAST-HUM-FULL-r1：150/150成功、0失败，346.624秒，549.148输出token/s。固定前150条0.45QPS、原随机采样配置。

|指标|P50|P90|P95|P99|
|---|---:|---:|---:|---:|
|TTFT(s)|0.748|7.731|10.266|12.403|
|TPOT(ms/token)|5.405|8.892|10.947|15.270|
|E2E(s)|6.608|19.512|28.556|61.602|

相比同E1环境、未开启完整收尾融合的Humming组：P95 TTFT+2.71%、TPOT+13.48%、E2E-16.97%；单轮随机输出长度不同，不可认定稳定的全面收益。结构代理检查全部0告警，173次DSML调用。

8卡×40层首次真实分支审计确认full finalize生效，实际routed scale全为1.0。三请求20步独立trace：每rank独立BF16加法860→60，减少800=40主模型层×20步；W4A8kernel1600、grouped finalize860、compact80、headrow1600均命中。证明shared加法被融合；不能宣称在当前模型消除了原本不存在的独立scale kernel。此短诊断不是整轮150条命中率。

P2完整收尾服务接入的缺口已关闭（限定安全形状1..128行，其他形状回退原路径），真实模型Graph及正式回放成功。Goal仍active：最终候选稳定性复验、收尾可回退交付审计尚未完成，不将单轮结果当获胜方案。当前保留E1完整融合服务frontend493036/worker493037，8000端口。


### 最终候选复验及长稳态启动

P-E1-FAST-HUM-FULL-r2完成150/150，0失败，349.691秒，521.032output token/s；P95 TTFT10324.944ms、TPOT10.042856ms/token、E2E30183.764ms。结构代理通过，170个DSML调用，八卡短trace再次验证每卡省800次独立shared add。

两轮P95中位数TTFT10.296秒、TPOT10.495ms/token、E2E29.370秒。3000次成对请求重采样（seed20261116）仅量化请求样本差异，不是跨独立运行的置信区间：相对A1平均P95差TTFT区间[-1.507,-0.982]秒，TPOT[-5.779,-1.030]ms/token，E2E[-13.223,+7.842]秒；E2E区间跨零，不能宣称稳健E2E收益。E0/E1及随机输出长度差异仍限制归因。完整统计在results/final-repeat-statistics.json。

默认关闭审计通过：导入GC/full Humming模块不修改MoERunner/分组finalizer；3个冻结主环境源码SHA未变。handoff/native-off包含显式关闭所有优化和fastokens的E0恢复脚本，bash语法检查通过，实际恢复尚未执行。

103脚本PID503556已启动长稳态：先至少30分钟，再至少2小时；仅循环前150条，每批0.45QPS，发完150条排空，每批检查质量，保留热缓存且单独报告，绝不混入冷A/B。结束当前150条后检查阶段最小时长，可能比目标略长。出现请求失败/新服务ERROR/进程退出或质量代理异常即停止后续阶段。当前第1批150/150及质量通过，第2批运行中。GPU2约81°C，当时热降频标志不活跃但有累计历史SW热降频计数；105侧车PID504043采集之后的计数增量，不能用累计值归因历史某一轮。

整体goal仍active，长稳态完成和实际主环境恢复尚未验证。
