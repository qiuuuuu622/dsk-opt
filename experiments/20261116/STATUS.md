# 最新状态（2026-11-17）

正式批次已结束。长稳测试于 2026-11-16 按用户要求停止，30 分钟及 2 小时阶段均未完成，原生服务最终恢复验证未执行；随后用户已清除 goal。历史 README、STAGE2、execution.json 中“正在运行”只表示当时快照，本文件与 stability/full-final-candidate/state.json 为最终状态。

2026-11-17 SSH 核实实验记录仍在 `/root/dsk_opt_goal_20261116`（约 400 MB）；`/root/gc_moe_isolated_20261116` 已不存在，需重建环境方可复跑。

公开导出移除了响应正文等字段，原业务请求 ID 一致哈希化。没有上传原始业务 ZIP、权重、环境二进制、服务全文日志或原始 profiler trace。每组 profile-diagnostic/four-paths/full-fusion-proof 等解析证据已保留。EXPORT-MANIFEST 同时保留来源文件和发布文件 SHA256。原始完整证据保存在实验主机和本地知识库。性能统计可由公开逐请求指标重新计算；业务输出质量检查无法仅凭公开版重跑。
