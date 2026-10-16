# 新增数据集配额构建（specific 分支）

这版在调用 API **之前**控制抽样和题型配额，不做“全量生成后丢弃”。仅生成经过原有质检的 QA；配额是严格上限，质量或预算不够时允许不足并报告原因。

## 固定计划

| 数据集参数 | 候选视频上限 | HGI | AD | ETP | 总计 |
|---|---:|---:|---:|---:|---:|
| `bdd100k` | 50 | 19 | 18 | 13 | 50 |
| `opendv-youtube` | 352 | 1,545 | 1,415 | 1,040 | 4,000 |
| `boreas` | 125 | 290 | 265 | 195 | 750 |
| 新生成合计 | 527 | 1,854 | 1,698 | 1,248 | 4,800 |

`configs/specific_targets.json` 记录完整六数据集计划。PIE/nuScenes/KITTI-360 的800/50/350条从已有 QA 中另行抽样，**本入口不会重建它们、也不会自动改写旧发布版**。全部目标相加为6,000条（HGI2,318 / AD2,122 / ETP1,560）。

视频目录多于表中数量时，按固定 seed 抽取指定数量；不足时使用实际数量，并打印 found/selected。`--video-limit` 可覆盖。表中时长仅作参考，不通过裁剪强制匹配时长。应提供正确的前视视频，不会自行从其他传感器/摄像头中辨认前视画面。候选视频数、实际扫描视频数和有合格 QA 的贡献视频数分别统计；**不保证每视频有题**。

## 安装与提交 ACP

```bash
git clone --branch specific git@github.com:HelloWorld416/QA-gen-pipeline.git
cd QA-gen-pipeline
pip install -e .
export OPENAI_API_KEY='YOUR_API_KEY'
export OPENAI_MODEL='gpt-6.1-sol'
export DATASET='opendv-youtube'
export VIDEO_DIR='/path/to/OpenDV/videos'
# 按需：export OPENAI_BASE_URL='YOUR_RESPONSES_ENDPOINT/v1'

# 先查看视频数、候选窗口数、配额与预算；不消耗 API token
bash scripts/run_specific_acp.sh --dry-run

# ACP 中执行；并发默认4，low推理
bash scripts/run_specific_acp.sh
```

BDD100K/Boreas 分别设 `DATASET=bdd100k` / `DATASET=boreas` 并修改 `VIDEO_DIR`。三个数据集默认使用不同输出目录 `outputs/specific/<dataset>`，可分别提交；如同时运行，合计并发是各任务并发之和。`.env` 若存在会被脚本加载并覆盖同名 shell 环境变量，检查旧 `.env` 的 `OUTPUT_DIR`，避免三个任务共用输出。

等价 Python 入口（不依赖可执行命令是否重新安装）：

```bash
python -m streamdrivebench.specific --dataset boreas \
  --video-dir /path/to/Boreas/videos --model gpt-6.1-sol
```

旧 `streamdrivebench` / `scripts/run_acp.sh` 仍是扫描所有窗口的通用入口。**要约束数量，请使用新 `specific` 入口。**

## 调度、配额与复用

1. 本地探测所有视频的时长，列出30秒不重叠窗口。末尾不足30秒也可列入，少于2个整数秒观测的窗口没有可确认转变，提前跳过。计划阶段不调用 API。
2. seed 默认 `20261016`：视频顺序随机、每视频窗口顺序随机，以轮换视频的方式抽样，避免先扫完长视频。当前按视频/时间抽样，没有天气或事件类型标签分层；随机抽样不能保证语义均衡。
3. 每轮最多24个窗口（`--batch-size`），最多4条并发链（`--workers`）。一批内每视频最多1窗，避免检查点并发写入冲突。每个窗口**最多1个事件**，先独立确认再出题。
4. 调度前为每个窗口的每个缺额题型预留1个名额：`已通过 + 在途预留 ≤ 目标`。预留及窗口计划持久化。共享出题请求只包含尚缺题型，达到配额的题型不再出题、不再核验。ETP尾部不足完整未来30秒时，不预留、不向作者要求ETP。
5. 同事件尽量一起生成各缺额题型；HGI/AD 不依赖 ETP 是否通过。每题通过原有消融、因果、关联与触发边界检查才计数。拒绝后释放名额，继续从未扫描窗口补足，**不降低标准凑数**。
6. 每批合并已完成窗口并重新平衡选项；HGI/AD 按题型×选项数全局平衡，ETP 的 SILENT 始终末位。输入、标签、映射一起更新。每批可能改变旧题的答案字母，评测应使用同一次完整发布。
7. 同一输出续跑会跳过已完成窗口，未完成窗口复用成功 API 请求。达到目标后重跑不再调用 API；不扫描剩余窗口。已扫描窗口不为了补某个题型而二次生成。

`summary.json` 记录逐题型目标/已通过/缺额、实际贡献视频数、已分配/完成窗口数、已请求窗口的合格率、按该合格率估算的补充窗口数。估算只用于进度参考，尤其小样本或0通过时不能视作保证（0通过用null）。实际补充始终受剩余名额和上限约束。

完整三元组数量受三个题型不等配额影响，不会把不支持的题强行组成三元组。`queries_strict_triplets.json` 仅包含同一原子事件且三题都通过的严格三元组。

## 历史不强制满30秒

作者为每题生成 `history_start_second`，在发现窗内选择问题需要的历史：

```
max(discovery_start, query_second - 30) <= history_start_second <= query_second
```

例如查询时刻18秒，可用历史 `[14,18]`，不必补齐 `[0,18]`，更不必向前凑满30秒。HGI必须含更早证据，且通过“当前帧不足以回答”与“历史必要”核验；AD/ETP在证据充分时可只用当前帧。时间精度沿用1秒网格。

选定起点会同步用于 HGI历史核验、AD前缀核验、ETP注入和触发确认、发布的 `input_policy.history_start_second`。历史较长时沿用每3秒+最近5秒的前缀抽帧；作者不能依赖未实际提供给核验器的帧。

**历史长度与未来监控不同**：AD仍预测3秒后；ETP仍从注入向后监控完整30秒，可跨发现窗边界。ETP触发核验保留所选注入历史及注入以来的观测。不会把短历史解释成短监控，更不会因视频提前结束生成虚假的SILENT负例。

## 预算与停止行为

```bash
# 自定义配额，小规模试跑使用独立输出
bash scripts/run_specific_acp.sh --targets 10 9 6 \
  --output outputs/specific/pilot --max-api-attempts 300 --max-windows 80

# 在服务商已报告总tokens达到100万后停止新增请求
bash scripts/run_specific_acp.sh --max-reported-tokens 1000000
```

- `--max-api-attempts` 是**整个输出目录跨续跑累计的硬调用次数上限**，包括429/网络重试；缓存命中不占额度。默认 BDD=1,000，OpenDV=60,000，Boreas=12,000；这些是安全停止上限，**不是预估用量或保证足够的预算**，可先设置更小值试跑。每次请求发出前先落盘预留；崩溃时已预留调用也计数，采用保守计数。
- `--max-windows` 是累计分配窗口上限，默认 `min(候选窗口数, 3 × 最大题型目标数)`，无需扫描全部素材。
- `--max-reported-tokens` 默认0关闭，是**软停止阈值，不是账单硬上限**：并发在途请求可能超过阈值，超时未返回的费用也无法统计。若必须限制费用，还应使用服务端硬额度。
- 达到任一预算/窗口上限，或没有可用候选时，保留有效结果，明确报告缺额，进程退出码2。真实错误非零退出；成功达到全部配额才退出0。API途中失败不会把尚未完成全部核验的窗口算入发布，成功响应仍缓存。
- 可提高调用/窗口/已报告token阈值后用**同一输出目录**继续；预算计数不会清零。workers/batch-size也可调整。目标、模型、视频清单/内容、seed、代码不允许混改，否则需新输出目录。
- `api_budget.json` 是持久化限额账本；`token_usage.json` 汇总服务商响应；`window_assignments.json` 是抽样与预留记录；`build_plan.json` 是冻结的可复现计划。不要删除账本来绕过预算后仍声称原预算有效。

所有视频的1fps帧按需在首次选中该视频时提取，该视频仍需本地完整解码；节省的是 API 扫描/出题/核验请求，不宣称已消除本地解码或存储成本。配额入口要求可靠的容器时长；时长缺失、非正值或与解码网格不一致时明确拒绝，需先整理源视频，不猜测窗口。

每批 `release/` 是自动质检通过的当前子集，并非人工真值；仅 `summary.json` 的 `build_state=targets_met` 表示已满足全部数量要求。预算不足时仍有完整格式的部分发布，不能误称达到计划。发布目录帧为符号链接，移动/分发时保留实际帧源。

## 验证范围

离线测试覆盖按比例硬配额、多任务预留与补充、无效/短尾窗口、可变历史、完整ETP监控、余额跨重启、预算停止及恢复、无重复调用的完成任务续跑、输入冻结、发布平衡。测试使用模拟 API 和合成视频，不消耗远程 token，不替代新数据集真实 QA 的人工质量审核。
