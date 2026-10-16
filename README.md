# StreamDriveBench QA Generation Pipeline

从任意长度的**前视驾驶视频**构建由事件串联的三类问题：HGI（历史证据理解当前状态）、AD（预测3秒后的状态）、ETP（等待可观察事件发生后及时回答）。这是从现有 PIE/KITTI 工作流整理出的独立实现，不依赖原工程、特定数据集目录或本机模型。

核心顺序：**先发现事件并独立确认，再围绕确认事件生成三类 QA**。每30秒一个不重叠发现窗口，默认最多1个事件；答案经过独立核验，完成后全局重排选项，ETP 的 `SILENT` 永远是最后一项。

## 最快开始

环境：Linux/macOS、Python 3.10+、可联网访问的 Responses-compatible 视觉模型服务。视频解码在 CPU 上执行，不需要本地 GPU/VLM。

```bash
git clone git@github.com:HelloWorld416/QA-gen-pipeline.git
cd QA-gen-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

export OPENAI_API_KEY='YOUR_API_KEY'
streamdrivebench --model gpt-6.1-sol --video-dir /path/to/videos
```

用户只需提供 **API key、模型名、视频目录**。默认继续使用原工程的接口地址 `https://www.hongwangwang.qd.je/v1`，请求发送到其 `/responses`。API key 不写入数据、配置快照、日志或仓库。密钥请使用环境变量；也支持 `--api-key`，但命令行参数可能被进程列表/终端历史记录。

如需换服务商：

```bash
export OPENAI_BASE_URL='https://api.openai.com/v1' # 官方直连示例
streamdrivebench --model YOUR_VISION_MODEL --video-dir /path/to/videos
```

模型必须支持 **图片输入、Responses API、JSON Schema strict 输出、reasoning.effort**。模型名称原样发送，不自动映射或替换；默认推理强度 `low`。只支持 Chat Completions 的服务不能直接使用。第三方同名模型的可用性和计费由服务商决定。

## 扫描与执行

程序会首先递归扫描视频目录并打印视频总数，然后逐视频探测元数据，最后才开始生成请求。支持扩展名 MP4/MOV/MKV/AVI/WebM/M4V/MTS/M2TS（实际可解码性取决于 PyAV/FFmpeg）。不同目录中的同名视频通过相对路径哈希区分；视频 ID 不包含用户本机绝对路径。

```bash
# 仅统计/检查输入，不发送API、不解码整段视频、不写输出
streamdrivebench --model gpt-6.1-sol --video-dir /path/to/videos --dry-run

# 降低并发、独立指定输出
streamdrivebench --model gpt-6.1-sol --video-dir /path/to/videos \
  --workers 2 --output outputs/my_dataset

# 小规模试跑：按相对路径排序的前1段；使用独立输出
streamdrivebench --model gpt-6.1-sol --video-dir /path/to/videos \
  --limit-videos 1 --output outputs/pilot
```

默认同时处理4段视频，最多4个在途请求；每段内部依次处理窗口。**一次运行可能产生大量图片 tokens**，正式全量前建议先用 `--limit-videos 1` 试跑。`--max-calls-per-video 100` 可限制每段每次运行的请求尝试次数（重试也计数），到限即保留断点退出；它不是全局 token 或金额限额，默认0表示不设调用次数上限。

## 任意时长如何处理

不要求视频恰好5分钟，不预先裁剪视频：

- 发现窗口为 `[0,30)、[30,60)…`，末尾不足30秒也扫描。
- 解码后以首个有效视频 PTS 为0，按秒取“不晚于该秒的最新帧”。支持非零起始PTS、常见可变帧率、容器缺少时长/帧数的情况。每次仅保留当前解码帧，元数据与提取图片随时长增加。
- 910×512 等比例填黑边，1fps，不扭曲原图比例。取到的源帧若比该秒陈旧超过0.5秒，标为不可用，避免用陈旧图像伪造时间证据；低帧率/大时间戳断层可能导致窗口被跳过。
- HGI/AD 的历史从发现窗口起点开始。AD查询和3秒后的标签都必须在本发现窗口内。
- ETP可从查询时刻向后跨发现窗口监控完整30秒（31个整数秒观测）；视频尾部不足完整监控期时跳过ETP，**不截短成负例**。
- 很短的视频可能没有事件或合格QA，合法输出为0条。不存在“任何视频都必须产出三类题”的保证。
- 无法解码、无PTS、非单调PTS会明确失败，不猜测时间。视频内容应为驾驶场景；这里“任意”指时长和文件组织，不是任意领域语义。

## 生成与质控

1. **发现事件**：当前窗口仅发现候选，不同时出题。
2. **确认事件**：独立核验目标、行为、前后状态、相邻秒级边界；不确定事件拒绝。
3. **共享图像出题**：所有确认事件通过后，再一次请求生成各事件的HGI/AD/ETP。每事件每类最多1题，无法生成某类必须说明原因；`--max-events-per-window` 可改变上限，默认1。
4. **前移检查**：查询范围、选项、AD时间跨度、ETP完整监控、已知硬编码/超时答案格式错误先检查，减少无效调用。
5. **独立核验**：HGI当前单帧消融与历史必要性；AD因果前缀合法性和独立未来标签；ETP注入合法性和完整监控。
6. **事件关联**：每题验证同一目标、同一连续行为及答案/选项覆盖；ETP正例分别检查触发前一秒应沉默、触发秒可以作答。
7. **发布与统计**：将全部完成视频合并；HGI/AD按“题型×选项数”全局平衡正确位置，ETP固定SILENT最后，正例答案在其余位置平衡；输出分类统计及映射记录。

ETP问题只允许一个原子触发条件，无触发时始终SILENT，包括监控结束时；不可要求超时报告“未发生”，不可把失去可见性作为第二触发，不应在题干硬编码答案文本。关键词防护不能覆盖所有语义错误，仍需人工复核。

**因果边界与限制**：作者看到整个发现窗和已确认事件，属于离线、知道结果的出题。后续核验请求只看其合法时间范围，但不能完全消除作者的未来选择/措辞偏差。模型核验不是人工真值；发布标记 `human_approved=false`。固定SILENT末位是评测协议，不代表正负例均衡。每窗最多1事件也不代表事件发现完整。

## 输出

默认 `outputs/streamdrivebench/`：

```text
build_plan.json           # 冻结输入、源码、模型与参数；不含密钥
status.json               # 总进度、活跃视频、失败原因
summary.json              # 完成后 HGI/AD/ETP 总数量及逐视频数量
token_usage.json          # 服务商已报告的输入/输出/总token
videos/<video-id>/        # 提取帧、事件、候选、拒绝原因、逐窗检查点
api_cache/                # 成功响应和请求使用量，供续跑复用
release/
  queries_all.json        # 完整数据（含 evaluator_only 私有标签）
  model_inputs.json       # VLM公开输入，不含事件/答案标签
  labels.json             # query_id -> 私有评测标签
  option_mapping_audit.json
  queries_strict_triplets.json
  events.json / event_groups.json
  episodes_metadata.json
  source_1fps/            # 媒体链接；保留生成目录
  summary.json / release_manifest.json
```

运行结束打印，例如 `FINAL_QA_COUNTS {"HGI": ..., "AD": ..., "ETP": ...} total=...`。发布只在所有选定视频完成后生成，发生错误不把部分结果冒充全量结果。每窗完成后保存检查点，成功API响应单独缓存；已完成窗口续跑零API，失败窗口复用成功请求。

同一命令同一输出目录可续跑。文件锁阻止两个任务写同一输出；源码、视频内容/清单、模型或生成参数变化时拒绝混用旧结果，需指定新输出目录。仅改变并发允许续跑。未完成时停止新视频派发，让已在执行的任务安全结束。HTTP 429/5xx/暂时网络错误最多重试5次，之后保留断点退出。

`release/source_1fps` 使用符号链接，移动/分发数据集时需要连同源提取帧一起迁移并更新路径；直接复制 JSON 不是完整可移植数据集。默认不自动划分train/val/test，同一事件及视频的样本应在后续划分中保持同组。

## ACP 任务

```bash
cp .env.example .env
# 编辑 .env：填入API key、模型、VIDEO_DIR，按需设置OUTPUT_DIR与WORKERS
pip install -e .
bash scripts/run_acp.sh
```

脚本使用当前Python环境，或通过 `PIPELINE_PYTHON=/path/to/python` 指定。可以追加 `--dry-run`、`--limit-videos 1` 等参数。`.env`、视频、输出、缓存及密钥均被Git忽略。不要把真实密钥写进可提交脚本。

## 离线测试

```bash
pip install -e '.[test]'
pytest -q
```

测试使用合成短视频和模拟API，不花费远程token，覆盖事件先确认后出题、时间边界、三类核验、断点复用、错误停止、媒体PTS处理、选项平衡、最终统计及Responses格式。CI只运行离线测试，不包含凭据。真实模型的标签准确率和API兼容性仍需用户自己的小规模试跑验证。
