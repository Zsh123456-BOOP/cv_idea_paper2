# revrender

> 第一次来先读 [`OVERVIEW.md`](OVERVIEW.md)（课题、数据、评测、试过什么、现在卡在哪）；给外部评审的简报和提示词在 [`REVIEW_BRIEF.md`](REVIEW_BRIEF.md)；每个实验的关键数字在 [`RESULTS.md`](RESULTS.md)。

本仓库有两部分：第一篇 RevRender（检出涂改）的干净移植，已在服务器上证明与旧实现等价（下面的结构、命令、出处表、等价验证都是这一部分）；第二篇（读出被划掉的内容）的代码，入口见「第二篇的入口」，实验叙述在 `OVERVIEW.md` / `RESULTS.md`。

## 结构

```
revrender/
  util.py            摘要、create-only 写入、冻结路径守卫、git 状态、run manifest
  config.py          读 configs/*.toml（路径、配方）
  determinism.py     严格确定性：配置 + 前向不变、反向确定的三个算子
  models/            RevRender：backbone.py（ResNet-50 + 细节通路）、detector.py（查询检测器）、renderer.py（框内逐点渲染器
                     + 上下文残差）、quality.py（掩膜质量头）、revrender.py（组装 + 四个变体）、losses.py（损失 + 匹配 + 分组裁剪）、
                     ops.py（采样与框）、legacy_init.py（只为复现随机数流而建、随后丢弃的模块）、
                     vlm_tokens.py（Qwen3-VL 的视觉 token 屏蔽：隐藏的 token 保留位置，自己的贪心解码；补丁嵌入改成矩阵乘）
  data/              bundle.py（数据包读取 + 训练池 + 测试集守卫）、extra.py（用户校正行）、sampling.py（平衡采样 + 预取）、
                     augment.py（增广）、gold.py（金标栅格化 / letterbox 与它的逆 / 笔宽校正）、lines.py（行目录）、
                     ocr_lines.py（整页 OCR 文本 → 目录里的那些行）、handwriting.py（整页的手写墨迹：印刷模板、答题框、按墨迹切多行区域、给没被框的手写补框）、
                     page_gold.py（第一篇的金标放回整页坐标，写成与整页输出同一布局）、clean_pages.py（整页清洁：只留本题自己的虚线框、去杂点）、
                     prior_annotations.py（旧项目已有的人工前后对、盲判无涂改行、三个模型的行转写，按行 id 合并）、
                     line_pack.py（人工标注行的图包：原图 / 擦除 / 涂白列带 / 红色叠加 + 掩膜框）、
                     annotation_set.py（第二篇要人工标注的整道题：按第一篇的书写者划分挑选）、page_labels.py（读回几何版的涂改标注：每笔要去掉的像素、每处的掩膜和作废区域、去掉涂改后的扫描图）、
                     strike_bank.py（从第一篇金标里抠出训练书写者的真实涂改：划线、涂黑）、
                     paste_strikes.py（干净行上的合成涂改：七种划法或真实涂改、贴进去的被划掉的字、整行划掉；目标文字和要去掉的像素由构造已知）、
                     reader_corpus.py（合成涂改的配方：宿主行的挑选、与测试 / 评测书写者隔开、写成文字文件）、
                     journal.py（写作者子集、合成阳性）
  train.py           训练循环：smoke → 精确重放 → 仅训练集学习门槛 → formal（早停、挑步）
  infer.py           推理与后处理：packed 预测存档、deliver（掩膜阈值 / NMS / 分数线）、predict_lines（把冻结点用到一批灰度行上）、命令行
  ocr.py             Unlimited-OCR 的调用（切行的那个模型，设置与当年相同）
  pages.py           整道题：OCR 的行 + 按墨迹切开的多行区域 + 补切的手写 → 每块过 RevRender → 实例拼回整页；命令行；`write_page` / `read_output` 是整页输出的唯一读写；`--frames` 丢掉框外的块
  tighten.py         掩膜收紧到非白像素（墨迹 + 紧贴的软边）：整页输出和放回整页的金标都能收；命令行
  recognize_lines.py 图包里的行或涂改区域用一个 VLM 在若干条件下读（vLLM；`--backend hf` 加 `--adapter`、置信度、遇到重复就停）
  train_reader.py    给 Qwen3-VL 训练 LoRA（语言模型的线性层）：第二篇的读者（读被划内容，或说读不出）
  router.py          读数路由：每个来源一个评判器，每处取最可信的读数
  eval/              coverage.py（人工标的整行前后文字 → 划掉的 / 重写的 / 仍有效的字符，读数持有几成）、leakage.py（泄漏 / 有效 / 重写 / 空 / 对照的评分与参照）、pages_score.py（整页输出对放回整页的金标：找到 / 画对）、matching.py（组件匹配 + F1 + 前沿 + 误报预算）、ap.py（掩膜 AP）、bootstrap.py（按书写者自助法）、
                     freeze.py（阈值冻结；`read_frozen` 是各读数共用的「读一份冻结」）、fates.py（找到 / 画对）、speed.py（推理计速）、
                     inloop.py（训练中验证读数）、report.py（重新打分、与已记录数字和另一份存档比对）、
                     ink.py（只算墨迹的视图：金标和交付掩膜都裁到墨迹；fates、ap、bootstrap 的 `--ink-only`）、
                     representation.py（掩膜网格的上限、掩膜加粗 / 变细后的读数）、pixel_errors.py（错误像素按六类统计）
  baselines/         native/（七个原生实例分割基线 + Mask Transfiner：factory、train、threshold、transfiner、dense、metrics、compat）
                     component_cls.py、removal.py、paper.py、mathscrub.py（涂改专用方法）、common.py
  tools/             annotate/（第一篇的逐行标注页 editor.html + 单文件打包；第二篇的整道题涂改标注页 page_editor.html + page_rules.js、框的起点 page_prefill.py、打包 page_pack.py）、compare_trace.py / compare_curve.py / compare_checkpoint.py（两份训练轨迹 / 验证曲线 / 检查点张量逐项比对）、
                     pages_view.py + pages_view.html（整页结果的静态 HTML：识别完拼回的、被字数过滤掉的）、
                     tighten_view.py + tighten_view.html（掩膜收紧前后的静态对比页：模型的和金标的）、
                     clean_view.py + clean_view.html（整页清洁前后的静态对比页）、
                     reader_view.py + reader_view.html（有标注的行：原图、涂白图、划掉前后的标注、每个运行的读数和它的泄漏 / 有效 / 重写占比）
scripts/             deploy.sh、launch_gpu_job.sh、launch_queue_runner.sh、launch_train_arm.sh、launch_native_arm.sh、
                     verify_equivalence.sh、verify_thresholds.sh、verify_analysis.sh、_common.sh
docs/                data_location.md（题目图在哪、怎么对上 image_id：只读调查）
configs/             revrender.toml（RevRender 配方）、baselines.toml、paths.example.toml
tests/               本地全过；torch 相关测试在没有 torch 时自动跳过，服务器上跑全部
```

命令行入口都是 `python -m revrender.<模块>`（有 `--help`）；`pip install -e .` 后有 `rvr-*` 别名（见 `pyproject.toml`）。服务器上不装包，用
`PYTHONPATH=<快照目录> /home/zsh/hme/envs/hme/bin/python -m revrender.xxx`。

## 怎么跑

```bash
python3 -m pytest -q                      # 本地（无 torch：torch 相关测试跳过）
bash scripts/deploy.sh                    # 部署当前提交到 /home/zsh/hme/code_paper2_<提交号>（要求工作区干净）
```

服务器规则：产物写 `/home/zsh/hme/outputs/paper2_*`；作业经 `scripts/_common.sh` 持 GPU 锁（共享锁），每卡最多两个我们的作业，3 号卡不用（有人在用）；不杀他人进程；不往公用环境装包。

| 做什么 | 命令 |
|---|---|
| 推理（验证集存档） | `launch_gpu_job.sh GPU NUMA OUT -- python -m revrender.infer --checkpoint best.pt --bundle B --bundle-sha S --output-dir OUT` |
| 重新打分 / 等价比对 | `python -m revrender.eval.report --bundle B --bundle-sha S --freeze rvr=freeze.json [--archive rvr=新存档] [--compare rvr=旧存档] [--reference-ap ap.json] [--reference-table table.json]` |
| 训练一个 RevRender | `launch_train_arm.sh control GPU NUMA B S C CS OUT`（smoke_a → smoke_b 重放 → 学习门槛 → formal → 冻结）；`STOP_AFTER=learning` 只跑三道门槛 |
| 冻结 | `python -m revrender.eval.freeze --bundle B --bundle-sha S --run OUT/formal --output freeze.json`（原生基线加 `--model maskdino` 等） |
| 掩膜 AP / 找到画对 / 自助法 / 计速 | `python -m revrender.eval.ap / .fates / .bootstrap / .speed`；三者加 `--ink-only` 就只在墨迹上读（金标和交付掩膜都裁到墨迹） |
| 看整页结果（HTML） | `python -m revrender.tools.pages_view --pages-output OUT --images 图片目录 --answer-text qwen_shard0.jsonl [...] --tags tags.json --min-chars 10 --output-dir SITE`：学生手写解答转写少于 `--min-chars` 个字的页被过滤；`SITE/kept.html` 是识别完并拼回整页的，`SITE/filtered.html` 是被过滤掉的，`SITE/index.html` 是汇总；图片拷进 `SITE/assets`，文件夹拷到哪都能打开（浏览器直接开 html，或 `python3 -m http.server`）。只含学生手写图，别提交 |
| 旧项目已有的人工标注和模型转写 | `python -m revrender.data.prior_annotations [--mapping index_manifest.json] [--same-student 对子.json] --output-dir OUT`：按行 id 合并人工前后对、盲判无涂改行和三个模型的行转写，写 `OUT/pairs.jsonl`（每行一条）和 `OUT/summary.json`；只读旧项目，文字只放本机 |
| 整页清洁 | `python -m revrender.data.clean_pages --pages 图片目录 [...] [--ids 页id文件] --output-dir OUT`：同一道题、同一尺寸的页共用一个虚线框（按该组各页共有的虚线定，不是每页自己的最外两条）；框外的一切（相邻题的虚线和字、页脚、页角黑块）涂白，去杂点和浅灰雾；页的尺寸不变；`OUT/<image_id>.png` 和 `OUT/frames.json`。只用 CPU，3,976 页约 7 分钟 |
| 清洁前后的对比页 | `python -m revrender.tools.clean_view --before 扫描目录 [...] --after 清洁输出 --per-question 30 --output-dir SITE`：每题抽若干页，左扫描（绿框 = 保留的范围）右清洁后；只含学生手写图，别提交 |
| 整道题：切行 → 标涂改 → 拼回 | `python -m revrender.pages --checkpoint best.pt --freeze freeze.json --pages 图片目录 --ocr-model /home/zsh/hme/models/Unlimited-OCR --templates templates.npz --output-dir OUT [--ids 页id文件] [--frames frames.json] [--overlay]`；`templates.npz`（每道题一个印刷模板，用来分出手写墨迹）由 `python -m revrender.data.handwriting --pages 图片目录 --output templates.npz` 生成；已有 OCR 文本时用 `--ocr-text ulocr.jsonl` 代替 `--ocr-model`（不占 OCR 的显存和时间）。每页写 `OUT/<image_id>.json`（切出的行、每个实例的行号 / 分数 / 框）和 `.npz`（每个实例在它的行裁图上的掩膜）。需要 GPU，约 16 GB（OCR 模型 14 GB + RevRender）；在清洁后的页上跑：`--pages` 指清洁输出，`--frames` 给它的 frames.json（框外的块丢掉） |
| 金标放回整页 | `python -m revrender.data.page_gold --bundle B --bundle-sha S --catalogue 行目录.jsonl --pages 图片目录 [...] [--split train --split dev] --output-dir OUT`（默认只验证行；写成与整页输出同一布局，只用 CPU，几秒） |
| 掩膜收紧到非白像素 | `python -m revrender.tighten --pages-output OUT --images 图片目录 [...] [--ids 页id文件] --output-dir NEW`：OUT 是整页输出，或放回整页的金标；每个实例只留页面上的非白像素（墨迹 + 紧贴的软边），没有墨迹的丢掉，`from_instance` 记它来自哪个实例；只用 CPU，3,976 页约 50 秒 |
| 整页读数（对金标） | `python -m revrender.eval.pages_score --pages-output OUT --gold-output GOLD --output scores.json [--ids 页id文件]`：整页输出对放回整页的金标，找到 / 画对 / 多画；收紧前后各读一次就是两边的对比 |
| 收紧前后的对比页 | `python -m revrender.tools.tighten_view --model 前 后 [--gold 前 后] --images 图片目录 [...] [--ids 页id文件] [--scores before.json after.json] --output-dir SITE`：`SITE/compare.html` 每个掩膜一张卡（原图、原掩膜、收紧后、擦除后、只取墨迹擦除后的对照），`SITE/index.html` 是汇总；只含学生手写图，别提交 |
| 掩膜网格上限与加粗变细 / 六类像素错误 | `python -m revrender.eval.representation --bundle B --bundle-sha S --freeze 标签=freeze.json ... --output out.json`；`python -m revrender.eval.pixel_errors ...` |
| 一个原生基线 | `launch_native_arm.sh MODEL GPU NUMA B S C CS OUT`（smoke → formal → 冻结 → 掩膜阈值）；Mask Transfiner：`MODEL=transfiner` 并设 `TRANSFINER_PIN=pin.json` |
| 涂改专用方法 | `python -m revrender.baselines.component_cls / .removal / .paper / .mathscrub`（都有 `--smoke`） |
| 训练轨迹 / 曲线 / 检查点比对 | `python -m revrender.tools.compare_trace 旧/trace.json 新/trace.json`；`python -m revrender.tools.compare_curve 旧/curve.jsonl 新/curve.jsonl`（摘要里的 `same_steps` 说两条曲线的步数是否完全相同）；`python -m revrender.tools.compare_checkpoint 旧/best.pt 新/best.pt`（逐张量） |
| 重做全部等价验证 | `scripts/verify_equivalence.sh OUT_ROOT REF_DIR [probe]`（REF_DIR 放第一篇记录的副本，见脚本头）；约 25 分钟（加 `probe` 另加约 15 分钟） |
| 重做七个原生基线的掩膜阈值冻结并与第一篇比对 | `scripts/verify_thresholds.sh OUT_ROOT REF_DIR`（三张卡并行，每个模型约 12 分钟） |
| 重做三个分析读数并与第一篇比对 | `scripts/verify_analysis.sh OUT_ROOT REF_DIR`（只用 CPU，约 25 分钟；REF_DIR 里另放 `representation.json`、`classes.json`、`ink_only.json`） |
| 标注页 | `python -m revrender.tools.annotate.single_file --package 包目录 --output 页.html` |
| 原生环境 | `baselines/native/environment.sh`（建 `native_baselines_v1`；服务器上已建好，只读使用） |

消融：`--variant control|nofine|nocontext|nodense`（第一篇用到的三个消融）。其余旧脚本里的调参模式、历史分支不搬。

## 第二篇的入口

做法和结果不在这里重复，见 `OVERVIEW.md` 和 `RESULTS.md`；每个编排脚本开头都写着它做什么、事先的判定规则。

| 做什么 | 入口 |
|---|---|
| 读被划内容的读者（当前主线） | `scripts/run_layer_reader.sh`（`PART=human / separator / views / distilled / evidence`，`CARDS="1 2"`）；训练 `revrender.train_reader`（`--pack --fold K --keep-best`、`--counterfactual`），读取 `revrender.recognize_lines`（`--backend hf --loop-limit 4`、`--intervention`），读数路由 `revrender.router` |
| 标注数据的合并、分折、各种给法的图 | `revrender.data.struck_dataset`、`revrender.data.struck_pack`（`folds / fold-sets / separated / views / counterfactual`） |
| 评分 | `revrender.eval.transcription`（AR、CR、全对率、NED、拒读，`--agreement`、`--gone`）、`revrender.eval.selective`（AURC、某覆盖率下的风险）；定义的出处在模块说明里 |
| 恢复网络基线（分离模型） | `revrender.baselines.restoration`，配方 `configs/restoration.toml`（`[finetune_long]`、`[real_only]`、`[prefill]`、`[pseudo]`），统一评测 `scripts/eval_unified_restoration.sh`（候选 `configs/unified_restoration.json`） |
| 标注预填模型 | `scripts/run_prefill_model.sh`（数据 `revrender.data.prefill_set`，评测 `revrender.eval.prefill`，主指标「要改的块数」）；伪标签 `revrender.data.pseudo_regions` |
| 标注页 | 全量库 `revrender.tools.annotate.pixel_library`，训练题分批 `scripts/build_training_batch.sh`，被划文字 `tools.annotate.struck_text`；页面规则见 `CLAUDE.md` |
| 去掉同门数据后的整套重跑 | `scripts/run_labmate_rebuild.sh`（名单守在 `revrender.data.allowed`） |
| 十月的一次性试验 | `scripts/run_annotation_value.sh`、`run_learning_curve.sh`、`run_wider_validation.sh`、`run_real_only.sh`、`run_realistic_synthesis.sh`、`run_whole_question.sh`、`run_pseudo_student.sh`、`run_struck_diagnostics.sh` 等 |

## 数据

原始数据（`~/Desktop/cv_idea/未标注数据`、`标注好数据`）只读，用 `configs/paths.toml` 指过去，不复制。第一篇用的数据包在服务器上：

| 用途 | 位置 | `run_manifest.json` 的 SHA-256 |
|---|---|---|
| 训练 + 验证金标 v2 数据包（476 条验证行、51 名书写者、365 处涂改） | `/home/zsh/hme/work/revision_bundle878_gold_v2/morphology_s42` | `574da74b6444699a4c375077c0aa9ff5819569692ba134a32db50d17da8acddf` |
| 用户校正的 2,127 行 | `/home/zsh/hme/work/revision_corrected861_v1` | `65a2db20b0a4ea563c153b7c1a1b63c42f45e68c65b63df86212201edcd9e429` |
| ImageNet R-50（detectron2 格式） | `/home/zsh/hme/work/native_baselines_weights_v1/R-50-torchvision.pkl` | `16fd03f0b0123e85b959cb9b64aaf9eeb4a7dc68bd9b0b134a9220d4b24c15c6`（文件） |
| 冻结的 RevRender（挑步 19,000） | `/home/zsh/hme/work/revision_tune880_v1/control/formal/best.pt` | `9f8ae601f1db17e1bb2e164d2e63344756bc3719a44a0e47f7e720a43a978153`（文件） |
| 原生基线上游源码 / 权重 | `/home/zsh/hme/work/native_baselines_sources_v1`、`_v2`、`_transfiner_v1`、`native_baselines_weights_v1` | 见各自的 pin / compatibility_manifest |

题目图在哪、`image_id` 怎么对到文件、每个来源的数量、第一篇划分下的书写者数：见 [docs/data_location.md](docs/data_location.md)。

只读入口：`data.bundle.load_bundle`（核对 manifest 与每个文件的哈希、记录与划分映射、没有测试行、没有跨划分的书写者）、`data.extra.extra_rows`（校正行；任何验证 / 测试书写者的行直接拒绝）、`data.lines`（行目录：`image_id`、`box_1000`、题号，以后接整道题用）。数据包的**构建器**（旧脚本里生成数据包的部分）没有搬：第二篇的数据划分另定。

## 约定与红线

- 输出目录已存在就报错，不覆盖；每次运行写 `run_manifest.json`（配置、种子、git 提交、输入文件哈希），评测输出里带 `provenance`。
- 种子只用 42。图像、权重、npz、npy、`id_mapping*` 不进 Git（`.gitignore` + `tests/test_repo_hygiene.py`）；提交前 `git add -An .` 干跑。
- 训练数据永不出现测试集的行：`train_pools` 拒绝验证 / 测试书写者；`tests/test_no_test_rows.py` 抽 60 个批次核对每一行都来自训练书写者。
- 代码之间不用 SHA-256 钉住；关卡证据（smoke、重放、学习门槛）按「配置 + git 提交」的 profile 相等来核对。

## 出处表

旧仓库的编号 → 新模块。「只拿真正用到的」：旧脚本里的历史分支、废弃模式、实验开关没有搬。

**RevRender 模型与训练**

| 旧文件 | 新模块 | 说明 |
|---|---|---|
| 654 | `models/backbone.py`（ConvNormAct、PixelDecoder、SeeDetails）、`models/ops.py`、`models/legacy_init.py` | 654 的 find / draw 在最终模型里被丢弃，但建造它们消耗随机数，保留为 `legacy_init.py` |
| 674 | `models/backbone.py`（MultiScaleSee）、`detector.py`、`renderer.py`、`revrender.py`、`losses.py` | 检测器、框内渲染器、组装、损失 |
| 685、687 | `models/quality.py`、`revrender.py`（forward）、`losses.py`（L_q） | 掩膜质量头与送分规则（几何平均） |
| 750 | `models/backbone.py`（`load_resnet50`、`build_segmenter`） | ResNet-50 的 detectron2 → torchvision 权重转换 |
| 816 | `models/renderer.py`（PointResidual、点特征） | point_context：4×8 固定点阵的交叉注意力 |
| 767、775、865（`off`） | 无代码 | 在默认值下都是透传，最终模型不需要 |
| 772 | `models/losses.py`（批量匹配、`Clipper`、`materialize`）、`data/sampling.py`（`prepare_batch`） | 速度路径；训练一直用它 |
| 668 | `determinism.py` | 严格确定性算子 |
| 675 | `train.py` | 训练循环、三道门槛、早停 |
| 655 | `train.py`、`eval/inloop.py`、`infer.py`、`data/bundle.py`（`make_rows`） | 数据行、检查点读数、packed 存档 |
| 656、483、354、357、442、480 | `eval/inloop.py`（`match_pairs`）、`data/sampling.py`、`data/augment.py`、`models/losses.py`（`dice_bce`） | 只取被调用的函数 |
| 514、833 | `data/bundle.py`、`util.py`（`safe_path`） | 数据包读取 |
| 861、878、853（corrected 路径） | `data/extra.py`、`data/bundle.py`（`excluded_writers`） | 只搬读取侧 |
| 880（control / nofine / nocontext / nodense） | `models/revrender.py`（变体）、`models/losses.py`（nodense） | 其余模式（bb01、wd1e3、lr05、frozenbn、drop01、ssl、nofine_nodense）不搬 |
| 881、launch_881_tune_arm.sh | `train.py` 的命令行、`scripts/launch_train_arm.sh` | |
| 422 | 无代码 | 只用它的种子表做核对，不搬（第一篇表里没有） |

**评测**

| 旧文件 | 新模块 |
|---|---|
| 701、702、703 | `eval/matching.py`（匹配、NMS、前沿、误报预算 41 / 55 / 83 / 111） |
| 872（`iou_matrix`、`greedy`）、889（找到 / 画对） | `eval/fates.py` |
| 883 | `eval/ap.py` |
| 887 | `eval/bootstrap.py`（5,000 次、种子 42） |
| 867、835 | `eval/freeze.py`（RevRender 的 0.3 / 0.4 / 0.5 规则；原生基线在 0.5） |
| 925 | `eval/speed.py` |
| 921（+889 的笔宽与三分位） | `eval/representation.py`（每个模型的网格上限；掩膜加粗 / 变细后的读数） |
| 923 | `eval/pixel_errors.py`（错误像素的六类统计） |
| 926 | `eval/ink.py` 与 `fates.py`、`ap.py`、`bootstrap.py` 的 `--ink-only`（金标与交付掩膜裁到墨迹后重新打分） |
| 547 的事件级读数 | 不搬 |

**基线**

| 旧文件 | 新模块 |
|---|---|
| 514、515、516、519、520 | `baselines/native/`（`factory.py`、`train.py`、`compat.py`、`environment.sh`、`metrics.py`） |
| 793、868 | `baselines/native/train.py`（早停、校正行） |
| 871 | `baselines/native/threshold.py` |
| 912（+911 的 `kept_writers`、`synthetic_positives`） | `baselines/native/dense.py`、`data/journal.py`、`native/train.py` 的选项 |
| 914 | `baselines/native/transfiner.py` |
| 906、907 | `baselines/component_cls.py`、`removal.py` |
| 915（含旧 424 的方法） | `baselines/paper.py` |
| 917（含它用到的旧 517 的函数） | `baselines/mathscrub.py` |
| launch_868_native_arm.sh、launch_journal_native.sh、launch_transfiner.sh | `scripts/launch_native_arm.sh` |
| 518 | 不搬（早期的 Python 队列，已被 shell 队列取代） |

**工具**

| 旧文件 | 新模块 |
|---|---|
| 344（`rasterise`、letterbox）、500（`correct_mask`） | `data/gold.py` |
| exact_crops_46902.jsonl 读取 | `data/lines.py` |
| gold/tools/build_line_edit_review.py（`parse_ulocr`、`reorder`、`mathnorm`）；严格行的规则 | `data/ocr_lines.py`（在 2,238 页上与目录逐行相同） |
| 服务器 annot_v1/annot_transcribe.py（Unlimited-OCR 的调用） | `ocr.py`（设置相同；在验证页上逐字符复现存档的 OCR 文本） |
| 699、782 | `tools/annotate/editor.html`、`single_file.py` |
| launch_gpu_job.sh、launch_queue_runner.sh | `scripts/launch_gpu_job.sh`、`launch_queue_runner.sh`（队列目录改为 `paper2_queue_v1`，不碰旧队列） |

## 等价验证

在服务器 `4090x4`（RTX 4090；torch 2.9.1+cu128，与第一篇训练时同一环境）上，用提交 `14701ec8` 的部署快照（后来在最终代码 `3ca89f94` 上整个重跑，结果相同，见第 4 节末），`scripts/verify_equivalence.sh`
一次跑完下列检查，输出在 `/home/zsh/hme/outputs/paper2_verify_14701ec8/`，全部通过（`summary.log` 末行的 `CHECKS` 全为 true）。数据是第一篇的验证集金标 v2 数据包（476 行、51 名书写者、
365 处涂改）；不读测试集。服务器上的全部测试（120 条）通过，本地 103 条通过（17 条需要 torch 的模型测试跳过）。

### 1. 推理与评测等价

新代码加载第一篇冻结权重（SHA `9f8ae601…`，挑步 19,000），在验证集金标 v2 数据包上推理，按冻结点（掩膜阈值 0.4、NMS 0.5、分数线 0.39311954…）读数：

| | TP | FP | FN | 组件 F1 | AP | AP50 | AP75 | 像素 IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 新代码 | 240 | 70 | 125 | 0.7111 | 0.2578 | 0.6614 | 0.1324 | 0.5795 |
| 第一篇（`freeze_rvr.json`、`ap883.json`） | 240 | 70 | 125 | 0.7111 | 0.2578 | 0.6614 | 0.1324 | 0.5795 |

- 与旧的 `dev_predictions.npz` 逐行比对：476 行同序、32 个查询的分数逐位相同、7 个掩膜阈值的 packed 掩膜逐位相同（0 位不同）。
- 冻结规则在旧运行上重新推导（0.3 / 0.4 / 0.5 取最佳 F1，平局取误报少者、再取低阈值）：掩膜阈值、NMS、分数线、各阈值读数、误报预算下的 TP，与 `freeze.json` 逐项相同。
- 读数与 `ap883.json`（AP、AP50、AP75、无 NMS 的 AP_raw、AP50_raw）、`main_table.json`（F1、TP、FP、像素 IoU）逐项一致，容差 1e-9。

### 2. 训练等价

- 初始权重：see / find / draw 三部分的 SHA 与第一篇运行记录的 `initial_part_sha256` 完全相同。
- smoke 前 4 步（同种子 42、同数据：验证集金标 v2 数据包 + 校正的 2,127 行）：损失 147.40126037597656 → 146.84219360351562 → 145.10865783691406 → 144.49916076660156；
  每步的损失、各损失项、输入批的 SHA、更新后模型状态的 SHA，共 52 项，与旧运行 `smoke_a/trace.json` 逐项相同（`tools/compare_trace`）。smoke_b 精确重放 smoke_a，相同。
- 512 步学习门槛：初始损失 82.31558132171631、最终损失 3.8986156284809113、TP 7 / FP 0，门槛的验证读数与旧运行 `learning/summary.json` 逐项相同。
- 完整重训（`launch_train_arm.sh control`，第一篇的完整配方：28,300 步上限、第 14,150 步起早停、耐心 6；快照 `14701ec8`，输出 `paper2_retrain_14701ec8`）：
  第 22,000 步早停，挑中第 19,000 步，与第一篇相同。`curve.jsonl` 的 23 个验证点（2,000 … 22,000，步数集合相同）的验证网格和损失项与旧运行逐项相同；
  `best.pt` 的 533 个张量名称、顺序、数值逐个相等（文件 SHA 不同：文件里的 profile 带运行信息，以张量为准，`tools/compare_checkpoint`）；
  `freeze.json` 的验证读数、掩膜阈值、NMS、分数线、挑中步数与 `freeze_rvr.json` 逐项相同；新训出的 `dev_predictions.npz` 与旧存档逐行逐位相同，
  读数 TP 240 / FP 70 / FN 125、F1 0.7111、AP 0.2578、AP75 0.1324、像素 IoU 0.5795。用时约 2 小时 40 分（其间同一张卡还跑着其他验证作业）。
  也就是说：旧权重可以由新代码、同一配方、种子 42 从头训出来，逐位一致。

### 3. 评测与基线等价

用新评测代码给 8 个模型各自已冻结的验证集预测重新打分（`eval.report`），TP / FP / FN、F1、像素 IoU、AP、AP50、AP75、AP_raw、AP50_raw
与第一篇表 3 的记录（`main_table.json`、`ap883.json`）逐项相同（容差 1e-9）：

| 模型 | AP | AP50 | AP75 | 组件 F1 | TP / FP | 像素 IoU |
|---|---:|---:|---:|---:|---|---:|
| RevRender | 25.8 | 66.1 | 13.2 | 71.1 | 240 / 70 | 0.5795 |
| Mask R-CNN | 19.0 | 54.6 | 6.8 | 60.1 | 202 / 105 | 0.4912 |
| PointRend | 19.7 | 55.9 | 7.0 | 63.5 | 219 / 106 | 0.5020 |
| SOLOv2 | 17.6 | 51.7 | 7.1 | 59.7 | 206 / 119 | 0.5248 |
| CondInst | 11.3 | 39.6 | 3.1 | 52.9 | 185 / 149 | 0.4845 |
| Mask2Former | 16.4 | 41.1 | 9.7 | 54.6 | 173 / 96 | 0.4077 |
| MaskDINO | 20.3 | 55.9 | 10.2 | 63.1 | 214 / 99 | 0.5269 |
| DI-MaskDINO | 18.2 | 50.0 | 8.6 | 61.1 | 208 / 108 | 0.5288 |

同一次运行里还核对了：

- 找到 / 画对（`eval.fates`）：8 个模型的 matched / extent / missed 计数与第一篇 `fsr_analysis` 记录相同。七个基线找到 72.1%–83.3%、画对 47.4%–60.0%，
  RevRender 找到 80.8%、画对 65.8%（画对 = IoU ≥ 0.5 的占全部涂改；`drawn_right_of_found` 另给出「找到的里面画对的」）。
- 按书写者自助法（`eval.bootstrap`，5,000 次、种子 42）：9 对 × 5 个指标的差值、95% 区间、「未高于 0」份额，共 180 个数，与第一篇 `bootstrap887.json` 完全相同（最大差 0）。
- 掩膜阈值冻结（`baselines.native.threshold`）：**七个原生基线**的冻结检查点各在 0.5 重跑，逐行重现旧存档（476 行掩膜与分数逐位相同），再在 0.3 / 0.4 / 0.5 重读；
  选出的冻结点、三个阈值的读数、误报预算下的 TP，与第一篇记录逐项相同。Mask R-CNN 在主验证脚本里、其余六个由 `scripts/verify_thresholds.sh` 在提交 `aeb945b1` 上重做
  （输出 `/home/zsh/hme/outputs/paper2_verify_thresholds_aeb945b/`；此后只改了推理的变体读取，与它无关）。
- 基线训练入口的 smoke：七个原生基线和 Mask Transfiner 经 `launch_native_arm.sh`（`STOP_AFTER=smoke`）跑通；Mask R-CNN 的头探针（一个合成矩形）的 5 项损失与第一篇记录逐位相同
  （反向梯度范数相对差 ≤ 4e-5，来自 RoIAlign 等 CUDA 原子加）。四个涂改专用方法的 `--smoke` 跑通。
- 推理计速（`eval.speed`，批大小 1、FP32、预热 20 行、每行 3 遍）：RevRender 27,743,113 个参数、显存峰值 541.0 MiB、31.3 ms / 行（第一篇记录 27.7 M、540 MiB、34.0 ms；
  耗时随卡的负载而变）；Mask R-CNN 13.1 ms / 行（第一篇 13.2）。

### 4. 三个分析读数

用提交 `3ca89f94` 的快照在 CPU 上重算八个主模型（RevRender + 七个原生基线），输出在 `/home/zsh/hme/outputs/paper2_verify_analysis_3ca89f94/`（`scripts/verify_analysis.sh`，约 25 分钟），与第一篇记录逐项相同（浮点数逐位相等，不是容差内）：

- `eval.representation`：六种网格的上限（每个金标涂改的 IoU 逐个相同，按笔宽三分位的均值和份额）、笔宽、三分位边界，以及每个模型在加粗 3 / 2 / 1 像素、原样、变细 1 像素下的 TP、FP、F1 和像素读数；原样读数同时重现冻结记录的 TP / FP。
- `eval.pixel_errors`：八个模型 × 六类像素的像素数、错误数、错误率、占全部错误的份额。旧记录里的三个消融变体不搬，没有比。
- 只算墨迹（`eval.fates` / `eval.ap` / `eval.bootstrap` 的 `--ink-only`）：区域读数的 TP / FP / F1、墨迹读数的 TP / FP / F1 / AP / AP50 / AP75、找到 / 画对 / 找到没画对 / 漏掉、金标无墨迹数和交付掩膜无墨迹数，以及 RevRender 减各基线的 F1 差与按书写者的 95% 区间（5,000 次、种子 42）。

另外，把旧的各自一份「读冻结记录」合并成 `eval/freeze.py` 的 `read_frozen` 之后，在最终代码（提交 `3ca89f94`）上把 `scripts/verify_equivalence.sh` 整个重跑了一遍（`/home/zsh/hme/outputs/paper2_verify_3ca89f94/`）：六项 `CHECKS`（推理报告、重新打分、冻结规则重推、找到 / 画对、自助法、阈值重跑）全为 true，服务器上的测试 133 条全过。本地的测试里还有用旧实现录下的金标（`tests/golden/analysis_tools.json`）。

### 5. 整道题：OCR 切行 → RevRender → 拼回

提交 `3bac77cd` 的快照，验证集 205 页（51 名书写者；不碰测试集），`python -m revrender.pages`：

- 每页都真正调用 Unlimited-OCR（`--ocr-model`，两张卡并行，每页约 14–16 秒，RevRender 读一页约 0.3 秒；输出 `paper2_pages_dev_a`、`_b`）：切出的行与目录里的行 205 页全部相同。
- 与用存档 OCR 文本（`--ocr-text`；输出 `paper2_pages_dev_stored`）跑的结果逐页比对：205 页的行（框和文字）与实例（行号、分数、框、掩膜比特）完全相同。模型重新调用的输出就是当年的输出。
- 对第一篇的行级金标（放回整页坐标）：365 处涂改，画对（IoU ≥ 0.5，一对一）236 处（64.7%），找到 81.4%；第一篇在行条坐标下的读数是 240 处（65.8%），差别来自放回整页后重新算 IoU。流水线交付 774 个实例，其中 463 个落在没有金标的行上：金标只覆盖每页约四分之一的行，这些不能当误报。
- 除第一篇 44 名测试书写者之外的 4,628 页（这些书写者的 374 页整个不读），用存档的 OCR 文本跑（`paper2_pages_nontest_stored_ocr`）：43,455 行、14,694 个实例，921 秒，输出 41 MB；12 页没有严格行，1,408 页没有实例。训练书写者的页，结果偏乐观。
- 全量（用户要求，取代上一条的 4,628 页）：不含外部集的全部 3,976 页（主扫描 2,340 + 补充资料包 1,636），包括第一篇测试书写者的 292 页（只做推理），存档的 OCR 文本。
  先按目录当年的「严格行」切（`paper2_pages_unlabeled3976`：37,689 行、12,586 个实例，710 秒），发现漏得不少，改成完整的切法（`paper2_pages_unlabeled3976_complete`，现行）：
  手写墨迹（答题框内、扣掉每道题的印刷模板和图）被盖住的比例：只用 OCR 的行 85.3%（每页中位数 94.0%，1,022 页低于 80%）；加上 OCR 返回的多行区域、按墨迹缺口切成行 94.6%；
  再给没被框的手写补框（漏掉的整行、超出行末的笔画、页边的批注）97.9%。块数 45,073（OCR 的行 37,689，拆出 5,074，补切 2,310），识别出 14,729 处涂改（严格行那次 12,586 处）。
  验证集 171 页（未标注集的部分）上，第一篇行级金标 265 处对上的数不变（171 处，64.5%），多出 90 多处来自新增的块；两次各抽 16 处看过，多数是真涂改（多行区域里的划掉、涂黑），少数是表格线、页边的笔画。
  字数过滤（学生手写解答转写少于 10 个字）：保留 3,866 页、过滤 110 页（46 页转写为空、字数 0）；保留的里 2,722 页至少检出一处涂改，过滤的里有 10 页也检出了涂改；
  9 页（都在保留里）OCR 没切出可用的行，所以没有结果。网页用 `tools/pages_view` 生成（学生手写图，只放本机的 `work/pages_view/`，不提交）。
- 对照：整页直接过模型（不切行），同一批页面（198 页、340 处涂改）上冻结点只画对 2.4%，不设分数线 23.8%，放大 1.78 倍更差；等距横条（不用行框）26.2%；按行目录的框切行 65.3%（探针，脚本和输出在 `paper2_probe_whole_question`）。

### 6. 掩膜收紧到墨迹

用户要金标和模型的掩膜只含非纯白的像素，之后去掉涂改（整块涂黑、横线涂抹都一样）只去墨迹、不碰旁边的白纸。`python -m revrender.tighten`（提交 `7ebfcef7` 的快照，只用 CPU）把每个掩膜与页面的非白像素取交：墨迹（灰度低于 `eval/ink.py` 的 `INK_BELOW`，200），加上笔画的软边——从墨迹出发、只经过灰度低于 240 的像素、最多走 2 步够得着的像素；只去像素、不加像素，没有墨迹的掩膜丢掉。

第一版的软边是「离墨迹 2 像素以内、灰度低于 240」，不要求与笔画相连。用户看出掩膜里有杂的像素，要先把浅灰的杂点滤掉再收紧。查下来：第一版的掩膜里每个模型掩膜平均有 3.65 个孤立的小杂点（≤ 3 像素的连通块），98% 是不含墨迹的浅灰像素（最暗的中位数 233），离笔画 2–3 像素、多半中间隔着 ≥ 240 的白像素，是 JPEG 的杂点。直接把灰度线调低来滤，先吃掉笔画的软边、杂点却去得慢；要求软边与笔画相连，杂点就没了（验证集 171 页的模型掩膜 764 处）：

| 软边的口径 | 每个掩膜的孤立杂点 | 擦除后原掩膜范围内仍可见（灰度低于 235）的残留 |
|---|---|---|
| 只取墨迹（低于 200） | 0.3 | 9.7% |
| 离墨迹 2 像素内、低于 240（第一版） | 3.65 | 0.5% |
| 离墨迹 2 像素内、灰度线降到 230 / 225 / 220 | 2.44 / 1.74 / 1.16 | 2.3% / 3.9% / 5.3% |
| 与墨迹相连、低于 240、最多 2 步（现行） | 0.27 | 0.9% |

- 金标放回整页（`paper2_gold_pages`）：第一篇数据包的训练和验证两部分，1,644 页、4,284 行、3,137 处涂改（训练 2,772 + 验证 365），一处不少。
- 金标收紧（`paper2_gold_pages_tight_connected`）：3,137 处里 2 处没有墨迹、被丢弃；掩膜总像素 2,749,895 → 1,651,984，39.9% 落在白纸上；每处金标白纸占比的中位数 17.9%、第 90 百分位 52.5%。
- 模型输出收紧：验证集 171 页的 764 处（`paper2_pages_dev171_complete3_tight_connected`）一处不丢，像素 739,839 → 438,871（40.7%），中位数 13.9%、第 90 百分位 52.6%；全部 3,976 页的 14,729 处（`paper2_pages_unlabeled3976_complete_tight_connected`）13,414,274 → 8,278,448（38.3%）。第一版的输出（`paper2_gold_pages_tight`、`paper2_pages_dev171_complete3_tight`、`paper2_pages_unlabeled3976_complete_tight`）已被取代，留着只作对照。
- 页级读数（`eval/pages_score.py`，验证集 171 页、金标 265 处，`paper2_tighten_dev171_scores_connected`）：不收紧时找到 79.2%、画对 64.5%（171 处）、多画 593，与之前的数一致；两边都收紧后找到 79.2%（不变）、画对 68.7%（182 处，多对上 11 处）、多画 582。
- 擦除之后：把掩膜涂白，原掩膜范围内灰度低于 235 的残留，占原掩膜像素的比例，只取墨迹 9.7%，收紧后 0.9%（金标 9.3% → 1.1%）。只取墨迹会留下一圈笔画的灰边。
- 收紧后的掩膜里还剩的小杂点（≤ 3 像素）：模型每个掩膜平均 0.27 个（122/764 个掩膜有），金标 0.18 个（39/265）；剩下的多是 1 像素、22% 含墨迹（掩膜边界切断笔画留下的碎片，留着擦除才干净）。
- 网页（`tools/tighten_view`，学生手写图，只放本机的 `work/tighten/site/`，第一版留在 `site_first/`，不提交）：验证集 171 页的 1,029 个掩膜（模型 764、金标 265）。
- 看到的限制（不是收紧造成的）：横线压在字上的那几笔，线和字是同一处墨迹，擦掉线之后字会断开；掩膜没盖住的墨迹（模型掩膜没覆盖到的笔画尾巴）收紧不会补上。

### 7. 整页清洁

用户要整张答题卡先清洁：去杂质小点，去掉相邻题露进来的虚线（页是围着本题的虚线框裁的，上下各带一截相邻题），再裁到本题自己的框。`python -m revrender.data.clean_pages`（提交 `a43a0bea` 的快照，只用 CPU）：

- **框怎么定。** 第一版取页上最外面的两条横虚线，裁到了相邻题的线上（用户看出来的）：相邻题的线离本题的线只有 11–22 像素。再试「间隔最大的相邻两条」也不行：页内长得像虚线的手写横线（划掉的线、下划线）有时让「最大间隔」落在页内。现行做法：同一道题、同一尺寸的页是从同一张答题卡的同一个窗口裁的（`handwriting.group_key`），虚线在每页的行位置只差一两个像素；整组各页共有的行（至少 30% 的页在 ±3 像素内有虚线）才是答题卡的，本题的框是这些行里间隔最宽的相邻两条，相邻题的线更近。只有部分页有的行是手写的，不参与。每页取与组内位置最接近的自己的虚线行（±8 像素内，否则用组内的），左右取该页最外的竖虚线（不会在组内中位数之内）。23 组各 113–194 页，每组 2–4 个共有行，一眼能核对。
- **框外涂白，页尺寸不变**，所以 OCR 的框、印刷模板、金标都不用动；框外留 6 像素（相邻线更近时少留，保证相邻题的线在框外）。要裁成小图用 `frames.json` 的框。
- **杂点**：一块不超过 8 像素的墨迹、6 像素内没有别的墨迹，就是灰尘，涂白；离墨迹 3 像素以外、灰度在 235 以上的浅灰（JPEG 的雾）漂成纯白。笔画和紧挨笔画的像素不动。随机 500 页里 79% 的页有这样的孤立小黑点，平均每页 7.9 个；全部 3,976 页去了 11,596 个。
- **结果**（`paper2_pages_clean_unlabeled3976`，`frames.json` 里 3,976 页都有框）：在本机的一次清洁上数清洁后的虚线行：每页只剩 2 条（本题自己的上下框线）的有 3,955 页，14 页多一条（页内长得像虚线的手写线，在框里面），7 页只检出 1 条（框用的是组内的位置）；相邻题的线一条不剩。框外被去掉的有页角的黑色定位块、页脚「第 1 页（共 2 页）」、相邻题的虚线和手写行；抽看 14 页框外的页独有墨迹，都是相邻题的字、线和页边的块，没有看到本题自己的字被裁掉。
- **整页流水线在清洁后的页上重跑**（`paper2_pages_clean_run_unlabeled3976`，`--frames`，存档的 OCR 文本，没重跑 OCR；GPU 约 14 分钟）：块 45,073 → 44,331（框外的 OCR 块 727 个丢掉），涂改 14,729 → 14,620 处；原来有 120 处的中心在清洁用的框外（页角的黑块被标成涂黑，分数多在 0.8 以上，和相邻题的字被标成涂改），现在是 0。
- **验证集 171 页对金标 265 处**（`eval/pages_score.py`，`paper2_clean_dev171_scores`）：

| | 找到 | 画对 | 多画 |
|---|---|---|---|
| 原页 | 79.2% | 64.5%（171） | 593 |
| 清洁后的页 | 80.0% | 64.5%（171） | 591 |
| 原页，两边收紧 | 79.2% | 68.7% | 582 |
| 清洁后的页，两边收紧（`paper2_pages_clean_run_unlabeled3976_tight`、`paper2_gold_pages_tight_clean_dev171`） | 80.3% | 69.7%（金标 264 处：1 处在框外的相邻题虚线上，清洁后没有墨迹） | 578 |

- **金标里在框外的**：有框的 1,463 个金标页的 2,499 处里，10 处（0.4%）的中心在本题框外：相邻题的手写行（`7bd2e73d…` 的 3 处）、页边、相邻题的虚线。
- 网页（学生手写图，只放本机，不提交）：`tools/clean_view` 的 `work/clean_view/`（每题 30 页，240 页）；收紧对比页在清洁后的页上重做，`work/tighten/site_clean/`；总览页在清洁后的页和收紧后的掩膜上重建，`work/pages_view_clean/`（字数阈值 30：保留 3,596 页、过滤 380 页）。
- 没做的：字数过滤用的 Qwen 转写还是原页的，相邻题露进来的字会让少数页的字数偏高；要精确得用清洁页重新转写。

### 本地测试

`python3 -m pytest -q`：本地全过（需要 torch / skimage 的测试跳过）；服务器上全部通过（各次验证目录里的 `pytest.log`）。含一条行为测试（训练数据永不产出测试集的行）、与旧实现逐项比对的增广 / 采样器金标测试、
冻结模型的参数清单（533 项名称、形状、类型）金标测试、仓库里不再出现旧脚本编号的测试。

## 做过的决定

用户让我自己决定、在这里记一句：

- 最终模型的链（654 → 674 → 685 / 687 → 816 → 750）里 775、767、865 在默认值下是透传；730 的单文件模型和「空查询」损失不在最终模型里，没搬。
- 权重初始化等价需要把 654 里被丢弃的 find / draw 也建一遍（它们消耗随机数）：`models/legacy_init.py`，另有 `fork_rng(devices=[])` 之后 CUDA 被重新播种的副作用，也原样保留。删掉它们是另一个模型。
- 旧循环每次验证存一份 333 MB 的 resume 状态但从不读取；新代码不存。
- 旧训练的数据守卫（登记的划分映射、冻结测试集 SHA）绑定第一篇的划分；新读取器只要求数据包自己的 manifest、映射与记录一致、没有测试行、没有跨划分的书写者，第二篇换划分时不需要改代码。
- 数据文件里带历史编号的标签（数据包 / 校正行 / 冻结记录的 schema）用正则读取；新写出的冻结记录用 `revrender-freeze-v2` 等新标签。唯一保留历史文本的是写作者子集的哈希盐（`data/journal.py` 的 `SALT`）：改了它每个比例的子集就变了。
- 冻结记录的 `split` 字段由旧的划分代号改为 `validation`。
- 917 用到旧 517 的 ViT 和组件提取函数，所以那些函数（只有那些）在 `baselines/mathscrub.py` 里；517 本身的训练入口不搬。
- 四个涂改专用方法各加了一个 `--smoke`（8 步 / 20 行，检查流程，输出里标 `smoke_only`），因为它们原来没有 smoke。
- 数据包构建器（生成数据包的旧脚本）、标注协议与补标链不搬；题目图的位置已核对，见 [docs/data_location.md](docs/data_location.md)。
- 各读数共用一份「读冻结」（`eval/freeze.py` 的 `read_frozen` / `frozen_point`）：fates、ap、bootstrap、report 和新的分析读数都走它，不各自再写一份；`--ink-only` 是 fates、ap、bootstrap 的选项（`eval/ink.py`），不是另一条流水线。
- 2026-10-03 用户要求整页流水线跑全部页（外部集 1,026 页除外：那是同门的另一种标注，第一篇的行级数据只是整页数据集的小子集，以后并进来），包括第一篇 44 名测试书写者的 292 页。只做推理，预测不用于训练或选择；输出和网页里标了每页书写者在第一篇的划分（`tags.json`），第二篇定划分时可以据此留出。
- 页的字数过滤：字数 = 学生手写解答转写的字数（`data/ocr_lines.answer_chars`：去掉标记和空白后的长度），转写用旧标注流水线的 Qwen3-VL-8B 结果（只转写学生手写的、不含印刷题干和被划掉的内容）。不用 Unlimited-OCR 的文字数：它含印刷的题干，几乎空白的页也有十来个字，而且「严格行」会漏掉整块的多行区域。旧的去重文件漏了 71 页的转写，改读四个分片文件、每页取第一条非空记录；46 页的转写全是空，字数记 0（它们的 OCR 文字只剩印刷题干，约 9 个字）。阈值 10（`--min-chars`）。
- 之前那次批量（`paper2_pages_nontest_stored_ocr`，4,628 页，含外部集的页）不含第一篇 44 名测试书写者的任何一页，已被 `paper2_pages_unlabeled3976`（3,976 页，不含外部集、含测试书写者的页）取代；两次都用存档的 OCR 文本而不是重跑模型（两者在 205 页验证页上逐位相同，重跑要 3 小时以上）。
- 整页的切法（`pages.page_lines`）：OCR 的「严格行」（第一篇训练用的那些行，原样保留，所以金标上的读数不变）；OCR 返回成一块的多行区域，按墨迹的缺口切成行（旧的等高均分会切歪）；OCR 没框的手写补框。补框先在每道题的印刷模板（同一道题、同一裁剪尺寸的页，至少一半的页在该像素有墨）和答题框之外找手写墨迹，按行合并，丢掉只有细线、虚线或单根笔画的（印刷框的残影、扫描边缘的线），一块跨多行的按墨迹缺口切成行。涂抹跨多行就按行拆开、不合并，与第一篇逐行金标一致。
- 试过「不取矩形」：补切的片段只留笔迹的形状、框里其余抹白。在有金标的验证行上，同样的 OCR 行改成这样读，265 处金标对上 167 处，矩形是 171 处，略差，所以不用，补切的块就是普通矩形（`paper2_shape_ab_rect`、`_shape`，脚本在里面）。
- 掩膜收紧（`tighten.py`）：口径 = 页面的非白像素：墨迹（`INK_BELOW` 以下）加与墨迹相连的软边（从墨迹出发、只经过灰度低于 `EDGE_BELOW`（240）的像素、最多 `EDGE_PX`（2）步）。不用「低于 255」：这些扫描件的空白纸正好是 255（随机抽 400 页，每页的众数都是 255），但 JPEG 在笔画旁留下 220–254 的浅灰杂点，低于 255 几乎什么也去不掉；不用逐像素腐蚀：它去不掉笔画之间的白纸；不只取墨迹（低于 200，第一篇 `--ink-only` 的口径，只用于读数）：擦掉之后笔画的灰边还在，是一圈灰色的影子；也不靠调低灰度线滤杂点：它先吃掉软边（见上表）。软边要与笔画相连，是用户指出第一版掩膜里有杂的像素之后改的。只与页面像素取交，不生长、不补笔画；没有墨迹的掩膜丢掉并记数。收紧后的输出与整页输出同一布局，`pages_view` 等工具可以直接读。
- 整页清洁（`data/clean_pages.py`）：框由同一道题、同一尺寸的页共有的虚线行定，不是每页自己找最外的两条（会裁到相邻题）或最大间隔（页内像虚线的手写线会骗它）；页尺寸不变（涂白不裁剪），否则 OCR 框、印刷模板、金标都要换坐标，印刷模板要求同一道题的页同尺寸对齐；清洁后的页存 PNG（不再压 JPEG，免得把杂点压回来），`pages.page_files` 找 jpg 或 png、同名时 jpg 优先；OCR 文本不重跑（框内的内容没变，框外的块用 `--frames` 丢掉）。浅灰雾只在离墨迹 3 像素以外、灰度 235 以上才漂白，浅色的笔画不动。
- 922（线性探针）要抽模型特征，这次不搬；923 里旧的消融变体（rvr_coarse、nofine_nodense、nocontext）不搬，只比八个主模型。

## 已知差异

- 原生基线的反向梯度范数在两次运行之间相对差 ≤ 4e-5（detectron2 的 RoIAlign 等 CUDA 原子加）；前向、损失、存档逐位一致。
- 本地没有 torch / detectron2 / segmentation_models_pytorch，模型类测试在本地跳过，在服务器上跑全部。
