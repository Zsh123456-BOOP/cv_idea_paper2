# revrender

## 恢复模型统一评测（2026-10-22 起为验证集版）

入口 `GPU_A=… GPU_B=… scripts/eval_unified_restoration.sh NEW_OUT`（默认 1、2 号卡），候选列表在 `configs/unified_restoration.json`：我们的纯合成版（框内 step10500）、201 处真实微调版（step600）和七个恢复基线，都是去掉同门数据后重跑的 `paper2_clean_*`，权重开始时冻结哈希，不重训、不选版。

- 只评验证集：像素 16 处（固定验证 `paper2_fixed_validation20261007`），OCR 41 处可读划线。旧测试集（22 题里 5 题是同门的）已在 10-18 停用，`--geometry`（旧 7 题像素测试）可省；新测试集只在最后评一次。
- 像素：原图灰度 <200 的墨迹域，恢复图亮度增加 >0.2 才算擦除，目标掩膜也限制到同一墨迹域。OCR：清洁原图、人工框加 6 像素，原图和全部模型在同一次 Qwen3-VL-8B TP2 中转写（erased_neutral，贪心，种子 42，256 tokens），报字符读回和严格全对。
- 网络预处理照各自训练方式：我们的模型 768×96 letterbox、top1、mask 0.4；恢复基线原生 256 分块，输出映射回原生坐标。不加标注页的残留清理。
- 统一评测不等于统一训练：我们的微调版有真实精修监督，七个基线用同一批合成训练对；比较的是现有模型的实际表现，不是控制训练数据后的架构优劣。

## 标注对每个模型都有用吗（2026-10-22 起）

`scripts/run_annotation_value.sh`（标记在服务器 `paper2_annotation_value20261022/`，可续跑）：
- 同一批合成数据：七个恢复基线训练用的 9,636 对，`restoration_pairs boxes` 切成框内去划线模型的条带（按变暗 >0.2 的像素定要擦的笔画，框外随机放宽 0–0.4 个框高），我们的模型在其上从 ImageNet 起训练（`revrender_strokes.toml`）。
- 选版统一按读回（用户定）：每个模型的候选 = 在 16 处固定验证上像素 F1 比之前都高的版本和最后一版（基线训练本来就这样存，我们的每 2,000 步一版），再取 41 处验证划线读回最高的，同分取早的（`unified_restoration choose`）。
- 同一批真实标注：201 处人工区域 + 同一批合成回放（`review_set --replay` 用上面的条带；`restoration_pairs human` 把同一批区域按基线的 256 分块切、回放取原来的那几对），每个模型从选出的合成版微调（`restoration.toml [finetune]`：与我们的微调同样遍数、各自学习率的十分之一、四次验证；我们的照 `revrender_strokes_human.toml`），再按同一规则选版。
- 判定标准（2026-10-22 读数之前写下）：**有用** = 8 个模型里至少 6 个读回变高，且 8 个模型配对差的平均（`compare` 的 `_all`，书写者一起重抽样）的 95% 区间在 0 以上；**方向一致但未定** = 多数（5 个以上）变高、平均区间跨 0（验证集太小，等扩大的验证集和测试集）；**没用** = 不到一半变高。像素 F1 另报每处平均，不参与判定。测试集不参与。
- `unified_restoration compare` 给每个模型微调前后读回的配对差（按书写者重抽样的 95% 区间）；`eval.restoration_errors` 给错误落在哪（压字处 / 远离处、条带放大 / 缩小）和真实与合成划线的统计（笔宽比、碰字比例、深浅、长度、直度）。只用验证集。

第二篇论文（整道题的涂改识别）的代码库。第一篇（RevRender + ExamRev，涂改实例分割）的模型、训练、评测、基线和工具，从旧仓库整理成一个普通的 Python 包。旧仓库的脚本是编号的一次性实验，互相按路径加载、用 SHA-256 钉住；这里是正常的包、正常的 import、一个命令行入口一件事。

- 移植自旧仓库 `~/Desktop/cv_idea_claude`，分支 `codex/claude-three-mechanisms`，提交 `aa8e2e01`（只读，未改动）；三个分析工具后来补搬自同一分支的 `47293cdd`。
- **已复现第一篇验证集数字**，三项等价验证的结果见 [等价验证](#等价验证)。
- 远端：GitHub 私有库 `Zsh123456-BOOP/cv_idea_paper2`（`origin`，分支 `main`）。
- 已接入整题清洁、优化切块、掩膜收紧，以及用户授权的 40 页逐行识别与标注试验；第二篇正式数据划分和训练方案尚未确定。

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
  recognize_lines.py 图包里的行用一个 VLM 在若干条件下读（vLLM；`--backend hf` 加视觉 token 屏蔽的条件和 `--adapter`）
  train_reader.py    在合成涂改上给 Qwen3-VL 训练 LoRA（语言模型的线性层），读「最终有效作答」
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

## 识别试验（第二篇的第一步）

第二篇的方向（2026-10-04 用户定，面向计算机方向，不做教育评分）：涂改感知的多模态识别。上游是第一篇的涂改掩膜（`tighten` 之后只含墨迹）；问题是 VLM 读手写数学作答时，划掉的被读成有效（泄漏）、有效的被读坏（误删）。第一步先量问题有多大，再决定做方法。

**旧项目（`~/Desktop/cv_idea`）已经有的，不用再标。** 都以目录里的行 id（`<image_id>:L<nnnn>`）为键，和本项目切页用的是同一个 id；读取用 `python -m revrender.data.prior_annotations --mapping index_manifest.json --same-student 对子.json --output-dir OUT`（`configs/paths.toml` 的 `[prior] root`；测试书写者按第一篇的划分映射认，外部集的书写者也在里面），文件清单和字段见 [docs/data_location.md](docs/data_location.md) 第 8 节：

| 内容 | 行数 | 说明 |
|---|---:|---|
| 人工前后对（整行划掉前的文字 `text_before`、整行划掉后的文字 `text_after`——整行，不是只含划掉的那一段，两者大部分相同；涂改类型） | 327 | 其中 288 行不是第一篇测试书写者的（测试 39 行）；类型：划掉重写、只划掉、覆盖改写、整行；标注者以三个模型的转写为起点修改，抽样是 200 个高分歧行加 100 个随机行 |
| 人工盲判「这一行没有涂改」 | 433 | 390 行不是测试书写者的；做对照块 |
| 人工判「有涂改」但没写前后文 | 67 | |
| 三个模型对每一行（46,902 行）的转写 | 46,902 | Qwen3.5-35B-A3B-LaTeX-OCR（`FutureMa`，即印象里的「32B」）、Infinity-Parser2-Pro，加 Unlimited-OCR 的文字；提示要求保留划掉的字 |

在 288 个人工确认的涂改行上，两个强模型读得很不一致：Qwen3.5-35B 与 Infinity 的相似度中位数 0.44、不低于 0.9 的占 31%；390 个无涂改行是 0.94 和 54%。要求「保留划掉的字」时，三个模型的读数里含有被划掉的字符的比例（只算划掉的那一段，按下面的字符覆盖率）：Unlimited-OCR 47%、Infinity 52%、Qwen3.5-35B 51%。

**Codex 做过的两轮（只是记录，没有人工答案，不当证据；代码已在第一步之后删掉，历史在 git，最后一版是 `6404e64`，输出留在本机 `work/recognition_*`）。** 都是 Qwen3-VL-8B 读我们的整题切块，B 原图 / C 掩膜遮白 / D 原图加红色掩膜图：

- 第一轮（`work/recognition_pilot40`，40 页 448 块）：提示要求「划掉、涂黑的不输出」，带整页上下文；其中 98 块有检测掩膜。
- 第二轮（`work/recognition_revision80`，80 个检测阳性块）：不提这句，分「全部墨迹如实转写」和「正常直接输出、作废的用 `<del>` 包裹」两个任务。
- 对着输出审计：在 80 个阳性块上，遮白与原图的读数只有 39% 一致，其余 368 块是 88%，第一轮「三组差不多」是被无涂改块稀释；「划掉的不输出」这句基本没改变输出（长度差不多的 41/80，变短的 19/80），抽看 12 块，整行划掉的被原样读出来；`<del>` 标签 B 10/80、C 5、D 19；C 有空输出 6/80 和幻觉；D 是两张图输入，模型会复述辅助说明（deletion 任务 6/80）。检测阳性不等于人工涂改。

**第一步的设计（先量大小，不训练）：**

- 样本：人工前后对里不是测试书写者的 288 行，加人工盲判无涂改的 390 行（对照）；测试书写者按第一篇的划分映射认（外部集的也排除）；不用新标。每行是目录里的精确裁图，由第一篇的 RevRender 在冻结点读一遍，掩膜收紧到墨迹（`data/line_pack.py`）。
- 条件（每个请求一张图、同一提示骨架，`recognize_lines.py`）：`raw_neutral` 原图；`raw_final` 原图 + 「只转写最终有效内容」；`erased_*` 掩膜的墨迹擦掉；`blanked_*` 掩膜下整列带涂白（一行一裁图，被划掉的字跟着没了）；`overlay_final` 红色标出涂改位置（一张图）；`hint_final` 把涂改位置写进文字提示。
- 指标（`eval/coverage.py`、`eval/leakage.py`）：标注者写的 `text_before`、`text_after` 是整行的前后文字，两者大部分相同，所以先按字符把一行分成「被划掉的」（前有后无）、「重写的」（后有前无）、「仍有效的」（前后都有）三部分；读数「持有」一个字符 = 读数与那段文字的最小编辑对齐保留了它。泄漏 = 读数持有被划掉字符的比例（按行取平均，越低越好），重写、有效同理，有效在涂白后会掉；空 = 整行划掉的行被读成空的占比；对照行上读数相对 `raw_neutral` 的改变。区间 95%，按书写者重抽样；`references` 给出标签允许的上下限（读数恰为划掉后的文字：泄漏约 1%；恰为划掉前的：100%；别的行的文字：泄漏 4%，和原图读数一样长时约 10%）。
- **第一版的指标有错，已作废。** 第一版问「划掉前的整行文字是否（模糊）作为子串出现在读数里」。标签是整行文字时，把 `text_after` 原样当读数，有重写的行里 89%（全部 51%）被判成泄漏——这条指标的下限（51%）高于原图的数（43–48%），当时没有用「读数恰为标签」去检验指标，结论「泄漏 43–48%、涂白后仍留 28–35%、放行」不成立。下面是改成按字符分段后重算的，并经过独立复核（三个只读审计：指标边界、标签语义与噪声、独立重写实现）。

**第一步的结果（`paper2_s0b_*`，贪心、种子 42、vLLM，每个模型 8 个条件 × 678 行；288 个涂改行里 275 行有划掉前的文字，其中 196 行有被划掉的字符、151 行有仍有效的字符、51 行有重写的字符）：** RevRender 在 288 个涂改行里的 256 行给出掩膜，在 390 个对照行里的 14 行给出掩膜（误标 3.6%）。

| 泄漏：读数持有的被划掉字符占比（196 行，行均值） | Qwen3-VL-2B | 4B | 8B | 32B-FP8 | Qwen2.5-VL-7B |
|---|---:|---:|---:|---:|---:|
| 原图，中性提示 | 37% | 45% | 47% | 36% | 40% |
| 原图 + 只转写最终有效内容 | 37% | 41% | 44% | 36% | 39% |
| 掩膜的墨迹擦掉 | 33% | 34% | 37% | 31% | 32% |
| 红色标出涂改位置 | 40% | 36% | 42% | 30% | 37% |
| 涂改位置写进文字提示 | 35% | 45% | 42% | 31% | 39% |
| 掩膜下整列带涂白 | 13% | 12% | 13% | 13% | 13% |
| 仍有效字符的保留（151 行）：原图 + 最终提示 → 涂白 | 84 → 76% | 87 → 79% | 87 → 81% | 86 → 80% | 83 → 78% |

- 提示、红色标记、文字坐标对泄漏的作用在 0–6 个百分点内，规模（2B 到 32B）没有规律；擦掉掩膜的墨迹少 4–7 个点（掩膜是划线的笔画，擦掉后被划掉的字更清楚）；涂白少 24–31 个点（区间都在 -18 以下），代价是仍有效字符少 5–8 个点（区间都不含 0）。
- 数字的精度（审计的结论）：对齐有并列最优解，同一个字符的两处拷贝（「8585」里被划掉的 85 和有效的 85）算哪个不定，所以每个泄漏数有约 3 个点的上下界（`strict`–`lenient`，涂白后 10–15%）；按字符加权的泄漏比行均值高（原图 49–61%，涂白 13–15%），因为 38% 的行只有 1–2 个被划掉的字符；别的行的文字当读数的下限是 4%（和标签一样长）到约 10%（和原图读数一样长）。**不要把泄漏单独拿来读**：什么都不读也是 0% 泄漏，涂白时 Qwen3 系列有 28–32% 的行读成空或只有占位符（原图 1–15%，Qwen2.5-VL 3–4%）。
- 涂白后剩下的泄漏在哪：有掩膜的行 9–10%（已在 6–10% 的随机下限附近）；没有掩膜的行（196 行里的 19 行，275 个有标签的行里的 29 行）原样不变，31–49%，这是第一篇检测的召回，不是识别能解决的；整行划掉且没有重写的 113 行基本解决（有掩膜的 107 行从 42% 到 6%）；有仍有效或重写字符、又有掩膜的 70 行是真正难的部分，35% → 12%（上界 20%）。
- 仍有效字符少的 5–8 个点是上界：差值是重尾的（中位数为 0，15 条最差的行占净损失的三分之二，去掉最差 25 条后是 -1 个点），最差的行里约一半是标签的问题（标注者把被划掉的字也写进「前后都有」，或前后文字写反），约一半是真损伤：重写在划线的上方或下方却被整列带涂白、下划线被当成划掉、掩膜太宽。涂白对没有标注涂改的 390 个对照行的读数改动小（相似度下降约 0.02–0.03，被误标的 14 行 0.12）。
- 标签有噪声：抽 29 行对着图像看，14%（4 行）严重错（前后写反、重写在图里看不到），约 38% 有字符不准（标注者以模型的转写为起点改的）；275 行里 69 行（25%）前后文字相同（划线是真的，被划掉的字没写进标签，泄漏测不到）；抽样是 200 个高分歧行加 100 个随机行，不代表涂改行的总体，泄漏率不是涂改在作答里的发生率。重写字符只有 51 行（49 个书写者），区间都含 0，**测不出重写的损失**。
- 条件放行：原图泄漏 ≥20% 稳（任何定义下区间下界约 30%）；「最好的免训练基线仍留 ≥10%」只是勉强满足（11–14.5%，其中随机下限 6–10%，没有掩膜的行占约 4 个点），所以**单看泄漏不足以支持一个新方法**。方法的空间在三处：仍有效字符的损伤（约 6.6 点，是上界）、没有掩膜的行（占涂改行的 10%，只有自己找划线的方法才够得着）、同时有仍有效和被划掉字符的行（70 行）；判法是在同样泄漏下比涂白多保住多少有效字符，不能只比泄漏。
- 没跑成：InternVL3.5-8B 和 Gemma-3-4B 在这台服务器的 vLLM 引擎启动时卡住（试了两个环境），没有结果，不是结论。
- 局限：掩膜是模型给的不是人工的（只有 19 个涂改行有第一篇的人工掩膜，没法做「人工掩膜」上限）；泄漏、重写、有效是字符对齐的占比不是整句准确率；`struck`（195 行）是配对标注者没细分类型的划掉。

**第二步：免训练的视觉 token 屏蔽（`models/vlm_tokens.py`、`recognize_lines --backend hf`，Qwen3-VL，不训练）。** 图像不动，把掩膜覆盖到的视觉 token 对文字隐藏：之后的任何位置（其余视觉 token、提示、正在写的读数）都看不到它们，每个 token 保留原来的 M-RoPE 位置，等于把这些 token 从序列里拿掉、留着空位；软屏蔽是给被隐藏 token 的注意力 logit 减一个偏置而不是删掉。解码是自己的贪心循环（4D 加性掩膜、模型自己的 `get_rope_index`、解码步沿用缓存里的偏置），不隐藏任何 token 时和 `generate` 逐 token 相同（`tests/test_vlm_tokens.py`，在 2B 上跑过）。踩到的一个坑：torch 2.9.1 + cuDNN 9.10 把 Qwen3-VL 的补丁嵌入（stride 等于 kernel 的 Conv3d）在 bf16 下跑到每张图 1.4 秒，改成它本来就是的矩阵乘后同数不同速（`patch_embed_as_linear`）。同一个 678 行的图包，泄漏 / 有效 / 读空（同上指标，196 / 151 行）：

| Qwen3-VL-8B（括号里 2B） | 泄漏 | 仍有效字符 | 读成空或占位符 |
|---|---:|---:|---:|
| 原图（vLLM 的数：47 / 88） | 46 (38) | 88 (84) | 2 (13)% |
| 掩膜下整列涂白（像素；vLLM 的数：14 / 81） | 14 (12) | 81 (77) | 28 (31)% |
| 隐藏整列覆盖 ≥10% 的 token | 15 (13) | 81 (78) | 28 (32)% |
| 隐藏整列覆盖 ≥25% 的 token（默认） | 15 (13) | 81 (79) | 27 (31)% |
| 隐藏整列覆盖 ≥50% 的 token | 17 (14) | 82 (80) | 24 (30)% |
| 隐藏整列覆盖 ≥75% 的 token | 19 (14) | 83 (80) | 23 (28)% |
| 软屏蔽（整列 ≥25%，logit −4） | 18 (13) | 81 (80) | 27 (31)% |
| 涂白 + 隐藏整列 token | 14 (11) | 81 (76) | 28 (31)% |
| 隐藏落在墨迹上的 token（墨迹面积 ≥1% / ≥5% / ≥20% / ≥50%） | 37 / 37 / 42 / 45 (26 / 27 / 32 / 36) | 86 / 86 / 87 / 88 (83 / 84 / 84 / 84) | 7 / 7 / 5 / 3 (19 / 19 / 16 / 14)% |

- HF 路径和 vLLM 一致（原图 8B 泄漏 46 对 47，有效 88 对 88；涂白 14 对 14，有效 81 对 81）。
- **隐藏 token 和涂白像素是同一件事**：隐藏整列的 token 与涂白整列的像素读数几乎相同（泄漏 15 对 14，有效 81 对 81），软屏蔽、涂白再隐藏也没有更好；视觉编码器本身的注意力在文字看到之前就把被隐藏的补丁混进了邻居，所以只隐藏 token 擦不掉底下的东西（涂白像素才能）。
- 所有变体落在原图与涂白之间的同一条线上：每少 31 个点的泄漏换 7 个点的有效字符（8B：原图 46 / 88，涂白 14 / 81；阈值 0.1 到 0.75 在 15 / 81 到 19 / 83，墨迹 token 在 37 / 86 到 45 / 88）。泄漏与有效字符的取舍由「去掉了多大的区域」决定，不由「怎么去掉」决定；没有一个免训练设置在同样泄漏下比涂白多保住有效字符。视觉 token 屏蔽不是一个方法，留作基线。

**第三步：在合成涂改上训练 LoRA（`data/paste_strikes.py`、`data/reader_corpus.py`、`train_reader.py`；不用掩膜、不改图）。** 问题的症结（上面三步）：模型读到被划掉的字就当有效，提示和标记都救不了，涂白会伤到仍有效的字，没有掩膜的行谁都管不了。这一步让模型自己学会「只读最终有效的作答」：

- 样本（配方文件 `reader_corpus`，画图在读取时）：宿主行是目录里判为没有涂改的行（人工或模型共识），写作者在第一篇的训练划分里，两个强模型（Qwen3.5-35B 与 Infinity）读出来相似度 ≥0.9 且至少 4 个字符，取前者的读数当目标；**不含图包里任何一行的书写者（360 人）、不含测试书写者**（评测行不能和训练同一只手）；剩下 3,013 行、130 位书写者，其中 7 位按哈希留作验证。每个宿主出四个样本：原样（目标是它的文字）、整行划掉（目标是空，`[空]`）、两个贴入（把同一书写者别的行里的一两个词剪下、划掉、插进这一行的空隙或行尾，目标仍是宿主的文字，被贴入的词不在目标里）。共 12,048 个配方。
- 划法七种（直线、双线、波浪、折线、叉、乱涂、涂黑块），颜色取宿主的墨色，笔宽随行高；40% 的贴入让笔画越过被划的字压到邻居的字上（和有效笔迹重叠）。目标文字由构造已知，不需要知道哪个字在笔画下面。
- 训练：只给语言模型的线性层加 LoRA（rank 16，alpha 32），视觉塔不动；请求的话是 `raw_final`，答案是 `<transcription>文字</transcription>`，损失只算答案；600 步 × 32 个样本；种子 42（全仓库只用 42，所以没有多种子的区间，下面的区间只来自书写者重抽样）。**对照臂**：同样的配方但只用原样的行（`--kinds clean`），同样的步数：它给出「书写者的手和答案的写法」带来的变化，划掉的字的作用是和它的差。
- 评测：同一个 678 行的图包（评测行的书写者不在训练里），原图直接读，**不给掩膜**。

| Qwen3-VL-2B（`paper2_reader_*`、`paper2_readeval_2b_*`） | 泄漏 | 仍有效字符 | 重写字符 | 整行划掉的 113 行里读成空或占位符的 | 全部评测行里读成空或占位符的 |
|---|---:|---:|---:|---:|---:|
| 原模型，原图 | 38 | 84 | 60 | 35 | 14% |
| 原模型，掩膜下整列涂白 | 12 | 76 | 55 | — | 32% |
| 对照臂（只训练原样的行），原图 | 45 | 88 | 63 | 4 | 1% |
| 对照臂 + 涂白 | 19 | 81 | 63 | — | 0% |
| **划掉的字（对照臂加上整行划掉和贴入的样本），原图** | **16** | **81** | 60 | **99** | 40% |
| 同上 + 涂白 | 9 | 76 | 58 | — | 37% |

- 总数看上去好（泄漏 45% → 16%，整行划掉的 113 行 99 行读成空，没有掩膜的行 47% → 16%），**但拆开看，收益几乎全在整行划掉的行上**（三个只读审计之后）：

| Qwen3-VL-2B，原图，泄漏 / 读成空（按行取平均） | 整行划掉的 122 行（目标是空） | 划掉一部分的 153 行（目标非空）：泄漏 | 其中仍有效字符 |
|---|---:|---:|---:|
| 原模型 | 38 / 31% | 38 | 84 |
| 对照臂（只训练原样的行） | 45 / 3% | 45 | 88 |
| 划掉的字（含整行划掉与贴入的样本） | **8 / 84%** | 29 | 81 |
| 原模型 + 涂白 | 9 / 70% | 17 | 76 |
| 对照臂 + 涂白 | 20 / 0% | 17 | 81 |
| 划掉的字 + 涂白 | 7 / 78% | 12 | 76 |

  - 整行划掉的行：划掉的字让模型学会说「没有」（84% 读成空），不需要掩膜；这部分是「记忆也解释不了」的（目标是空）。
  - 划掉一部分的行（难的部分）：泄漏 45% → 29%，比原模型（38%）少 9 个点，**不如「对照臂 + 涂白」（17%，同样的有效字符 81%）**；所以「和涂白同一水平」这句话在这部分不成立，只在整行划掉那部分成立。
  - 对着图像人工看了 40 行（独立审计，读数先于图像看，不是盲的）：整行划掉的 14 行里，划掉的字模型失败 1 行，对照臂失败 12 行；有仍有效字的 25 行里，失败 9 对 11（p=0.69，没有差别）；划掉的字模型有 2 行把有仍有效字的行读成了空；40 行里 16 行的标签错或不能用（10 行与图像矛盾，4 行划线看得见但前后文字相同）。**所以评测数字的绝对水平不可靠，方向可靠。**
  - 对照臂本身比原模型差（泄漏 38 → 45）：它看了 6.7 遍同样的 2,859 张原样的行（划掉的字那一臂是 1.7 遍），且没见过空目标和贴入的墨迹，所以「45 → 16」夸大了划掉的字的作用，对原模型是 38 → 16（整行划掉的行 38 → 8，划掉一部分的行 38 → 29）。对照臂不能隔离「学会认划线」和「学会说没有」，要补两个对照臂（只有原样和整行划掉；原样的行按同样的遍数）。
  - 没有记忆：评测行的标签与训练宿主文字（同一道题，不同书写者）近乎相同的只有约 10%，划掉的字的读数抄训练宿主的比例（14.5%）不比对照臂高（12.7%），泄漏的下降在训练里出现过的文字与没出现过的文字上相同（15 与 16 个点）。
  - 训练语料里有两位书写者（81 个宿主）与评测书写者是同一个学生（两个编号）；只影响 3 个没有涂改的对照行，不影响泄漏 / 有效 / 重写的数字；现在的生成器已排除（`reader_corpus_main`：2,932 个宿主、128 位书写者），已报的 2B 臂是用没排除的那份训练的。
- 代价：仍有效字符 88 → 81（对原模型 84 → 81），对没有涂改的 390 行，划掉的字训练出来的读数与对照臂读数的相似度 0.957，1.5% 的行读成空（6 行）。
- 注意这是一个种子、一个合成配方，评测行 275 个。

**2B 上补的对照臂（审计指出的混杂；全部原图直接读，`paper2_reader_qwen3vl2b_*`、`paper2_readeval_2b_*`）：**

| 2B 的臂 | 泄漏 | 仍有效字符 | 读成空 | 整行划掉：泄漏 / 读成空 | 划掉一部分：泄漏 / 有效 | 加涂白：泄漏 / 有效 |
|---|---:|---:|---:|---:|---:|---:|
| 原模型 | 38 | 84 | 14% | 38 / 31% | 38 / 84 | 12 / 76 |
| 对照臂（只有原样，600 步） | 45 | 88 | 1% | 45 / 3% | 45 / 88 | 19 / 81 |
| 对照臂，同样的遍数（150 步，每张原样的行约 1.7 遍） | 44 | 88 | 0% | 47 / 1% | 40 / 88 | 21 / 80 |
| 只有原样 + 整行划掉（只学「说没有」） | 14 | **61** | 60% | 4 / 96% | 31 / **61** | 10 / 73 |
| 原样 + 整行划掉 + 贴入（上面的主臂） | 16 | 81 | 40% | 8 / 84% | 29 / 81 | 9 / 76 |
| 再加叠放（`reader_corpus_main`，已排除同一学生的另一个编号） | 19 | 83 | 38% | 12 / 80% | 30 / 83 | 10 / 77 |
| 训练时就把掩膜下整列涂白（原样 + 整行划掉 + 贴入，`--blank`） | 36 | 87 | 24% | 32 / 52% | 43 / 87 | 10 / 78 |

- 对照臂没有被过拟合拖偏：遍数对齐后（150 步）和 600 步一样（泄漏 44 对 45，有效 88 对 88）。
- **只学整行划掉不行**：泄漏也低（14），但有效字符掉到 61%，60% 的行读成空；所以贴入的样本才是教会模型「划掉一部分的行里别的字还要读」的（同样的泄漏下有效字符 81 对 61）。
- 叠放（划掉的字放在宿主某个字的上方或下方、与有效字的列重叠）没有进一步压低划掉一部分的行的泄漏（30 对 29），有效字符多 2 个点，在区间内。
- 训练时涂白：不涂白的原图读不好（泄漏 36），涂白后 10 / 78，和主臂 + 涂白（9 / 76）、对照臂 + 涂白（19 / 81）在同一条取舍线上，没有多保住有效字符。
- 划掉一部分的行在 2B 上仍然是难的部分（泄漏 29–43，涂白能到 12–19）。

**Qwen3-VL-8B（`paper2_reader_qwen3vl8b_*`、`paper2_readeval_8b_*`；同样的配方、同样的 600 步，对照臂同样只训练原样的行）：**

| 8B | 泄漏 | 仍有效字符 | 读成空或占位符 | 整行划掉：泄漏 / 读成空 | 划掉一部分：泄漏 / 有效 | 没有掩膜的行：泄漏 |
|---|---:|---:|---:|---:|---:|---:|
| 原模型，原图 | 43 | 87 | 3% | 49 / 6% | 33 / 87 | 55 |
| 原模型 + 涂白 | 14 | 81 | 28% | 12 / 61% | 16 / 81 | 52 |
| 对照臂，原图 | 46 | 88 | 0% | 55 / 0% | 33 / 88 | 48 |
| 对照臂 + 涂白 | 15 | 82 | 0% | 14 / 0% | 18 / 82 | 47 |
| **划掉的字，原图（`raw_final` 的话）** | **17** | **86** | 36% | 14 / 77% | 22 / 86 | **16** |
| 划掉的字，原图，中性的话（`raw_neutral`） | 33 | 89 | 17% | 36 / 39% | 28 / 89 | 42 |
| 划掉的字 + 涂白 | 10 | 81 | 36% | 8 / 77% | 14 / 81 | 16 |

配对差（同一批行，按书写者重抽样的 95% 区间，所有 275 个有标签的行）：

| 比较 | 泄漏 | 仍有效字符 |
|---|---:|---:|
| 8B：划掉的字 − 对照臂 | −29.3 [−34.5, −24.3] | −2.5 [−5.3, 0.0] |
| 8B：划掉的字（不用掩膜）− 对照臂 + 涂白（用掩膜） | +1.8 [−2.8, +6.5] | **+3.4 [+0.3, +6.2]** |
| 8B：划掉的字 − 原模型 | −26.0 [−31.2, −20.6] | −1.5 [−4.9, +1.8] |
| 8B：划掉的字 + 涂白 − 对照臂 + 涂白 | −5.3 [−8.0, −2.9] | −1.5 [−3.5, +0.1] |
| 2B：划掉的字 − 对照臂 + 涂白 | −2.7 [−8.2, +2.7] | +0.9 [−3.7, +4.8] |
| 2B：同上，只看划掉一部分的行 | **+12.8 [+2.5, +23.6]** | +0.9 [−3.7, +4.8] |

- 8B 上不用掩膜的适配器与「对照臂 + 涂白」泄漏没有差别（17 对 15），多保住约 3 个点的有效字符；2B 上整体没有差别，但划掉一部分的行上更漏（+12.8）。放大有帮助，但这是一个种子的两个点。
- 适配器的行为随「话」变：训练时用的是 `raw_final` 的话（只转写最终有效的）；换成中性的话（逐字转写）泄漏回到 33%（原模型的 43%、对照臂的 47% 之间还是低一些），说明它学到的一部分是「听从这句话」，不是无条件地删。
- 适配器 + 涂白最低（10%），有效字符与「对照臂 + 涂白」相同。
- 伤害：对没有涂改的 390 行，8B 划掉的字读成空 0.8%（3 行，对照臂 0），与对照臂读数的相似度 0.970（4.1% 的行低于 0.8）；有仍有效字的 153 个有标签的行里 4 行读成空（对照臂 0 行，2B 是 8 行；其中有标签本身错的行）；整行划掉的 122 行里 94 行读成空（原模型 7，对照臂 0）。
- 8B 的臂是用没排除同一学生两个编号的那份语料训练的（见上），只影响 3 个对照行。

## 第二篇的人工标注（整道题）

定位和口径见 CLAUDE.md「第二篇的划分、标注和评测口径」。现在（2026-10-05 起，第三版）每处涂改只标一个框（被作废的字在哪）和被划掉的字（写什么）。第一版「行框 + 每行文字 + `[[ ]]`」和第二版「几何」（划线一段段画、压到字就停）用户试过都太费时；几何版标好的测试集前 7 题（40 处）留作像素级的小考卷，用下面第 7 步读回。被划掉的字和笔画像素分开靠合成数据（第一篇的涂改金标贴到干净行上，答案由构造已知），不靠人工。流程：

1. 挑题：`python -m revrender.data.annotation_set --mapping index_manifest.json --same-student 对子.json --output-dir OUT`。书写者按第一篇划分（训练 / 第一篇的 dev 当验证 / 测试，同一学生的两个编号合并），每题至少 12 行、至少 3 行判为有涂改，按题号轮流、一位书写者一题，种子 42。现在的选择（`work/annotation_set`，服务器 `paper2_annotation_set`）：测试 22 题 417 行（目录判为有涂改 122 行）、验证 13 题 217 行（56）、训练 11 题 211 行（77）；题号 2–9；测试的前 5 题是试标。
2. RevRender：`revrender.pages` 在这些题上跑一遍（第一篇的冻结点、存档的 OCR 文本、印刷模板 `paper2_page_templates`；服务器 `paper2_pages_annotation_set_templates`：46 题、998 个块、422 个实例。不带模板的那次是 1,000 个块、423 个实例，几乎一样）。
3. 框的起点：`python -m revrender.tools.annotate.page_prefill --selection selection.json --pages 扫描目录 … --revrender PAGES_OUT --which test [--previous 几何版导出.json] --output-dir OUT`。每个 RevRender 实例的墨迹，连同它碰到的、外扩 14 × 18 像素窗口里连着的墨迹，取外接框（再放 3 像素）；细长的算划线、其余算涂黑；重叠超过小框 30% 的并成一处；落在剩余墨迹块（不是 OCR 行）上、比 1.5 行还高的实例丢掉（那是图形的线）。`--previous` 的题直接用几何版里每处的区域。按阅读顺序编号 r1、r2…。写 `OUT/prefill.json` 和一个行图包（`OUT/pack.json` + `OUT/crops/`，每处一张：框外扩 30 × 12 像素）。
4. 文字的起点：服务器上 `recognize_lines --pack OUT --conditions raw_struck`（只转写被划掉或涂黑的那部分，公式写在 `$ $` 里）。
5. 打包：`python -m revrender.tools.annotate.page_pack --prefill OUT --pages 扫描目录 … [--readings predictions.jsonl] [--first N] --output-dir PAGE` 写 `PAGE/标注.html`（一个文件，图、框、文字、规则都在里面，用 Chrome 打开；公式用 cdnjs 的 KaTeX 排版，没网就显示原文）。读数里的 □ 换成 ?，换行换成空格；涂黑默认「读不出」。包的 id 只由第 3 步的 `--which` 定：重新打包后，浏览器里已经改过的题保留。
6. 标注页（`page_editor.html`，规则在 `page_rules.js`，一份实现）：每处一个框，拖白点改形状、拖边中点加点、⌥ 点白点删点；空白处拖出新框；`2` / `1` 改划线 / 涂黑，`X` 读不出，`D` 删；右边每处一个输入框，下面实时排版；点一处或在输入框里回车 / Tab，图放大到那一处；记每题的用时（两次操作之间的间隔，每段最多算 60 秒）；自动保存在浏览器里，「导出」下载 JSON（`paper2-revision-box-v1`：每题的框、种类、文字、读不出、是否完成，和每题用时），「导入」接着标。
7. 几何版的读回（只用于那 7 题）：`python -m revrender.data.page_labels --export 涂改标注_….json [--export …] --selection selection.json --pages 扫描目录 … [--all] --output-dir OUT` 写每题的笔画和每处涂改（种类、像素数、区域）、掩膜（每笔、每处、每个区域）、去掉涂改后的扫描图（`OUT/removed/`），以及按划分的计数。划线的像素 = 画线两侧、按线下墨迹量出的笔宽之内的墨迹，方头、不越过画线的终点，再加不碰字的软边；涂黑 = 框内墨迹加软边。几何版的标注页在 git 历史里（提交 `1be20db`）。

## 合成涂改：第一篇金标的真实涂改（2026-10-05）

笔画和下面的字怎么分开，不靠人工逐笔标（太贵，见 RESULTS 里那 7 题的试验），靠合成数据：答案由构造已知。

1. 素材库（2026-10-05 改：只要划线，涂黑当读不出）：`python -m revrender.data.strike_bank --gold paper2_gold_pages --pages 扫描目录 … --mapping index_manifest.json --same-student 对子.json --output-dir OUT`（要 scikit-image，在服务器上跑）。第一篇金标放回整页、**未收紧**的带子（标注者沿划线画的线），只取第一篇训练书写者的（同一学生的另一个编号在别的划分里的不取）。带子比笔画宽（第一篇把细笔画加粗过，金标又经 96 × 768 条带来回缩放），按带子抠会把旁边的字一起抠进来；所以取带子的中心线，按中心线下墨迹量出这一笔的真实笔宽（到墨迹边缘距离中位数的两倍），只取中心线两侧半个笔宽（再加 1 像素软边）内的非白像素。比所在行高的 0.2 还宽的（涂黑块）、中心线短于 3 个笔宽的、端点多于 4 个的（乱划盖住整个词）、墨迹多于一笔应有的 1.6 倍的（带着字）去掉。服务器 `paper2_strike_bank_strokes`：保留 1,699 块（290 位书写者），去掉 998（带字 446、太短 419、乱划 93、涂黑 39）；`pieces.json` 记每块在原页上的位置，`left_out.json` 记去掉的和原因。（第一版按收紧的掩膜整块抠、划线涂黑都要：`paper2_strike_bank`，抽查约两成不干净，已不用。）
2. 贴：`data/paste_strikes.compose(..., bank=Bank(OUT), gold_share=…)`。给了素材库，一部分划法（`gold_share`）换成真实涂改：划线取一块、按两行行高之比缩放、沿被划的那段拉长（最多比行高比偏 0.5–2 倍），涂黑按行高比取几块真实的涂黑并排盖过去（学生是一个字一个字涂的）；都用宿主的墨色，随机左右、上下翻转。不给素材库时，抽样和以前一个像素都不差（同一配方画出同一张图）。
3. 答案：每个样本多给 `remove`，即去掉这处涂改要擦的像素（连淡边，不透明度 ≥0.1）。划线只擦不是字的像素（字连它的软边，灰度 <235，一个也不擦，所以被划过的字是完整的）；涂黑全擦（盖住的字本来就没了）。去掉之后的图 = 原图把 `remove` 涂白。抽样图在本机 `work/strike_previews/`。
4. 训练集：`python -m revrender.data.synthetic_set --corpus 读者语料/corpus.jsonl --bank 素材库 --pages 扫描目录 … [--copies 4] [--gold-share .7] [--second .3] --output-dir OUT`。第三步读者语料的每个配方（第一篇训练书写者的干净行，不含测试书写者和图包的书写者，5% 的书写者按哈希留作验证）用自己的种子画 `--copies` 份（原样的行一份）；七成笔画用真实涂改（划线和涂黑按素材库里的比例），整行只划不涂；贴入和叠放三成再加第二处。每个样本按读行的方式（灰度、BILINEAR、白底）缩放到第一篇的 96 × 768 条带，涂改编号随之缩放（NEAREST）：一处涂改一个实例，实例 = 擦掉它要擦的像素。服务器 `paper2_synthetic_strokes`：49,820 条（训练 47,202、验证 2,618），128 位书写者，46,888 条有涂改、10,556 条有两处；笔画 57,444 笔里真实划线 28,274、真实涂黑 11,952，程序画的七种各约 2,500。
5. 训练：`SYNTHETIC=1 MAPPING=index_manifest.json RECIPE=configs/revrender_strokes.toml bash scripts/launch_train_arm.sh control GPU NUMA 合成集 它的manifest的SHA - - OUT`：第一篇的 RevRender 和整条门（冒烟、精确重放、学习门、正式）不变，只换数据（`train --synthetic --mapping`：第一篇的验证 / 测试书写者一个都不许在里面）和目标（一处涂改一个实例，不按连通块切）；配方只改一处：不早停（`revrender_strokes.toml`）。每次验证都存权重（`step_<n>.pt`），不冻结：版本在用户的验证题上选。
5c. 框内去划线（用户 2026-10-05 定：找和擦拆开；涂黑当读不出、只做划线）：合成集 `--no-blots --crops`（不画涂黑；每处涂改按它作废的那几个字裁一个框，四周随机放宽 0–0.4 个框高，模拟人画的框；干净行裁 2 个词框，什么都不擦），服务器 `paper2_synthetic_boxes`（74,714 个框）；从第一组最后的权重微调（`revrender_strokes_boxes.toml`）。用的时候不让它自己找：`python -m revrender.box_remover --checkpoint … --point 0.4 0.5 0.1 --labels 框的导出 --selection … --split val --pages … --output-dir OUT` 在人画的每个划线框（加 6 像素）里读，写成整页输出的格式，`revision_boxes` 照常读；`sweep_checkpoints.sh` 加 `BOXES=… SELECTION=…` 对每个版本这样读。
5b. 第二组（只用合成数据在真实题上只找到三分之一，见 RESULTS）：合成集加 `--over 3`（每条宿主行再画 3 条「原地划掉 / 涂黑自己的 1–3 个字」，`paste_strikes` 的 `over`；第二处也可以是原地的），服务器 `paper2_synthetic_strokes_over`（58,616 条，原地 8,796）；训练从第一篇冻结的权重开始（`--init`），每批一半来自第一篇的真实行（金标 v2 数据包的训练行 + 用户校正的 2,127 行，`--bundle --corrected-bundle --real-share .5`；测试学生的另一个编号的行去掉，`--same-student`），一半来自合成集，循环内的验证读第一篇的开发行；配方 `revrender_strokes_mixed.toml`（学习率是第一篇的五分之一 6e-5、14,000 步、不早停）。启动：`SYNTHETIC=1 MAPPING=… SAME_STUDENT=… INIT=第一篇best.pt REAL=数据包 REAL_SHA=… REAL_CORRECTED=校正行 REAL_CORRECTED_SHA=… RECIPE=configs/revrender_strokes_mixed.toml bash scripts/launch_train_arm.sh control GPU NUMA 合成集 SHA - - OUT`。
5d. 对抗网络学怎么放划线（用户 2026-10-05 定：先试这条，约 24 小时全自动）：真样本 = 目录里判为有涂改的行（人和模型判的，约 97% 对），书写者只取第一篇训练书写者和第一篇没用到的（同一学生有验证 / 测试编号的不取），且第一篇 RevRender 在这一行找到的全是细长划线（含涂黑、盖写的行不要）；宿主 = 同样这些书写者判为没涂改、RevRender 也什么都没找到的行（`data.gan_sets`；没跑过 RevRender 的页先跑一遍）。形状不由网络发明：生成器（`models.strike_gan`）拿到一行干净字、噪声和发给它的几块真实划线（`strike_bank` 的 1,699 块），只学放在哪、拉多长多高、转多少、用几处、多深，用仿射采样贴上去（可求导，所以哪些像素是划线永远精确）；判别器看 96 × 768 的整行，真假两边都过同样的随机平移、模糊、明暗；铰链损失 + R1 + 多样性项，生成器取滑动平均（`train_gan`）。训练完由生成器在宿主行上按行的原分辨率重画、每处裁一个框（`data.gan_strikes`），训练框内去划线 A（第一组最后的权重起步）和 B（框内模型 step_8000 起步），验证集选、测试集记。整条：`bash scripts/run_strike_gan.sh OUT 小时数`。试车（`paper2_gan_smoke_*`）：真样本 3,448 行（389 位书写者）、宿主 24,085 行；600 步 39 秒。
6. 在用户的框上读：每个存下的版本整页推理一遍 `bash scripts/sweep_checkpoints.sh GPU NUMA OUT/formal ids.txt SWEEP`（即 `python -m revrender.pages --checkpoint step_<n>.pt --point 0.4 0.5 0.1 …`（`--point` 直接给掩膜阈值、NMS、低的分数线，不必先冻结），再 `python -m revrender.eval.revision_boxes boxes --labels 导出.json --selection selection.json --split val --pages-output 上一步 --pages 扫描目录 … --output-dir OUT`：一处涂改算找到 = 框里墨迹被擦掉至少 5%（且至少 15 像素）；框（外扩 4 像素）外被擦掉的墨迹连成至少 30 像素的一块算一处误擦。每个分数线给找到率、准确率（找到 / (找到 + 误擦)）、F1 和框外墨迹被擦的比例，F1 最好的分数线在验证集上就是选定的、在测试集上只记录。`OUT/pack` 是写了字的每处（不含读不出）的框加 6 像素：原图和擦掉之后；服务器上 `recognize_lines --conditions raw_neutral erased_neutral` 读，`revision_boxes read --pack OUT/pack --readings predictions.jsonl --output read.json` 给读数保住被划掉的字的几成（按 `eval.coverage` 的最少编辑对齐，`?` 不算），划线和涂黑分开。
7. 选版本（2026-10-05 定，看结果之前）：验证集上 Qwen3-VL-8B 读擦掉之后的框、保住被划掉的字的几成（可读的划线），最高的版本；同分看框的 F1。分数线用这个版本在验证集上 F1 最好的那条。测试集只在选定之后读一次，同时记下测试集上最好的版本和差距。

## 精细标注工具试验（2026-10-06）

目标是减少人工逐像素修订的时间；工具模型与论文最终方法分开。保留书写者划分，训练题的复核可用于后续微调，测试题的精细标注仅作评估。

`strike_bank --render traced` 沿原金标带子的骨架重新渲染，只从扫描图估计笔宽和墨色，不复制交叉文字的像素；`--render pixels` 保留原来按中心线邻域抠图的方法作对照。这是合成图层，不声称从扫描图恢复了交叉处真实的墨迹层。`synthetic_set --crops --union` 将一个框内所有待擦笔画合成一个目标，与每框取一个掩膜的推理方式一致。

首轮用 `scripts/run_stroke_review.sh OUT prepare` 生成两份材料，检查样图后执行同入口的 `train`。两臂使用相同语料、随机种子、初始化和训练预算，只改变素材渲染；每臂 4,000 步，配置在 `configs/revrender_strokes_review.toml`。素材边界会影响裁框：本轮抠图 / 重绘各 24,003 / 24,005 个框，共有 23,955 个样本 ID，其中 17,367 个原始裁框坐标相同；共有样本的宿主行、书写者与划分相同。训练完成后执行同入口的 `test`：先将验证集按字符保留率、框 F1、较早步数排序固定版本，再记录全部版本的测试结果及差距。OCR 字符保留率仅作辅助，最终省工效果必须由用户复核的操作数和用时衡量，不能由合成准确率推断。`read` 还报告规范化后严格转写匹配率 `exact`（不接受多输出字符，含 `?` 的标签不进入该分母）；历史 `whole` 只表示标签字符全部出现。

首轮完成（训练 `28930529`、评测 `61cd2b15`）：两臂均从原框内模型 step 8000 起步，按真实验证集 13 题的 41 个可读划线框选择 step 2000，选择文件先于测试生成。测试转写为 22 题中的 81 个可读划线框；像素为已有逐笔金标的 7 题，划线像素共 7,149。下表百分数不是人工省时比例。

| 模型 | 验证字符保留率 | 测试字符保留率 | 测试严格匹配率 | 擦除像素中划线占比 | 划线像素召回 |
|---|---:|---:|---:|---:|---:|
| 原扫描图 | 48.1% | 49.5% | 8.6% | — | — |
| 原框内模型 step 8000，本轮重新转写 | 60.4% | 53.6% | 11.1% | 55.2% | 56.8% |
| 抠图素材，step 2000 | 53.6% | 47.1% | 6.2% | 45.6% | 58.6% |
| 重绘素材，step 2000 | 56.5% | 57.4% | 12.3% | 54.1% | 58.5% |

本轮重绘相对同预算抠图的字符保留率差为 +10.3 个百分点，按 18 位书写者成组配对自助抽样（10,000 次，种子 42）所得 95% 百分位区间为 [+3.7, +17.9]；相对原框内模型为 +3.8，区间 [-2.9, +10.0]，未作多重比较校正。单次试验支持重绘优于同预算抠图继续训练，尚不能证明稳定胜过原模型；裁框随素材变化，也是比较的一部分。原模型此次 OCR 重读是 53.6%，历史那次 54.1% 保留不改。两臂与各自测试最好版本的差距分别为 3.14、0 个百分点；所有测试结果仅记录，未据此换版本。标注页继续用原框内模型预填。

合成验证各 1,251 框，`eval.validation_loss` 事后逐版本补记 eval 模式损失，不含训练专用去噪和采样 ROI 辅助项，因此不与训练批次的总损失直接比较。抠图 / 重绘选定版本的验证损失为 8.3194 / 6.9844，最终 step 4000 为 7.8742 / 6.4891；真实验证的选择不变。结果在服务器 `paper2_stroke_review_union_summary.json`、`_selected.json`、`_test_read/comparison_intervals.json` 和 `_loss_{pixels,traced}/summary.json`，本机小文件备份在 `work/stroke_review_results/`。首次验证 OCR 因 vLLM 不接受 UUID 设备编号失败，改用整数编号后从 `read-val` 续跑；旧失败记录保留，成功完成标记为 `paper2_stroke_review_union.test.COMPLETE`。

点击纠错单独训练：`python -m revrender.train_refiner --synthetic SET --mapping MAP --same-student PAIRS --recipe configs/stroke_refiner.toml --output-dir OUT`。小型条件分割网络输入原图、上一轮掩膜、补擦点和保护点；用合成目标模拟漏擦、误擦和纠错。合成训练与验证的宿主书写者分开，划线素材库共用；合成验证只用于工具版本选择，不代表真实题准确率。首轮选择 step 3000；事后 CPU 补算其原合成验证损失为 0.1349（`paper2_click_refiner/validation_loss.json`），未据此重新选版本。按页面的逐次点击、局部范围推理复核 96 个合成验证框：75 个误差下降、19 个不变、2 个上升，误差像素 95,468 → 69,220（减少 27.5%）；184 次点击在服务器 CPU 上计算中位 35 毫秒。此数不含网络与页面开销，真实题使用效果待用户复核。

标注页：`python -m revrender.tools.annotate.pixel_review pack --prefill TRAIN_PREFILL --checkpoint BOX_CHECKPOINT --pages SCANS --first 5 --name PACK_ID --output-dir OUT`（也可用 `--masks BOX_PREDICTIONS` 读取已有预测），再用该入口 `serve --pack OUT --checkpoint REFINER_CHECKPOINT --port 8769`；服务仅绑定服务器回环地址，可通过 SSH 本地转发访问。逐处看原图红色掩膜和擦除结果，点击补漏/保护文字，或画笔橡皮精修；可撤销、导入导出、整题补漏框，记录活跃操作间隔和操作数。无预填区域的题也可点题目标题检查。点击模型只能修改点击附近，当前显式点击优先。导出 `paper2-pixel-review-v1`，继续由 `data.page_labels` 读回原分辨率像素；未确认预测不能进入人工金标。首批 `paper2_pixel_train_review` 是 5 道训练题、20 个预填划线区域，起点为旧框内模型 step 8000，纠错用新 refiner step 3000；本机备份在 `work/pixel_review_train/`。界面验收用的 `paper2_pixel_qa` 数据不作为人工训练标签。

2026-10-07 按用户要求改为默认完整答题图，直接点击涂改区域进入局部精修。区域调整复用原 `page_rules.js` 的多边形命中规则：拖顶点改形状、拖边中点加顶点、Alt 点顶点删除，滚轮缩放、空格或右键拖动平移；空白处拖框补漏。“调整区域范围”可扩大裁图，掩膜按整页坐标搬入新裁图、不拉伸；撤销同时恢复范围和像素，导入导出保留修改后的顶点和裁图坐标。多边形表示作废内容范围，裁图四周仍保留 6 像素上下文，具体擦除由像素掩膜决定。

旧 5 题页面误用了原扫描图；全部 5 题都有同尺寸的清洁版本。用 `pixel_review refresh --pack OLD --clean-pages /home/zsh/hme/outputs/paper2_pages_clean_unlabeled3976 --output-dir NEW` 创建新包，保留题目、区域和包 ID，换入清洁 PNG，预填掩膜与清洁图的墨迹（灰度 <245）相交；输入哈希写入 manifest，旧包不覆盖。服务、显示、画笔和点击纠错使用同一套清洁图。浏览器保持原地址及存储键，第一次迁移前在本机 localStorage 另存原标注，可“导出更新前备份”；已有人工像素按原坐标保留，只剔除清洁后变白的像素，受影响区域取消确认、要求复核。导出附带图像来源；后续读回、微调的 `--pages` 也应使用这套清洁图。

之前的在线服务包为服务器 `paper2_pixel_train_review_clean20261007`（本机 `work/pixel_review_train_clean20261007/`），页面 / 服务源码 `4dbc123a`，仍用 8769 端口和原包 ID。5 张页面与既有清洁 PNG 逐像素一致；20 个原预填区域的掩膜无需删像素。20 项针对性回归检查通过，覆盖控制点增删、区域扩展保持原坐标、旧导出兼容、人工像素保留、清洁图换入与训练划分守卫；实际 CPU 服务已验证扩框后的纠错坐标、点击范围外不变及白像素禁止标记。

离线复核使用同一套编辑器：`python -m revrender.tools.annotate.pixel_review standalone --pack CLEAN_PACK --review-export HUMAN_JSON --output-dir NEW`。输出 `标注.html` 包含全部 PNG、模型预填和导入的人工进度，双击直接打开，无需服务器；离线时使用画笔 / 橡皮，不运行点击纠错模型。上下两张大图同步拖拽、滚动和缩放，支持完整裁切适配、最高 64 倍放大；空格 / 右键拖动，Enter 确认。错误框可以删除、撤销，或从左侧“已删除”恢复。导出的删除记录在 `deleted_revisions`，不混入训练用的 `revisions`。

新批次默认 80 个模型候选划线区域：`python -m revrender.tools.annotate.pixel_batch --selection TRAIN_SELECTION --mapping MAPPING --same-student PAIRS --exclude-export PREVIOUS_JSON --pages CLEAN_SCANS --revrender CLEAN_DETECTIONS --count 80 --output-dir SELECTED`，仅取训练书写者并排除已标注者及其关联编号，按完整题的候选区域凑足数量，不截断题目；随后 `pixel_review pack --prefill SELECTED --checkpoint BOX_CHECKPOINT --pages CLEAN_SCANS --review-clean --first 0 --name UNIQUE_PACK_ID --output-dir PACK`。新批次 `standalone` 可省略 `--review-export`，全部预填保持未确认。精修快捷键 **B 画笔、E 橡皮**，输入文字、输入法组字和组合键时不触发。

2026-10-07 第二批 `paper2-pixel-train-batch80`：80 处候选、30 道题、30 位不同书写者、8 个题号，旧框内去划线模型 `paper2_strokes_boxes/formal/step_8000.pt` 预填，全部区域有非空预测。服务器产物为 `paper2_pixel_batch80_selection20261007` / `paper2_pixel_batch80_prefilled20261007`，本机离线包 `work/pixel_batch80_offline20261007`，桌面入口 `划线精细标注_第二批80处.html`。检测与预填都需人工复核，不是人工金标。上一批最新 JSON 原样归档至 `work/pixel_review_human19_20261007/annotations.json`：19 处已确认、5 题均未标记整题完成，独立保留，本轮仅运行预填推理，未启动微调。

精修范围以用户多边形为准：用多边形的精确外接范围分配画布（不额外扩边），按原生像素中心判断框内，框外显示白色且不能画笔标记、自动清理或送入在线纠错模型；不拉伸或矫正手写图像。凹多边形可绕开相邻行。打开精修时同时适配窗口宽高，完整显示裁切；仍可继续同步缩放和平移。旧数据也按同一规则读取。框外的既有修正保存在 `outsideMask`，缩框再扩框可按原坐标恢复；只有框内 `runs` 导出为有效掩膜，框外留存不作为训练标签。更新 HTML 保持原路径和草稿键，并备份旧文件。

残留清理按用户 2026-10-07 截图固定为灰度 ≥100、掩膜外 4 像素、孤立碎点 ≤12 像素，灰边和碎点两项开启。载入 / 导入整批标注、画笔结束、模型纠错和调整范围后自动清理并保存，不再预览确认或调参数；保留深色文字邻域和切块边缘的保护规则。自动清理作为用户授权的统一处理保留已有确认状态，人工修改仍按原规则取消确认。清理结果写入原生像素掩膜，导出 / 训练读回一致。橡皮手动擦掉的掩膜像素记录为原生坐标 `manualKeep`，优先于自动清理及模型建议；保护随自动保存、JSON、续标副本和多边形缩放保留，只有主动用画笔重涂才解除，撤销 / 重做同时恢复保护。保存处理策略及结果，重复载入不继续扩张擦除范围；撤销后也记录当前结果，刷新不重做被撤销的清理。初次整批自动处理可撤销，浏览器草稿和内嵌草稿照旧保留。

只升级离线界面时使用 `pixel_review update-html --source EXISTING_HTML --output-dir NEW`：完整保留内嵌图像、草稿、包 ID 和图像版本，避免因重新生成包而换掉浏览器存储键。备份旧文件后，在同一路径替换为新 HTML，浏览器草稿仍用原键读取；无需重新导入旧 JSON。

2026-10-10 精修改为 **1「擦划线」／2「保原字」二选一**，每处记住所选模式，只显示该模式的标记。两种模式都有 **B 画笔、E 橡皮擦**：模式 1 画红色要删除的划线，橡皮撤红并保护该像素免于自动清理；模式 2 画绿色要保留的数字／文字／符号，橡皮撤绿，框内未涂绿的墨迹全部擦白，下图实时更新，无额外预览／应用按钮。两套标记分别保存，切换、撤销、导入导出、缩框扩框不会覆盖另一模式；范围变化后重新计算当前模式结果。旧模型预填和人工修正归模式 1；旧版已画的绿色保护在首次进入模式 2 时作为起点。原自动清理配置仍为 100 / 4 / 12；模式 2 的绿色笔画全部保留。桌面 `划线精细标注_人工框选108处.html` 沿用原 26 题、108 处预填及存储键，更新与原文件备份在 `work/pixel_two_modes108_20261010/`。

“文字答案”可选，填写被划掉的内容（包括学生写错的内容）；局部看不清用 `?`，整处无法辨认另勾选。`text_confirmed` 与像素 `confirmed` 分开，不填写文字也能确认像素。导出仍为 `paper2-pixel-review-v1`，`annotation_version=3`：`annotationMode` 指明当前模式，`runs` 始终是当前模式要删除的原生像素，训练读取无需取反；`keepMarks` 保存模式 2 的绿色笔画，`eraseDraft` 保存切入模式 2 前的模式 1 像素状态，两者不直接作为训练标签。旧 `manualKeep` 橡皮保护及 `uncertainMask` 未知记录在模式 1 中继续保留，模式 2 不混入这些标记；未知可用模式 1 的画笔／橡皮重新判定。二值训练／几何评估读取器仍拒绝活动区域含未知像素的标签。页面修改不改变模型或数据划分、不运行训练。

每次修改自动存浏览器 localStorage（画笔中途也保存），状态栏提示成败。静态页不能无授权改写自身：“保存可续标副本”会下载含当前进度的完整 HTML；支持文件系统访问的浏览器还可手动绑定 JSON 备份文件，当前会话内逐次自动写入。换浏览器、换位置或清理浏览器数据前应保留续标副本或 JSON；重新打开时选择浏览器草稿与 HTML 内嵌草稿中较新的进度。

离线包额外调用 `review_clean`，复用 `tighten.non_white` 的深色墨迹（<200）及相连两像素软边（<240），其余浅灰像素置白；不改变原生尺寸和深色像素，也不改变历史实验的清洁默认值。`images/<UUID>.png` 是后续微调所用的同源图像。原人工 JSON 原样备份在 `source_annotations.json`，清洁后标注在 `imported_annotations.json`；受影响的确认状态撤回并提示复核，人工形状、有效像素、备注和用时保留。2026-10-07 提供的部分标注含 22 区域、10 已确认、0 整题完成；清洁从标注中剔除 187 个变白像素，9 个已确认区域需复核。尚不启动微调。


收到真实复核后，可在上述 `train_refiner` 命令增加 `--init REFINER_CHECKPOINT --review-export EXPORT.json --selection SELECTION --pages SCANS --review-share .25`，并指定新的微调配方及输出目录。每批混入已完成训练题的真实裁块；读回使用与界面相同的像素中心多边形，框外置白并裁去掩膜，忽略恢复编辑用的 `outsideMask`；误检被确认成“不是涂改”的区域作为全零负例。入口拒绝未确认区域、非训练题、第一篇验证/测试书写者，以及合成验证宿主书写者；默认数据路径不含任何界面验收导出。本轮尚未进行真人复核后的微调。

## 人工框选与真实修正微调

用户 2026-10-07 改为：先用已有逐处确认的像素修正微调框内模型，再由用户在整题上框选新区域，模型预填后逐处精修。`data.review_set` 接受像素导出中 `confirmed=true` 的现存区域；不伪造整题完成，不把删除或未确认记录当标签。按书写者留出独立微调验证集，原生多边形外涂白，训练图像和标注页同源；加入小批原合成正负样本。沿用 `train` 的 smoke、精确重放、训练集学习门槛和正式训练。配方 `configs/revrender_strokes_human.toml`，从原框内模型 step 8000 开始。

- 空白框选包：`pixel_review select-regions --selection TRAIN_SELECTION --pages CLEAN_PAGES --mapping MAPPING --same-student PAIRS --exclude-export PREVIOUS_JSON --count 30 --name PACK_ID --output-dir PACK`；再用 `standalone --pack PACK --output-dir SITE` 得到单文件 HTML。按完整题显示，只拖框、编辑多点范围、标记本题框选完成；约 80 处后导出。框选完成和像素确认分开记录，空框不作为负样本。
- 模型预填：`pixel_review prefill-regions --pack PACK --regions-export FRAMES_JSON --checkpoint CHOSEN_CHECKPOINT --output-dir PREFILLED`，然后 `standalone`。按用户多边形裁切，独立预测每处，保持区域 ID 与坐标；新包 ID 不覆盖框选草稿，所有预测待人工确认。拒绝覆盖已有像素标注。
- 验证：`python -m revrender.eval.review_removal --dataset REVIEW_SET --baseline ORIGINAL_CHECKPOINT --checkpoints RUN/formal --output-dir EVAL`，比较独立原生验证区域的划线召回、擦除精确率、有效墨迹误擦率。固定原预填阈值和 top 1，报告自动残留清理前结果；F1 提高且精确率下降不超过 2 个百分点才替换原模型。测试集本轮不用于选择或调参。

当前标注辅助模型（2026-10-07）：`/home/zsh/hme/outputs/paper2_pixel_finetune20261007/formal/step_800.pt`，由 `paper2_pixel_finetune_eval20261007/summary.json` 的原生验证结果选定。66 处确认中 50 处训练、16 处按书写者留作验证，具体数字见 RESULTS。后续框选入口为桌面 `划线区域框选.html`，源包 `work/pixel_region_selection20261007` / 服务器 `paper2_pixel_region_selection20261007`，30 道新题、初始空框；离线包 `work/pixel_region_selection_offline20261007`。用户导出约 80 个框后用此模型运行 `prefill-regions`，再生成精修单文件。验证对比页在 `work/pixel_finetune_eval20261007/验证对比.html`，两处真实多边形已完成预填集成检查。不同推理批量在阈值边缘出现 1 像素差异，已记入 `work/pixel_region_prefill_smoke20261007/verification.json`。

## 补被划文字（2026-10-11）

用户定：论文单位是「一处修改」（输入整题、输出每处被划内容，教育分析看改对没有），像素精修过的区域补上划线下面的原字，作训练和预实验数据；对错由模型对照标准答案先判、用户逐条确认，不让用户另标。框只框被划掉的字（与测试 / 验证框同口径），写被划掉的原字，改正后的字不写。

- 裁切：`python -m revrender.tools.annotate.struck_text crops --export 全量导出.json --review paper2_full_review20261010 --pages 扫描 ... --split train --split val --output-dir OUT`。只取确认且未删除的区域；划分沿用全量预标包（已并入第一篇书写者、同学生对和固定验证书写者），测试书写者的区域不进。每处存扫描小图（给 VLM 起稿）、清洁小图和按用户擦除结果涂白的小图，每页存清洁整页。
- 起稿：`recognize_lines --conditions raw_struck`，Qwen3-VL-32B-FP8 TP2，与测试 / 验证框标注的起稿相同。
- 页面：`struck_text page --crops OUT --export 全量导出.json --readings predictions.jsonl --questions 第一篇 questions_with_answers.jsonl --answer-groups 第一篇 answer_groups.jsonl --output-dir PAGE`，生成像素精修页的文字模式（`task: strike_text`）：PAGE/pack、PAGE/site/标注.html。用户的像素、多边形、两种模式和画笔 / 橡皮全部照旧（`pixel_editor.js`），右栏是被划文字（`pixel_text.js`；打 `\` 选符号，`symbol_picker.js`）、模型初稿和题面 / 标准解答；Enter 同时确认像素和文字。离线页同时存 localStorage 和 IndexedDB（读回核对），可绑定备份文件。第一次打开时自动接入旧「补被划文字」页存在浏览器里的进度（`text_source_pack`），调过的框按新框裁切、像素待再确认。之后改编辑器只能用 `pixel_review update-html` 更新现有页（存储键含图像版本）。题面与标准解答来自第一篇 `line_edit_question_answer_authority_20260814_v1`（22 道考试题，派生参考答案），作答图经 `line_edit_response_index_20260814_v1` 对到题。
- 改起稿（用户 2026-10-11 定）：在用户精修后的图上读，只读框选的多边形。`struck_text erased --crops OUT --export 全量导出.json --review paper2_full_review20261010 --output-dir ERASED`（擦除像素涂白、多边形外涂白，留 4 像素白边），`recognize_lines --conditions erased_neutral`（Qwen3-VL-32B-FP8，服务器最大的 Qwen 识别模型），再 `struck_text update --source 桌面页 --readings ... --output-dir NEW` 原位更新（包 ID、图像版本、进度不变）；编辑器只替换既未确认、也未改过（仍等于旧初稿）的文字。服务器 `paper2_struck_text_erased20261011`、`paper2_struck_text_erased32b20261011`；215 处中 204 处初稿变了。注意：初稿来自 VLM 读擦除图，后面「先擦后读」的评测要换别的模型读，并在论文里写明。
- 对错初判（用户 2026-10-11 定：模型先判、用户确认）：对照标准解答和整份作答逐处判断被划内容对不对、改后对不对、修改类型（纠错 / 改错 / 错改错 / 抵消 / 笔误重写 / 等价改写 / 换思路 / 删除多余 / 放弃 / 不确定）和理由，存 `work/annotation_exports/修改对错初判_2026-10-11.json`（`paper2-revision-judgment-draft-v1`，216 处，作者为模型）；`struck_text update --source 桌面页 --judgments 该文件` 原位加入页面，右栏「修改对错」可改，Enter 同时确认像素、文字和对错；导出字段 `judgment`、`judgment_confirmed`。论文里须写明这批标签是"模型初判、人工逐条确认"并报告改动比例。
- 本轮：导出 `全量划线标注_2026-10-11.json`（服务器 `paper2_annotation_labels/`）的 238 处确认区域中，训练 199（含第一篇未用书写者 79）、验证 16，测试 23 未放入。服务器 `paper2_struck_text_crops20261011`、`paper2_struck_text_struck32b20261011`；本机 `work/struck_pixels_page20261011`，桌面 `涂改精修与文字标注.html`（包 ID `paper2-struck-pixels-5a688fece9ad`）。第一版独立文字页只写 localStorage、失败只闪提示，用户约 150 处文字没有存下；已并入本页并在无头 Chrome 的 file:// 下验证接入旧进度、真实按键与鼠标、刷新后仍在。
- 诊断实验（2026-10-11，方案 `docs/struck_diagnostics_plan.md`）：`ALLOW_GPU3=1 bash scripts/run_struck_diagnostics.sh OUT`（可续跑；`eval.struck_diagnostics build / score`）。结论见 RESULTS「被划内容诊断实验」：分离是瓶颈，擦除优于标红，现有擦线方法连字一起擦。
- 第二阶段（2026-10-12，方案 §12.1）：`eval.revision_judgment build --config configs/revision_judgment.toml --output-dir PACK`，`recognize_lines --pack PACK --conditions from_pack`（每行自带提示和系统提示），`eval.revision_judgment score --pack PACK --judgments NAME=predictions.jsonl`。结论见 RESULTS「第二阶段」。
- 符号与渲染（2026-10-22，用户提）：常用栏分组（最近 / 几何 / 关系 / 运算 / 结构 / 编号），大括号（`cases`，里面 Enter 换一行，Ctrl／⌘+Enter 才确认），带圈数字 ①–⑳ 面板（公式外是普通字，公式里写 `\text{①}`），全等判定 SSS 等；`\大括号`、`\圈3`、`\全等` 可搜；插入走浏览器自己的编辑，Ctrl+Z 能撤；多行公式按整块渲染，没写完的公式保留上一次的渲染并提示；预览旁放这一处的原图。`struck_text update --katex KATEX_DIST` 把 KaTeX 与字体内嵌进页面（离线也能渲染）。评分的统一写法（`data.ocr_lines.normalise`）：≌ 与 ≅ 同一个、∽ ∼ ~ 同一个，`\left\{ … \right.`、`array` 的列格式、`&` 当排版去掉，① 记作 1；验证集 41 处标签不受影响，216 处里 1 处、测试导出 431 处里 8 处的写法因此变了。测试页换前在无头 Chrome（独立配置、断网）里验证：导入用户 10-22 导出后换新页，447 处除当前处的计时外逐字段相同，旧页备份在 `work/page_backups/`。
- 测试集标注（2026-10-12，用户定：像素和被划文字一次标完，全部用 203 处版 step400 预填，论文写明"测试像素由该模型预填、人工修正"）：`annotation_set --lines test=1432` 取全部 72 道合格测试题（≥12 行、≥3 行有涂改；前 22 题即原测试题），`work/test_annotation_set20261012`。服务器 `paper2_test_review20261012.sh` 跑清洁页、RevRender 检测、tighten（与全量复核同检查点、OCR 文本、模板）；`paper2_test_text20261012.sh` 用 step400 `pixel_library infer`（带入全量导出里这 72 题的 17 处人工记录），`pixel_library prefill-export`（预填作导出，重复框按库的规则合并），`struck_text crops --unconfirmed`、`erased`，32B 读擦线小块（`erased_neutral`）作文字起稿，`struck_text page --condition erased_neutral`。桌面 `测试集涂改与文字标注.html`（包 `paper2-struck-pixels-192d8eaf313e`，516 处，511 处有起稿）。之后改编辑器用 `struck_text update`。先前只含像素的测试库未使用，已移到 `work/page_backups/`。
- 移出同门数据并重跑（2026-10-21）：名单 `configs/allowed_writers.txt`，守卫 `revrender/data/allowed.py`（`python -m revrender.data.allowed scan DIR` 列出目录里其他书写者的页）；`python -m revrender.tools.replay --from 旧输出 --to 新输出 --map 旧=新 --set=--选项=值` 按旧输出 run_manifest 里记录的命令重跑；`review_set --fixed-validation 旧集` 新建集时保留冻结的验证；`struck_diagnostics build --detector --box-remover`；`run_struck_diagnostics.sh` 的卡可由 C1/I1/C2/I2/PAIR/PAIR_INDICES 指定；全部重跑 `scripts/run_labmate_rebuild.sh`。
- 移出同门标注集（2026-10-18，用户定）：只用未标注集书写者的页面（`--allowed-writers`：服务器 `/home/zsh/hme/work/replacement944_allowed_writers.txt`，本机 `work/labmate_guard/`，只有 UUID，不进 Git）；选题 `annotation_set --allowed-writers ... --exclude 旧selection --lines test=all`，建包 `pixel_library infer --allowed-writers ... --pages A --pages B --detector A --detector B --ids`。测试页重建：服务器 `paper2_test_text20261018.sh`（用户导出的进度并入 60 道保留题，13 道补题用预填），`struck_text crops --unconfirmed` 保留删光框的题（`empty_pages.json`），`page` 保留用户原文字（含空）。详见 RESULTS「移出同门标注集的测试题」。
- 全量库换预填（2026-10-12）：同一输入只换检查点为 step400，`paper2_full_review20261012_step400`；页、区域 ID 和分组 ID 与 10-10 完全相同，模型版本变化后页面只替换未确认、未修改、未删除的区域（Chrome 中验证）。旧桌面目录移到 `work/page_backups/全部涂改标注_before_step400_*`。

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

`python3 -m pytest -q`：本地全过（需要 torch / skimage 的测试跳过）；服务器上全部通过（见 RESULTS.md 里各次验证的 `pytest.log`）。含一条行为测试（训练数据永不产出测试集的行）、与旧实现逐项比对的增广 / 采样器金标测试、
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

### 恢复 baseline 与预测素材配对（2026-10-07，首轮完成）

用户授权 DocDiff、DE-GAN、NAFNet 在 4090 的 0、3 号卡上复现并适配本任务；允许 `ALLOW_GPU3=1`。先做相同预算的任务适配对照，不把它称为复现原论文所有数据集的指标。`data.restoration_pairs` 从原始干净宿主构造干净/受损对，不把预测擦除后的图当干净真值。三组为人工金标素材、预测掩膜与墨迹相交后筛选的素材、同批预测路径重绘素材；相同宿主、裁框、位置、随机种子。验证宿主书写者和已有人工微调验证书写者均不进入素材库，第一篇 dev/test 书写者不进入训练宿主或素材。合成验证采用各组相同的独立程序划线，仅作受控比较；真实场景需另用人工验证/测试读数。

入口 `python -m revrender.baselines.restoration`，队列 `scripts/run_restoration_pilot.sh`。首轮每组 4,000 步 / batch 4 / 256×256；每 1,000 步在独立人工 16 区域上按固定亮度增量 >0.2 的擦除 F1 选版本，同时记录有效墨迹误擦、新增深色像素和固定 64 张合成验证图的 PSNR、改动区域 MAE、保留字迹 MAE。人工擦除标签并不提供被覆盖笔画的真值，这些读数不能证明隐藏文字恢复正确；还需选定模型后的转写评测。

直接加载固定提交的官方源码，逐文件校验，源码清单在运行产物中。NAFNet：`megvii-research/NAFNet@2b4af71ebe098a92a75910c233a3965a3e93ede4`，官方 width32、encoder [1,1,1,28]、PSNR loss、AdamW 和梯度裁剪；DocDiff：`Royalvice/DocDiff@f422a3af10657eeabeca90f2e862d88c72725845`，32 通道双网络、100 步扩散、PRE_ORI、高低频损失，短预算用原始权重不做 EMA；DE-GAN：`dali92002/DE-GAN@7281836455608337c940fa0de67cceffbc5c026d`，官方 Keras 1024 瓶颈生成器和判别器，按发布代码 MSE 对抗项 +100 BCE 重建，两次 D 更新后一次 G 更新。DE-GAN 用独立 TensorFlow 2.16 环境和旧 Keras 兼容层，不改共享环境。数据构造用已有印刷模板排除印刷墨迹占比 >2% 的宿主行。

试跑数据 `paper2_restoration_pairs20261007_v4`：10,788 训练配对 +596 合成验证配对，排除 84 行印刷宿主；预测库留下 202 份候选（并非人工确认的纯划线），像素版和重绘版一一对应，额外排除 46 份非横向细长碎片，避免把竖向碎片强行拉宽。宿主底稿保持不变，交叉处也有合成真值。金标素材与预测素材的数量、覆盖形态不同，因此两者对照同时包含素材质量与多样性的影响。

像素审计发现 v3 的普通 alpha 叠加会把更黑的底层墨迹变浅，约占改动像素 5%；测试评估前已停止旧队列。v4 取 `min(底稿, 叠加图)`，仅纠正变浅像素，其余宿主、素材、位置、划分和随机数全部不变，目标仍是原底稿。`data.restoration_pairs darken --source SOURCE --output-dir OUT` 支持可审计的等价修复；新构造器默认就用只能加深的公式。旧队列有 `SUPERSEDED.txt`，不作为最终实验。

测试入口 `scripts/eval_restoration_pilot.sh`：要求两卡的 9 个训练作业全部完成后，只加载人工验证选定版本。`baselines.restoration_export` 在人工标出的真实涂改区域中恢复，导出同一批原图/恢复图给固定 Qwen3-VL-8B、中性转写指令，统计已有 `revision_boxes.read` 的字符保留与严格完全正确率；另读 7 题逐笔测试金标，固定增亮阈值统计误擦漏擦。此处使用人工区域，是去划线模块的条件评测，不是整页检测到恢复的端到端成绩；读取测试结果后不能回头选 checkpoint。

首轮 9 组训练和冻结后的真实测试均已完成，完整结果见 RESULTS 末节；本机离线对比页 `work/restoration_pilot20261007/index.html`。本轮最高字符读回为 NAFNet 原金标素材 54.54%（同次原图 48.88%）；新素材尚未稳定提升文字读回，路径重绘减少部分误擦。仅为短预算任务适配，现有人工标注预填模型未因此替换。

### 六个已发表恢复系统的同协议适配（2026-10-08）

用户授权使用 GPU0/2/3。清单与出处见 `BASELINE_RESEARCH.md`；入口复用 `baselines.restoration`，配方 `configs/restoration.toml`，队列 `scripts/run_restoration_suite.sh`。在上一轮冻结的 darken-only v4 / gold_pixels 上从头训练：COR-Generator、普通 Strikethrough-CycleGAN、NAFNet、Restormer、DocDiff、DE-GAN。先只用合成数据；50 处真实训练标注不混入，16 处人工验证照旧用于选版；无测试图、人工掩膜或真实划线类型输入模型。VCGAN 仍是待核对候选，本队列不含它。

COR 和 CycleGAN 最大 80,910 次 batch4 更新，相当于 30 轮的期望见样本量（有放回采样，不宣称逐轮遍历）；NAFNet/Restormer/DE-GAN 最大 80,000 步，40,000 步后连续 6 次验证不提升停止；DocDiff 最大 100,000 步，50,000 步后同规则。每 5,000 步及末步验证，最大真实验证擦除 F1 选版，固定增亮阈值 0.2；同时记录误擦和新增墨迹。64 张固定合成验证图仅作诊断。这里保留旧像素选版口径，尚未采用调研建议的转写验证选版；更长预算仍不等于复现原论文完整 benchmark，也不保证已收敛。

原架构与损失保留。COR 使用反色灰度 BCE；CycleGAN 使用独立抽样的干净/划线两域、双向 L1 cycle×10、LSGAN、20 张历史池，不使用配对监督或属性标签。Restormer 使用完整 dim48/[4,6,6,8] 架构和 L1，batch4 以单图累积实现，保持原生 256 裁块；官方 flat/cosine 调度按当前预算同比例缩短，不使用 GoPro 的 128→384 渐进尺寸或旋转增强。DocDiff 同时保留 raw 和严格按源码更新的 EMA（初始复制、step>2000 每100次、decay .9999），两者是预先声明的验证候选，不看测试挑选。EMA 早期接近随机初始化的问题会如实记录。

所有权重按冻结选择重新加载，与保存的原生验证预测比较，最多允许 1 灰度级浮点差异。训练保存输入/源码哈希、优化器、步数、验证曲线和设备峰值。共同推理采用原生 256 网格重叠裁块，不拉伸手写字；恢复图之后不加方法专属清理。公开源码仅存忽略产物目录，Git 只保存适配代码。

完成入口 `scripts/finish_restoration_suite.sh SUITE UPSTREAM` 等待三条训练队列成功后，调用已有 `eval_restoration_pilot.sh --suite`，只评估冻结权重，在空闲 GPU3 上运行共同 Qwen3-VL-8B，并生成 `SUITE/report/index.html` 与 `summary.json`。模型或验证失败则中止评测并留错误标记；不以不完整模型表冒充最终比较。100 步冒烟已经在三卡全部通过，六个选定模型的验证图重载一致（DE-GAN 最大 1 灰度级浮点差异）。

### 作者公开权重的直接验证（2026-10-09）

用户要求先下载作者已经训练好的模型，再在我们的验证数据上选候选。入口 `python -m revrender.baselines.pretrained`：`prepare` 只接收既有划分中完成标注的可读验证区域，`catalogue` 保存下载文件哈希和预先声明的选版规则，`infer` 严格加载匹配的官方架构／参数，`report` 检查转写完整性并生成离线比较页。合并图包和 Qwen 读取复用 `baselines.restoration_export merge`、`recognize_lines`、`eval.revision_boxes read`。

下载来源见 [BASELINE_RESEARCH.md](BASELINE_RESEARCH.md) 第 12 节：CycleGAN 普通及辅助判别器各 5 份，DocDiff 去模糊／去印章，NAFNet GoPro／SIDD width32，Restormer 去模糊，DE-GAN 去水印／去模糊／二值化，共 18 份。COR 尚未确认作者权重，Restormer 去噪官方链接下载配额超限；不能用本项目训练权重替代后声称直接测试作者模型。下载清单在产物 `paper2_official_weights20261009/download_manifest.json`。

本轮不训练；11 题的 41 处可读划线按固定 Qwen3-VL-8B 字符读回率选版，同分依次看严格完全正确、较低正常墨迹误擦和名称。另 1 处可读涂黑不用于划线排名，独立 16 处人工像素验证用于统计漏擦与误擦。CycleGAN 使用作者缩放／补白几何再逆映射，其余原生 256 裁块；均不加标注界面的自动残留清理。来源、尺寸和预训练任务不同，这是作者权重的跨域适用性比较，不能与上一节同数据重训的测试成绩混排。结果目录 `paper2_pretrained_validation20261009/report/`，精确数值与限制见 RESULTS。

### StainRestorer 补充基线（2026-10-09）

用户授权先验证 WACV 2025 StainRestorer 三套官方权重，再做同数据适配；WACV 作为补充实验，不计入此前按 CCF 筛选的六个核心系统。出处：[作者代码](https://github.com/CXH-Research/StainRestorer)、[Weights release](https://github.com/CXH-Research/StainRestorer/releases/tag/Weights)。原版源码及下载权重保留在独立产物目录，记录 Git 版本、文件哈希和 URL。

入口复用 `baselines.pretrained` 和 `baselines.restoration`。推理取作者训练验证使用的第一个输出，即最终图；官方 `infer.py` 把两个输出组成的列表传给指标的问题在适配入口纠正，模型源码不改。污渍／水印／印章三种官方权重在既有 41 处划线转写验证及 16 处像素验证上比较，固定原生 256 裁块，仍按字符读回选版；不能据此选择正式训练初始化。

正式适配从随机初始化开始，与六基线共用 10,788 对 darken-only v4 gold_pixels、596 对合成验证及 16 处人工选版区域，不混入 50 处真实训练标注。使用作者两级输出 MSE + 0.2×(1−SSIM)、AdamW（2e-4，weight_decay=0.01）、缩短到任务预算的 cosine（终值 1e-6）；有效 batch4，由单图梯度累积实现，SSIM 每图计算与作者 batch1 配置一致。最大 80,000 步、40,000 步起连续 6 次验证不提升停止，每 5,000 步验证；选版按人工擦除 F1，与既有六基线相同。随机种子 42，灰度复制三通道，不旋转、不拉伸长行、不采用作者 MixUp／CutMix，不加专属清理。此为任务适配，不是原文 300 epoch／1024 测试协议的完整复现。

`scripts/run_restoration_candidate.sh CARD MODEL DATA UPSTREAM WEIGHTS VALIDATION_INPUT OUT` 串联 100 步训练／重载检查、官方权重验证与离线报告、从头正式训练、冻结重载、既有 81 处可读划线测试和 7 题像素测试、最终报告。任一阶段失败即停并留 `FAILED`；输出目录只新建，禁止重复启动。`run_restoration_suite.sh` 可在第七个参数起指定补充模型；`eval_restoration_pilot.sh --runs` 只接受已经重载验证的训练结果，避免重跑六个已完成系统。GPU 限 0/2/3，3 号卡须空闲；额外 `torchmetrics==1.4.0.post0` 装到独立依赖目录，不改公用环境。已有标注预填模型不替换。


### 灰度挖洞预标注试验（2026-10-10）

`revrender.gray_holes` 仅使用扫描灰度、第一篇检测掩膜和用户框选范围，人工擦除答案只在报告中评分。固定四组：直接擦除、内缩 1 像素并交墨迹、局部灰度挖洞、额外要求两侧墨迹支持。参数为 19×19 局部墨迹直方图（8 灰度一级）、至少 12 个参考像素、更黑阈值为中位数−max(16, 0.5×IQR)、候选扩 1 像素；两侧支持在四个方向、12 像素范围内检查，允许 1 像素横向容差。灰度候选不是已确定的原字，不应直接覆盖人工标签。

入口 `python3 -m revrender.tools.gray_holes_view --review-export work/pixel_finetune_inputs20261007/annotations_1.json --review-export work/pixel_finetune_inputs20261007/annotations_0.json --frames-export /Users/zhongsuhua/Downloads/划线精细标注_paper2-pixel-region-selection_2026-10-07.json --images work/pixel_finetune_inputs20261007/images --frame-pack work/pixel_region_prefill108_20261007 --detector work/pages_run_complete --count 100 --output-dir NEW`。全取 66 处已确认，种子 42 从另 108 个人工框中抽 34 处，包含 5 处无检测覆盖的区域，不挑效果好者。第一篇输入是原页推理的 complete 检测存档，扫描使用与人工标注同源、同尺寸的清洁图；没有拿框内恢复模型预填或人工擦除掩膜当算法输入。图像与输出保持原生坐标。

输出 `对比.html` 是内嵌全部图像的独立静态页面，可切换处理前／后方法、查看绿色挖洞、上下放大对比及人工结果。`selection.json` 记录每处来源、范围和像素统计，`proposals.npz` 存候选擦除掩膜（按行 packbits，形状单独记录），`summary.json` 保存总体结果，manifest 记录输入哈希并在结束时复核原文件未变。只在 66 处计算按区域累计的像素指标，34 处只展示；66 处包含已有训练 50／验证 16，属于探索性结果，不能当独立测试。没有训练或 VLM 评估。产物 `work/gray_holes100_20261010/`，桌面 `灰度挖洞_100处前后对比.html`；原标注与 108 处编辑页未改变。

对比页缩放以原生像素为准：默认同时适配宽高，小图最多放大 2 倍；提供缩小、原始大小和完整显示，四图同步横纵滚动。切换样本重新适配，切换方法保留倍率。显示修复不重算试验或覆盖原标注。


灰度交叉比较扩展：同一入口另存 `work/gray_crossing100_20261010/`，沿用相同种子与 66＋34 选择。新增 `global/joint/structure` 三组以 `direct` 为共同擦除底图，不做掩膜收缩。人工多边形内灰度 <200 且连通块至少 3 像素的墨迹参与分析，灰度均值减去 max(16, 0.5×标准差) 为整体候选阈值；联合组再要求通过 19×19 局部中位数阈值；结构组再要求 8 方向的双侧连续掩膜外锚点及横向掩膜内墨迹。均至少 12 个统计像素，统一补 1 像素且只在分析墨迹内保护。浅灰排除仅影响统计和候选，原图不变，原掩膜其他像素继续擦除。结构规则不能保证识别原字，饱和黑和完全包住的文字可能无候选；不按本批人工评分调参。新页面默认对照 `direct/structure`，可选全部 7 组；保留原页面和标注。


### 全量预标注与自由筛选（2026-10-10）

用户取消每批 80 处限制：全部检测掩膜一次预标，人工决定筛选量和导出时机。`eval.review_removal --candidate NAME=PATH` 在固定 16 处原生人工验证上重评冻结候选，沿用 mask=.4 / score=0 / top1，以原框内模型为参照，精确率下降不超过 .02 且 F1 提高才可替换。此轮原框内、人工微调、重绘、抠图、GAN A/B 六个候选中选人工微调 step800；无测试选版、无新训练，不把本轮像素 F1 与历史 OCR 分数混排。指标不含页面自动残留清理。

`python -m revrender.tools.annotate.pixel_library infer --selected SELECTION/summary.json --pages CLEAN_PAGES --detector TIGHT_DETECTIONS --mapping MAPPING --same-student PAIRS --validation FIXED/validation_native.json --review-export EXPORT [--review-export EXPORT] --output-dir NEW` 覆盖检测目录的全部实例，不按数量、分类或分数再筛除。用检测墨迹的相连区域与既有优化行高度构造初始框，用户可调整多边形。检测实例都有 coverage 对应记录；若已确认／修改／删除的人工作业与候选框重叠超过较小框的一半，保留人工记录代替重复候选。ID 绑定原检测序号；新预测不确认、不当标签。每页先使用已有 `review_clean`，检查尺寸，保存模型与输入哈希。

`pixel_library bundle --pack FULL_PACK --output-dir NEW` 生成可直接打开的目录型离线库：统一 index.html、每 20 题一个内嵌图片编辑页，仅为加载分页，不是标注额度。入口可自由跳组／继续上次位置，一次导出所有已查看进度或已确认训练标签。编辑器保留双模式、B/E、Enter 确认、N 跳过、多边形调整、删除、撤销与固定自动清理；IndexedDB 由入口集中存储，各子页面通过来源校验的消息调用，避免 file 页各存一份。导航等待保存完成；保存失败明确提示导出。浏览器数据仍应导出备份，不能当永久文件备份。

旧标注导入复用现有逐像素校验，导入前下载全量备份；同 ID 已确认／修改／删除记录优先保留，新人工框可替代重叠自动候选。已知导出中的人工记录内嵌保留；其他旧页面尚未导出的浏览器草稿不读取、不覆盖，用户从旧页导出后可导入总入口。模型更新时仅刷新未确认、未修改、未删除的预测，已人工处理的掩膜及两套标记不变。分页 ID 由题目 ID 列表确定。

划分由第一篇书写者映射、关联编号与固定 16 处验证书写者共同确定；验证／测试标注可查看，但训练导出排除它们，并排除未确认、删除和不确定区域。后续微调必须继续使用已冻结的验证区域，不得重新随机划验证集，也不能直接把全量草稿交给训练。此次只完成预标与页面，不启动下一轮训练。

全量页重复框修复：离线打包时将同页、上下边位置接近且覆盖较小框至少九成的自动框做保守合并；要求组内两两满足，避免串联误并相邻行。预填掩膜按原坐标取并集，所有原记录仍保留。首次载入先备份浏览器草稿，再仅合并整组未被人工处理的候选；人工修改、确认、删除、改框均优先保留。重复记录放入 `duplicate_revisions` 归档，不重复显示、不进入训练导出。原包 ID、图像版本和存储键不变。浏览顺序按题目、从上到下排列；精修的上一处／下一处／确认固定在双图上方，双图按剩余高度上下排列，备注与说明折叠。

增量复核微调：`python -m revrender.data.review_set --base-set PREVIOUS_SET --review-export TRAIN_ONLY_EXPORT --pages MATCHED_IMAGES --mapping MAPPING --same-student PAIRS --output-dir NEW_SET` 保留旧训练与合成回放、逐字节复制固定原生验证。输入先按可信书写者划分筛掉测试和验证；入口再次拒绝其关联编号及固定验证书写者。第一篇未使用的书写者可进训练池。`scripts/run_review_round.sh DATASET INIT GPU_UUID NUMA OUT` 跑原配方的学习检查、正式微调和相同原生验证；不自动替换标注页。`python -m revrender.tools.review_removal_view --dataset SET --evaluation EVAL --output-dir NEW_VIEW` 生成可切换各保存版本的离线同图对比。首组结果及小验证集的取舍见 RESULTS。
