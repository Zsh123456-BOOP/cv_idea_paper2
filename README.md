# revrender

第二篇论文（整道题的涂改识别）的代码库。第一篇（RevRender + ExamRev，涂改实例分割）的模型、训练、评测、基线和工具，从旧仓库整理成一个普通的 Python 包。旧仓库的脚本是编号的一次性实验，互相按路径加载、用 SHA-256 钉住；这里是正常的包、正常的 import、一个命令行入口一件事。

- 移植自旧仓库 `~/Desktop/cv_idea_claude`，分支 `codex/claude-three-mechanisms`，提交 `aa8e2e01`（只读，未改动）。
- **已复现第一篇验证集数字**，三项等价验证的结果见 [等价验证](#等价验证)。
- 第二篇的数据划分、整道题的数据管线不在这里（由用户在另一个窗口讨论，定了再接）。本仓库只放第一篇已经证实的东西，不含研究决定。

## 结构

```
revrender/
  util.py            摘要、create-only 写入、冻结路径守卫、git 状态、run manifest
  config.py          读 configs/*.toml（路径、配方）
  determinism.py     严格确定性：配置 + 前向不变、反向确定的三个算子
  models/            RevRender：backbone.py（ResNet-50 + 细节通路）、detector.py（查询检测器）、renderer.py（框内逐点渲染器
                     + 上下文残差）、quality.py（掩膜质量头）、revrender.py（组装 + 四个变体）、losses.py（损失 + 匹配 + 分组裁剪）、
                     ops.py（采样与框）、legacy_init.py（只为复现随机数流而建、随后丢弃的模块）
  data/              bundle.py（数据包读取 + 训练池 + 测试集守卫）、extra.py（用户校正行）、sampling.py（平衡采样 + 预取）、
                     augment.py（增广）、gold.py（金标栅格化 / letterbox / 笔宽校正）、lines.py（行目录）、journal.py（写作者子集、合成阳性）
  train.py           训练循环：smoke → 精确重放 → 仅训练集学习门槛 → formal（早停、挑步）
  infer.py           推理与后处理：packed 预测存档、deliver（掩膜阈值 / NMS / 分数线）、predict_lines（把冻结点用到一批灰度行上）、命令行
  eval/              matching.py（组件匹配 + F1 + 前沿 + 误报预算）、ap.py（掩膜 AP）、bootstrap.py（按书写者自助法）、
                     freeze.py（阈值冻结）、fates.py（找到 / 画对）、speed.py（推理计速）、inloop.py（训练中验证读数）、
                     report.py（重新打分、与已记录数字和另一份存档比对）
  baselines/         native/（七个原生实例分割基线 + Mask Transfiner：factory、train、threshold、transfiner、dense、metrics、compat）
                     component_cls.py、removal.py、paper.py、mathscrub.py（涂改专用方法）、common.py
  tools/             annotate/（标注页 editor.html + 单文件打包）、compare_trace.py / compare_curve.py（两份训练轨迹 / 验证曲线逐项比对）
scripts/             deploy.sh、launch_gpu_job.sh、launch_queue_runner.sh、launch_train_arm.sh、launch_native_arm.sh、verify_equivalence.sh、_common.sh
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
| 掩膜 AP / 找到画对 / 自助法 / 计速 | `python -m revrender.eval.ap / .fates / .bootstrap / .speed` |
| 一个原生基线 | `launch_native_arm.sh MODEL GPU NUMA B S C CS OUT`（smoke → formal → 冻结 → 掩膜阈值）；Mask Transfiner：`MODEL=transfiner` 并设 `TRANSFINER_PIN=pin.json` |
| 涂改专用方法 | `python -m revrender.baselines.component_cls / .removal / .paper / .mathscrub`（都有 `--smoke`） |
| 训练轨迹 / 曲线比对 | `python -m revrender.tools.compare_trace 旧/trace.json 新/trace.json`；`python -m revrender.tools.compare_curve 旧/curve.jsonl 新/curve.jsonl` |
| 重做全部等价验证 | `scripts/verify_equivalence.sh OUT_ROOT REF_DIR [probe]`（REF_DIR 放第一篇记录的副本，见脚本头）；约 25 分钟（加 `probe` 另加约 15 分钟） |
| 重做七个原生基线的掩膜阈值冻结并与第一篇比对 | `scripts/verify_thresholds.sh OUT_ROOT REF_DIR`（三张卡并行，每个模型约 12 分钟） |
| 标注页 | `python -m revrender.tools.annotate.single_file --package 包目录 --output 页.html` |
| 原生环境 | `baselines/native/environment.sh`（建 `native_baselines_v1`；服务器上已建好，只读使用） |

消融：`--variant control|nofine|nocontext|nodense`（第一篇用到的三个消融）。其余旧脚本里的调参模式、历史分支不搬。

## 数据

原始数据（`~/Desktop/cv_idea/未标注数据`、`标注好数据`）只读，用 `configs/paths.toml` 指过去，不复制。第一篇用的数据包在服务器上：

| 用途 | 位置 | `run_manifest.json` 的 SHA-256 |
|---|---|---|
| 训练 + 验证金标 v2 数据包（476 条验证行、51 名书写者、365 处涂改） | `/home/zsh/hme/work/revision_bundle878_gold_v2/morphology_s42` | `574da74b6444699a4c375077c0aa9ff5819569692ba134a32db50d17da8acddf` |
| 用户校正的 2,127 行 | `/home/zsh/hme/work/revision_corrected861_v1` | `65a2db20b0a4ea563c153b7c1a1b63c42f45e68c65b63df86212201edcd9e429` |
| ImageNet R-50（detectron2 格式） | `/home/zsh/hme/work/native_baselines_weights_v1/R-50-torchvision.pkl` | `16fd03f0b0123e85b959cb9b64aaf9eeb4a7dc68bd9b0b134a9220d4b24c15c6`（文件） |
| 冻结的 RevRender（挑步 19,000） | `/home/zsh/hme/work/revision_tune880_v1/control/formal/best.pt` | `9f8ae601f1db17e1bb2e164d2e63344756bc3719a44a0e47f7e720a43a978153`（文件） |
| 原生基线上游源码 / 权重 | `/home/zsh/hme/work/native_baselines_sources_v1`、`_v2`、`_transfiner_v1`、`native_baselines_weights_v1` | 见各自的 pin / compatibility_manifest |

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
| 699、782 | `tools/annotate/editor.html`、`single_file.py` |
| launch_gpu_job.sh、launch_queue_runner.sh | `scripts/launch_gpu_job.sh`、`launch_queue_runner.sh`（队列目录改为 `paper2_queue_v1`，不碰旧队列） |

## 等价验证

@@EQUIVALENCE@@

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
- 数据包构建器（生成数据包的旧脚本）、标注协议与补标链不搬；题目图在原始数据里的确切路径尚未核对（行目录只给 `image_id`）。

## 已知差异

- 原生基线的反向梯度范数在两次运行之间相对差 ≤ 4e-5（detectron2 的 RoIAlign 等 CUDA 原子加）；前向、损失、存档逐位一致。
- 本地没有 torch / detectron2 / segmentation_models_pytorch，模型类测试在本地跳过，在服务器上跑全部。
