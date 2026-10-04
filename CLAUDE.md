# CLAUDE.md

本文件给在本仓库里工作的 Claude Code 会话。完整说明（结构、出处表、等价验证结果）见 `README.md`。

## 任务

第二篇论文（2026-10-04 用户定，面向计算机方向，不做教育评分）：涂改感知的多模态识别。第一篇的涂改分割是上游，整页涂改掩膜（`tighten` 之后只含墨迹）当先验；问题是 VLM 读手写数学作答时，划掉的被读成有效（泄漏）、有效的被读坏（误删）。第一步（README「识别试验」）已做：用旧项目已有的人工前后对（整行的划掉前 / 后文字，不是只含划掉那段）和盲判无涂改行，五个 VLM 读原图时持有被划掉字符的 36–47%（泄漏），提示、红色标记、文字坐标、规模都没用（0–6 点），涂白掩膜下整列最强（12–13%），代价是仍有效字符少 5–8 点（上界，约一半是标签问题）；剩下的泄漏主要是没有掩膜的行（10%）和随机下限，所以单看泄漏不够支持新方法，结论是有条件放行：方法按「同样泄漏下比涂白多保住多少有效字符」判。第二步（视觉 token 屏蔽，不训练）已做：隐藏 token 和涂白像素是同一件事，所有变体落在原图与涂白之间的一条取舍线上，不是方法，留作基线。第三步（在合成涂改上训练 LoRA，不用掩膜）已做：8B 上泄漏 43 → 17、有效字符 87 → 86，与「对照臂 + 涂白」泄漏持平、多保住约 3 个点有效字符；收益的主体是整行划掉的行（学会说「没有」），划掉一部分的行 8B 持平、2B 不如涂白；只学整行划掉会把有效字符掉到 61，贴入的样本才是保住有效字符的原因；评测标签噪声大（40 行里 16 行有问题），绝对水平不可靠，方向可靠。

本仓库的主体是第一篇代码的干净移植（RevRender、评测、基线、工具），**已在服务器上证明与旧仓库等价**。整道题的数据管线和训练/验证/测试划分由用户另定。

识别试验的规矩（用户 2026-10-03 授权，10-04 改定）：第一步和视觉 token 屏蔽不训练；之后的 LoRA 只在合成涂改上训练（`data/paste_strikes`、`data/reader_corpus`、`train_reader`，用户 10-04 说「按顺序往下走」授权），宿主行只取第一篇训练书写者的干净行、不含图包里任何行的书写者（图包的评测行不能和训练同一只手）、不含测试书写者，语料文字（学生的字）只放服务器和本机、不进 Git；不改原始数据、不读第一篇测试书写者的行；评估用旧项目的人工前后对（`data/prior_annotations`），指标是划掉字符的泄漏、仍有效字符的保留、重写字符的保留（`eval/coverage`，按字符对齐；泄漏永远和有效字符的保留一起读，新指标先拿「读数恰为标签」「空读数」「别的行」三个参照检验再用），不算整句 CER；没有人工答案时不报告准确率，指标是「待评估」不是零；检测阳性不等于人工涂改；B 对 C 错不能直接说成遮白的因果损伤，按指令读出划掉的字也不是不遵循指令。Codex 的两轮（`work/recognition_pilot40`、`work/recognition_revision80`）没有人工答案，只是记录，不当证据。同一时间只让一个窗口改仓库。

## 数据位置（只读，不复制）

- 原始数据：`~/Desktop/cv_idea/未标注数据/`、`~/Desktop/cv_idea/标注好数据/`。文件夹名含学号，只读；UUID ↔ 学号的映射表（`id_mapping*.csv`）不进 Git，处理链路一律用 UUID。
- 行目录：旧仓库 `gold/line_edit_all5002_policy_decisions_20260814_v1/canonical/exact_crops_46902.jsonl`（46,902 行、4,990 张题目图，每行有 `image_id`、`box_1000`）。读取：`revrender.data.lines.default_catalogue()`。
- 路径配置：`configs/paths.toml`（本机，已被 git 忽略）；模板 `configs/paths.example.toml`。服务器上第一篇数据包、权重、上游源码的位置在 README 的数据表。
- 题目图（5,002 张）在哪、`image_id` 怎么对到文件：见 `docs/data_location.md`。副本以 `image_id` 命名、与原件逐字节相同，读图不需要映射表。

## 红线

- 原始数据不改不删；图像、权重、npz、npy、`id_mapping*` 不进 Git。提交前先 `git add -An .` 干跑，确认没有 `.jpg/.png/.pdf/.zip/.pth/.pt/.npz`。
- 训练数据永不出现测试集的行；不读测试集。`tests/test_no_test_rows.py` 守着这条，改了数据读取代码必须先过它。
- 种子只用 42（`revrender.util.SEED`）。
- 输出目录已存在就报错、不覆盖；每次运行写 `run_manifest.json`（配置、种子、git 提交、输入文件哈希）。
- 代码之间不用 SHA-256 钉住，不按编号互相加载脚本；仓库里除 README 出处表外不出现旧脚本编号（`tests/test_repo_hygiene.py`）。
- 服务器（`ssh 4090x4`）：
  - 代码以快照部署到 `/home/zsh/hme/code_paper2_<提交号>`，产物写 `/home/zsh/hme/outputs/paper2_*`，不写旧的 `work` 目录；
  - 持 GPU 锁（`scripts/_common.sh` 的 `hold_card`），每卡最多两个我们的作业，3 号卡有人用时不用；
  - 不杀他人进程，不往公用环境装包；`pkill -f` 会杀掉自己的 ssh，先 `ps` 取 PID 再 `kill`。

## 常用命令

```bash
python3 -m pytest -q                                   # 本地：torch 相关测试自动跳过；服务器上跑全部
bash scripts/deploy.sh                                 # 部署当前提交（要求工作区干净）
bash scripts/launch_gpu_job.sh GPU NUMA OUT -- 命令    # 持锁跑一个短 GPU 作业（OUT 必须不存在）
bash scripts/launch_train_arm.sh control GPU NUMA BUNDLE SHA CORRECTED CSHA OUT   # smoke_a→smoke_b→learning→formal→freeze
bash scripts/launch_native_arm.sh MODEL GPU NUMA BUNDLE SHA CORRECTED CSHA OUT    # 原生基线：smoke→formal→freeze→阈值
python -m revrender.infer / .train / .eval.report / .eval.freeze ...                # 都有 --help
```

服务器上用 `PYTHONPATH=<快照目录> /home/zsh/hme/envs/hme/bin/python -m revrender.xxx`；原生基线用 `/home/zsh/hme/envs/native_baselines_v1/bin/python`（只读，不装包）。

## 写代码的约定

- 普通 Python 包 `revrender/`，正常 import；命令行入口在各模块的 `main()`，`pyproject.toml` 里有 `rvr-*` 别名。
- 配方在 `configs/revrender.toml`、`configs/baselines.toml`，训练时整份写进 manifest；改一个值就是一个声明过的实验。
- 模型的建造顺序和随机数流是模型的一部分（见 `revrender/models/legacy_init.py` 的说明），不要“顺手整理”；改了就要重新核对 smoke 前 4 步的轨迹（`python -m revrender.tools.compare_trace`）。

## 长期规矩：不要让代码再变成旧仓库

旧仓库每做一个实验就新写一个编号脚本、复制一份旧代码改几行、加一个模式开关、写一段流水账，久了同一个功能有十几份实现，谁也不敢删。从这里起：

- 不写编号脚本，不为一个实验复制一份代码。新实验 = 配置文件 + 已有入口的参数；新能力加进已有模块，或新开一个有明确名字的模块。加东西前先问：能不能改已有代码？
- 同一功能只有一份实现。一个实验分支结束（不进论文、不再用）就删掉它的代码和开关，历史留在 git 里。不留「以防万一」的开关、注释掉的代码、`_v2` / `_new` / `_old` 文件（文件名由 `tests/test_repo_hygiene.py` 守）。
- 不在代码里硬写哈希常量、不写大段防御性校验；溯源靠 run manifest。
- 文档只留 `README.md`、`CLAUDE.md`、`RESULTS.md`（每次正式实验一行：日期、配置、提交、输出目录、关键数字、一句结论）。不写流水账。
- 产物目录按运行名（配置里的 name）区分，不按复制的脚本区分。
- 单个文件超过约 800 行就拆；模块边界保持 models / data / train / infer / eval / baselines / tools。
