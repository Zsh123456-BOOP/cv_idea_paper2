# 题目图在哪、怎么对上 `image_id`

2026-10-03，只读调查。没有写数据管线，没有定第二篇的划分。本文只出现 UUID 和数量：学号、姓名、原始文件名都不写；学号 ↔ UUID 的映射表只在本机，不进 Git。

## 结论

1. 题目图（整道题的扫描）一共 **5,002** 张。行目录的 4,990 张都在其中，另外 12 张没有合格的行。
2. 每一张都有一份**以 `image_id` 命名、与原件逐字节相同**的副本，分在两个互不重叠的文件夹里。读一张题目图不需要映射表：文件就是 `<image_id>.jpg`，没有一个 id 对到两张图。
3. 行目录的 `box_1000` 就是第一篇行图的来源：在服务器的这两个文件夹里，按框裁出来、再按旧流程缩放，与第一篇数据包里的行图逐像素相同（训练 / 验证行随机抽 150 条，150 条全部相同）。
4. 需要你拍板或留意的有两件：11 名学生有两个书写者编号，其中 2 名一个编号在训练、一个在测试（第 6 节）；第一篇的行级金标只覆盖有金标的题目图里约四分之一的行（第 5 节）。

## 1. 图在哪

| 来源 | 张数 | 原件（只读，文件名含学号） | 副本（用这个） |
|---|---:|---|---|
| 三张卷的主扫描 math1 / math2 / math3 | 872 / 680 / 788 | `~/Desktop/cv_idea/未标注数据/<卷>/<学号>/<学号>_<题号>_<小问>.jpg` | `~/Desktop/cv_idea/work/unlabeled_anon/images/<uuid>_<题号>_<小问>.jpg` |
| 补充资料包 math1 / math2 / math3 | 551 / 544 / 541 | `~/Desktop/cv_idea/未标注数据/补充资料包/<卷>/<学号>/<学号>_<题号>_<小问>.jpg` | 同一个文件夹，`<uuid>_s<题号>_<小问>.jpg`（题号前加 `s`，免得和主扫描撞名） |
| 外部已标注集 | 1,026 | `~/Desktop/cv_idea/标注好数据/test/<学号>_<题号>_<小问>.jpg`（70 张多一个批次段） | `~/Desktop/cv_idea/work/line_edit_external_full1026_images/<uuid>_…jpg` |

服务器上的副本：`/home/zsh/hme/data/unlabeled_v1/images`（3,976 张）和 `/home/zsh/hme/data/full1026_v1/images`（1,026 张），与本机同名同内容（张数相同，各抽 60 张核了字节）。

`image_id` 的形状说明它来自哪个文件夹：

| `image_id` | 张数（行目录里） | 副本所在 |
|---|---:|---|
| `<uuid>_<题号>_<小问>` | 3,289 = 2,336（主扫描）+ 953（外部集） | 主扫描在 `unlabeled_anon`，外部集在 `line_edit_external_full1026_images`，同一个 id 只在其中一个里 |
| `<uuid>_s<题号>_<小问>` | 1,631 | 补充资料包，`unlabeled_anon` |
| `<uuid>_<批次段>_<题号>_<小问>` | 70 | 外部集（试卷第 17 题），`line_edit_external_full1026_images` |

uuid 的第三段首位也能区分来源：`5` 的是未标注 / 补充（3,967 张），`4` 的是外部集（1,023 张），两者没有例外。

## 2. 要回到原件时怎么对

一般用不着：副本和原件逐字节相同（3,976 + 1,026 张全部核过）。

- 未标注 / 补充：映射表 `~/Desktop/cv_idea/work/id_mapping_unlabeled.csv`（本机、不进 Git），三列 `batch, original, uuid`，561 行、499 名学生；`batch` 是 `math1/2/3` 或 `supp_math1/2/3`。原件 = `<对应目录>/<original>/<original>_<题号>_<小问>.jpg`；副本名把 `original` 换成 `uuid`（补充的题号前加 `s`）。主扫描里 math2 的 62 名学生也出现在 `supp_math3`，同一个 uuid。
- 外部集：没有映射表。副本名 = 原名的第一段（学号）换成随机 uuid、其余不变；1,026 张按内容哈希一一对上，没有重复。
- 这次没有生成任何含学号的文件。

## 3. 数量

行目录：`~/Desktop/cv_idea_claude/gold/line_edit_all5002_policy_decisions_20260814_v1/canonical/exact_crops_46902.jsonl`（只读；服务器旧代码快照里有同一份，SHA-256 `eaf3aaeb…`）。

| 来源 | 题目图 | 在行目录里 | 行 | 平均每张几行 | 书写者 |
|---|---:|---:|---:|---:|---:|
| math1 | 872 | 870 | 7,262 | 8.35 | 110 |
| math2 | 680 | 680 | 6,780 | 9.97 | 108 |
| math3 | 788 | 786 | 9,250 | 11.77 | 107 |
| 补充 math1 | 551 | 548 | 3,518 | 6.42 | 73 |
| 补充 math2 | 544 | 543 | 4,979 | 9.17 | 86 |
| 补充 math3 | 541 | 540 | 5,900 | 10.93 | 71 |
| 外部集 | 1,026 | 1,023 | 9,213 | 9.01 | 146 |
| **合计** | **5,002** | **4,990** | **46,902** | **9.40** | **640** |

- 三张卷合起来：math1 1,423、math2 1,224、math3 1,329 张；外部集没有记录它属于哪张卷。
- 每张题目图的行数：最少 1、中位数 8、第 90 百分位 18、最多 110。
- 书写者列按来源数，合计 640 是去重后的（math2 的 62 名学生也在补充 math3 里）。

## 4. 第一篇划分下的书写者

登记的划分映射（数据包的 `index_manifest.json`）：

| | 书写者 | 行 | 题目图 |
|---|---:|---:|---:|
| 训练 | 346 | 3,808 | 1,439 |
| 验证 | 51 | 476 | 205 |
| 测试 | 44 | 476 | 191 |

三者两两没有重叠。测试只数了编号和数量。另外，用户校正的 2,127 行（校正行）来自 297 名书写者：195 名在上面的训练 346 名里，另 102 名不在登记的映射里；它们都不是验证 / 测试书写者（读取时核过）。所以第一篇训练用到的书写者是 346 + 102 = 448 名。行目录的 640 名书写者里，第一篇一行都没用到的有 97 名。

## 5. 第一篇的行级金标覆盖了多少

没有碰测试：下面只算数据包的训练行、验证行和校正行。

- 含行级金标的题目图 **2,238** 张，其中 **1,569** 张至少有一处涂改。按来源：math1 456、math2 326、math3 446、补充 math1 242 / math2 314 / math3 273、外部集 181。加上测试的 191 张是 2,429 张。
- 这 2,238 张在行目录里共 25,534 行，有第一篇行级金标的是 6,345 行（24.8%）：平均每张 11.4 行，其中 2.8 行有金标。要给整道题完整的金标，其余行还得补。
- 2,127 条校正行里有 64 条的 line id 不在行目录里（它们所在的题目图在）；其余对得上。

## 6. 需要你拍板或留意的

1. **同一个学生有两个书写者编号。** 11 名学生既在外部集（随机 uuid）、又在未标注 / 补充集（另一个 uuid），共 85 张图在未标注 / 补充、51 张在外部集。第一篇的划分下：1 名两个编号都在训练；2 名一个编号在训练、另一个在测试；其余 8 名只有一个编号被第一篇用到（4 名是校正行、4 名是训练）。所以第一篇的测试和训练之间有 2 名学生以不同的编号同时出现，第二篇按学生划分时要先把这 11 对合并。对照表在 `work/same_student_pairs.json`（本机，被 Git 忽略，只有 uuid）。
2. 外部集属于哪张卷没有记录；要按卷划分的话，这 1,026 张得另行判断。
3. 没有别的歧义：图源、数量（5,002 = 3,976 + 1,026；4,990 + 12）、id 与文件的对应都对得上。

## 7. 行是谁切的

行目录里的框来自 OCR 模型 **Unlimited-OCR**（Baidu 开源、MIT；服务器 `/home/zsh/hme/models/Unlimited-OCR`，6.4 GB）。它把整道题读成带坐标的版面区域 `<|det|>类型 [x0, y0, x1, y1]<|/det|>内容`，框是千分比坐标。切行的规则（`revrender/data/ocr_lines.py`）：

- 区域内容按换行和 `\\` 拆成视觉行；一个区域拆出多行时只能把区域框等高均分（旧记录里叫 `inferred`）。
- 目录里只留「严格行」：不是页码 / 图（`page_number`、`image`、`image_caption`），一个区域恰好一行，框有正面积；相同的框只留一条；顺序先左栏后右栏、栏内自上而下。
- 在第一篇训练 / 验证 / 校正用到的 2,238 页上，这套规则从 OCR 原文逐页复现了目录里的 25,534 条裁图：框、顺序、文字完全相同。
- 候选行里有 11.7% 是「一个区域多行」的均分行，第一篇没有用，目录里也没有；整页流水线同样不切它们。

原始 OCR 输出（模型对 5,002 页各读一遍）：未标注 / 补充 3,976 页在本机 `~/Desktop/cv_idea/work/line_edit_all5002_authority_v1/ulocr_3976.jsonl`、服务器 `/home/zsh/hme/annot_v1/ulocr_dedup.jsonl`（两份 SHA-256 相同，`3e5ee451…`）；外部集 1,026 页在本机 `~/Desktop/cv_idea_claude/outputs/line_edit_external_blind_v1/ocr_v1/shard_0[0-3].jsonl`、服务器 `/home/zsh/hme/artifacts/line_edit_external_blind_20260811_v1/ulocr/shard_0[0-3].jsonl`。每行 `{image_id, text}`，`text` 就是上面的区域文本。运行方式见 `revrender/ocr.py`（提示词 `<image>document parsing.`，与当时相同）。

## 做法与范围

- 对应关系：原件 ↔ 副本按文件名规则和映射表、再按内容哈希核对；行目录、数据包的记录只读 id、划分和数量。
- 裁行复现：训练 / 验证行随机抽 150 条（种子 42），用服务器上的题目图和行目录的框，`PIL` 转灰度后按 `data.lines.crop_line` 裁、`data.gold.letterbox` 缩放（双线性、白底 255），与数据包的行图比。
- 范围：核对字节一致时对全部图像文件算了哈希（包括测试书写者的图），没有解码、查看或分析它们的内容；内容层面的核对（裁行复现）只用了训练 / 验证行。测试书写者只统计了编号和数量。

## 8. 旧项目已经有的人工标注和模型识别（2026-10-04，只读调查）

旧项目 `~/Desktop/cv_idea`（本机；`configs/paths.toml` 的 `[prior] root`）里，同一批行已经有人工标注和三个模型的转写，键都是行目录的行 id（`<image_id>:L<nnnn>`）。读取：`revrender.data.prior_annotations`。文字是学生作答，只放本机的 `work/`，不进 Git。

| 内容 | 文件（相对 `[prior] root`） | 行数 | 字段 |
|---|---|---:|---|
| 涂改前后对（人工） | `outputs/revision_pair_annotation_20260822_v1/key.jsonl`（uid → 行）+ `gold/revision_region_annotation_exports_20260829_v1/returned_annotations/revision_pair_returned_v1.json` | 300 | `text_before`（划掉的原文）、`text_after`（重写的新文）、`ink_kind`、`no_rewrite`、`before_unreadable`、`no_revision`、`illegible` |
| 同上，另一批 | `outputs/combined_annotation_export_20260820_v1/export.json`（B_ 开头）+ `outputs/combined_annotation_page_v3/task_b_key.jsonl` | 200 抽，103 有填写 | `edit_kind`（strike_rewrite / strike_only / whole_line / overwrite）、`text_before`、`text_after` |
| 盲判有无涂改（人工） | 同一个 export.json（A_ 开头）+ `outputs/policy_label_audit_sample_v1/sample_blind.jsonl` | 500 | `has_edit`：433 无、67 有 |
| 三个模型的行转写 | `outputs/ocr_line_all46902_results_20260816_v1/server_original/{qwen_v1,infinity_v1}/crop_predictions.jsonl` | 46,902 | `model_text`、`ulocr_text`；Qwen3.5-35B-A3B-LaTeX-OCR（`FutureMa/Qwen3.5-35B-A3B-LaTeX-OCR`）、Infinity-Parser2-Pro；提示要求保留划掉的字；vLLM 四卡 |
| 涂改区域标注（第一篇金标的原始标记框 / 折线） | `gold/revision_region_annotation_exports_20260829_v1/download_exports/` | 2,100 条不重复 | `marks`；与 `returned_annotations/` 有重叠，用前按样本键去重 |

合并后 903 行：有前后对的 327（训练书写者 215、验证 23、没用到 56、测试 33）；盲判无涂改的 433（253 / 26 / 121 / 33）；盲判有涂改但没写前后文的 67；配对标注者判「没有涂改」的 22；其余（看不清等）54。前后对和盲判两批没有重叠的行。第一篇测试书写者的行只计数，不读。

旧项目里和第二篇相关的其余记录，都不是识别或转写：`complete_answer_*`（整题判错的预标注和三条件提示实验，用 Qwen3.5-122B / Qwen3.6-35B，评判错题，不是转写）、`removal_transcription_probe_v1/v2`（Qwen3-VL-8B 读 175 个涂改行：原图、涂白、还原三档，要求保留划掉的字；划掉的文字被读出 48% / 35% / 36%）、服务器 `/home/zsh/hme/ftv_code_9a7b264`（VLM 判错项目，11 个模型，结论是整合而非感知的损失）。服务器上的 VLM 权重在 `/home/zsh/hme/models`（Qwen3-VL 2B/4B/8B/32B-FP8、Qwen2.5-VL-7B、InternVL3.5-8B、GLM-4.1V-9B、MiniCPM-V-4、Gemma-3-12B / 4B、Llama-3.2-11B-Vision、Phi-4-multimodal，以及 DeepSeek-OCR、GLM-OCR、GOT-OCR2、MonkeyOCRv2、PaddleOCR-VL、Unlimited-OCR）。
