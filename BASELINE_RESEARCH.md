# 划线下手写内容恢复：Baseline 调研与实验清单

更新日期：2026-10-09。对象：第二篇论文，学生完整作答中的划线分离、文字保护及被划内容识别。第 1–11 节保留 10-08 的调研记录；第 12–13 节补充近期文献与官方资源核查；**第 14–16 节核查原论文训练／测试数据，并比较行级与整题两种论文方案**。实际实验结果另见 RESULTS，旧章节中的执行状态以日期和最新实验记录为准。

本文区分“论文已正式发表”“发表场所达到要求”“作者公开了代码”“代码可以正确复现”四件事。检索到论文或 GitHub 页面，不代表后两项已经满足。10-08 的调研未启动训练；10-09 按用户要求下载作者发布的权重并直接验证，不进行新训练，也不替换现有标注模型。

## 1. 决策：不能只跑原来的三个通用恢复模型

**建议正式比较包含六个核心恢复系统，另将 VCGAN 列为必须解决复现问题的第七个候选。** 六个系统是：划线专用 CycleGAN、COR-Generator、NAFNet、Restormer、DocDiff、DE-GAN。前三个已做过的模型继续保留，但 4,000 步结果只作先导试验。DocRes 作为补充，不优先扩展训练；缺少完整代码或任务不一致的近期论文列入相关工作和待核查清单。

此前清单遗漏了 ICDAR 2021 的专用划线恢复方法、IJDAR 2024 的 VCGAN，以及 2025/2026 年直接研究作答恢复、划线识别的工作。这一遗漏会使比较对象偏向通用恢复，应当纠正。

| 实验角色 | 确定的系统 | 为什么需要 | 当前条件 |
| --- | --- | --- | --- |
| 核心 B1 | Strikethrough-CycleGAN，ICDAR 2021 | 直接研究恢复被划掉的字；检验非配对训练路线 | 官方训练、测试、配置已检查；普通版本先进入同信息量比较，属性引导版本另列条件 |
| 核心 B2 | COR-Generator，按 IJDAR 2026 采用的恢复系统复现 | 直接相关的配对监督网络，避免只和不擅长划线的通用网络比较 | 原始算法与代码来自 DAS 2022；2026 正式期刊明确采用，必须双重归属，详见第 4 节 |
| 核心 B3 | NAFNet，ECCV 2022 | 简洁、强的通用卷积恢复对照 | 已完成短预算适配，需统一协议正式重跑 |
| 核心 B4 | Restormer，CVPR 2022 | 补足通用 Transformer 恢复路线 | 官方训练代码公开，尚未在本任务运行 |
| 核心 B5 | DocDiff，ACM MM 2023 | 文档专用残差扩散，检验细笔画恢复 | 已完成短预算适配；正式版须重新审查 EMA 和训练预算 |
| 核心 B6 | DE-GAN，TPAMI 2022 卷期 | 文档条件 GAN；与专用 CycleGAN 区分 | 已完成短预算适配，不能因先导结果差就删除 |
| 条件 B7 | VCGAN，IJDAR 2024 | 直接针对划线及文字结构的非配对方法 | 官方源码核心 Top-k 损失断开梯度；修复并核对论文后才允许训练和计入正式表 |
| 补充 B8 | DocRes，CVPR 2024 | 验证文档预训练和任务提示是否有用 | 原生五任务不含去划线；应明确命名任务适配版本，不直接把掩膜塞入并声称原论文复现 |

这里的“核心”是本项目的实验选择，不是论文水平排名。B2 的 CCF 合格依据是 **IJDAR 2026 正式论文中采用的系统**；原始 DAS 2022 论文没有因此变成 CCF C。若主表要求“每个算法首次发表场所本身都入 CCF”，将 B2 独立列为强相关补充对照，不隐去其出处。

## 2. 任务定义与纳入条件

输入为含划线的真实作答图或定位后的区域。目标是分离作废划线，保留原有文字结构，改善被划内容的读取。恢复出的内容仍保留“已作废”属性，不能重新混入最终有效答案。

纳入核心比较需要同时满足：

1. 正式论文可由出版社、会议论文集或作者机构记录确认，不能只有 arXiv 和作者声称录用。
2. 场所达到 CCF A/B/C；或中科院 1/2/3 区且分区年份与学科口径可核实。会议不套用期刊分区。
3. 输出可以用于我们的恢复任务；检测、整词删除、整页去阴影分别标明，不能只凭标题相似纳入。
4. 代码来源、关键损失、输入信息和训练适配可交代；没有作者代码时必须注明自行实现，不能伪装成官方复现。

完全遮黑且无法确认的原文字不设虚构像素真值。真实数据中的“可确认擦除区域”与“完整干净底稿”不同；后者主要由合成配对数据提供。图像看起来干净、划线像素 F1 高、文字读得对，也分别是不同结论。

## 3. 发表等级核验

CCF 已发布第七版目录，公告为 2026-03-31，页面注明 2026-04-09 勘误。本报告交叉查看了官方分类页和 CCF 2026 目录原文的高校镜像；官方分类网页有更新滞后迹象，不能只看网页标题就认为其内容已经同步新版。[CCF 正式公告](https://www.ccf.org.cn/Academic_Evaluation/By_category/2026-03-31/870181.shtml)、[CCF 2026 目录原文，高校图书馆镜像](https://lib.ecnu.edu.cn/_upload/article/files/e2/d9/dd45764345088869915f6b3617a4/28891f1f-eb8a-4000-a279-6e7141d92488.pdf)。

| 场所 | 本轮确认的 CCF 等级 | 对应候选 |
| --- | --- | --- |
| IEEE Transactions on Pattern Analysis and Machine Intelligence | A | DE-GAN |
| CVPR 主会正式长文 | A | Restormer、DocRes、MMDIR、MPRNet |
| ACM Multimedia 主会正式长文 | A | DocDiff |
| ECCV 主会正式长文 | B | NAFNet、Partial Convolutions |
| Pattern Recognition | B | Chaudhuri–Adak 2017 |
| ICDAR 正式长文 | C | Heil 等 2021 |
| International Journal on Document Analysis and Recognition，IJDAR | C | VCGAN；Pathirage 等 2026 |
| Signal Processing: Image Communication | C | Zhong 等 2025 |

相关官方分类入口：[人工智能](https://www.ccf.org.cn/Academic_Evaluation/AI/)、[计算机图形学与多媒体](https://www.ccf.org.cn/Academic_Evaluation/CGAndMT/)。CCF 等级属于发表场所，本表用当前核验口径，不暗示这些场所在论文发表当年等级一定相同。

**DAS、WACV、ICVGIP 不在本轮所查 CCF 2026 目录中。** 不能因为由 Springer/ACM 出版、被 CVF 收录，或与 ICDAR/CVPR 存在联系，就套用其他会议等级。CCF 公告也明确区分正式长文与 workshop 等形式。

中科院口径另外处理：中国科学院文献情报中心 2026-03-27 声明自 2026 年起不再更新、发布期刊分区表，其他机构的表与该中心无关。因此不把网上“2026 新锐分区”写成“2026 中科院分区”，也不用 JCR Q1 代替中科院 1 区。[官方声明](https://www.las.cas.cn/news/tzgg/202603/t20260327_8178738.html)

本报告的核心清单已有 CCF 路径满足要求，没有必要用未核实分区补齐。MathScrub 所在 TLT 的历史分区本轮未取得可核验的官方条目，保留“待核验”，不计入等级已确认清单。

## 4. 核心方法逐项审查

### B1. Strikethrough Removal from Handwritten Words Using CycleGANs

- 作者：Raphaela Heil、Ekta Vats、Anders Hast。
- 正式发表：ICDAR 2021，LNCS 12824，572–586；2021-09-02 online；CCF C。[出版社与 DOI](https://link.springer.com/chapter/10.1007/978-3-030-86337-1_38)
- 方法：在有划线词图与干净词图之间学习非配对转换，并研究属性引导与辅助识别损失。任务明确是恢复被划前的词，不是连词一起删掉。
- 官方资源：[作者代码](https://github.com/RaphaelaHeil/strikethrough-removal-cyclegans)、[代码与权重存档](https://doi.org/10.5281/zenodo.4767168)。仓库包含划线分类、识别有无划线、去划线三个组件，MIT 许可；训练测试配置实际存在。
- 数据：IAM 合成多书写者词图与真实单书写者 Dracula 配对数据；真实配对通过划线前后扫描及配准获得。它不是中文数学答题卡，迁移效果必须重测。[原论文数据附录](https://link.springer.com/chapter/10.1007/978-3-030-86337-1_38)

**适配决定：** 首先复现 `ORIGINAL` 非配对版本，使用同一训练池的有划线域和干净域，训练时不读取对应底稿作为逐像素监督。`ATTR_GUIDED` 是重要补充，但需满足额外输入条件。

**关键公平性问题：** 官方 `dataset.py` 的 CHANNEL 分支会把样本的真实划线类型编码为七通道 one-hot；辅助模型是七类 DenseNet121。我们的真实区域目前没有这套完整七分类标注，多笔混合区域也未必能一一对应。不能把测试真实类型直接送入模型，再把结果写成完全自动识别。可分别报告普通版本、预测类型版本及人工类型上限，后两者必须说明适配和附加监督；未准备好时不声称已经复现属性引导版。[固定版本数据读取](https://github.com/RaphaelaHeil/strikethrough-removal-cyclegans/blob/91555b22cac6b6a379597aa94c23bdf02c9970a7/strikethrough_removal/src/dataset.py#L126)、[辅助模型加载](https://github.com/RaphaelaHeil/strikethrough-removal-cyclegans/blob/91555b22cac6b6a379597aa94c23bdf02c9970a7/strikethrough_removal/src/utils.py#L172)。

### B2. COR-Generator：原始配对方法与 2026 正式研究

来源必须同时保留：

1. Heil、Vats、Hast，*Paired Image to Image Translation for Strikethrough Removal From Handwritten Words*，DAS 2022，309–322，2022-05-18 online，DOI `10.1007/978-3-031-06555-2_21`。该工作比较 SimpleCNN、Shallow、UNet 和 Generator；原始会议不按 CCF C 计。[正式出版记录](https://link.springer.com/chapter/10.1007/978-3-031-06555-2_21)、[作者论文全文](https://arxiv.org/html/2201.09633v2)、[DAS 官方报告](https://iapr.org/archives/das2022/wp-content/uploads/OS-slides/92.pdf)、[作者代码](https://github.com/RaphaelaHeil/paired_strikethrough_removal)。
2. Pathirage、Corbillé、Wåhlin、Islam、Barney Smith，*A study of handwritten text recognition with cross-out words*，IJDAR，2026-07-02 正式在线发表，CCF C。其第 3.3 节明确采用上述 Generator 和代码作为 COR 恢复系统。[正式论文全文与 DOI](https://link.springer.com/article/10.1007/s10032-026-00603-8)、[2026 作者项目](https://github.com/LTU-Machine-Learning/HTR_with_cross-out)。

Generator 使用卷积下采样、Dense bottleneck、上采样，由划线图预测干净词图。名字含 Generator，但该配对训练分支使用 BCE 重建，并不因此等于 GAN。原始训练配置为 30 epochs、batch 4、Adam、学习率 0.001，图像配置 128×512；这些是原始配方，不代表复制到本数据后已有同等效果。[原始配置](https://github.com/RaphaelaHeil/paired_strikethrough_removal/blob/19cdba071d82d3da1f69773123f4de86262e9519/config.cfg)、[训练实现](https://github.com/RaphaelaHeil/paired_strikethrough_removal/blob/19cdba071d82d3da1f69773123f4de86262e9519/src/train.py)。

**适配决定：** 先跑 `GENERATOR / DENSE / block_count=1`，不把论文四种小网络全部展开成主实验。保留真实比例、灰度极性和配对关系；需要改尺寸时记录为任务适配。

**2026 论文给我们的直接提醒：** 它比较了“训练识别器适应划线”“先清除划线再识别”及组合路线，结论依赖识别器和数据。我们不能只证明恢复图比原图适合一个未适配的 VLM，就推断恢复是唯一或最佳路线。中文和二维数学公式与其英文词图任务不同，不能直接复制其 CER 数字作我们的基线成绩。

### B3. NAFNet

- 论文：Liangyu Chen、Xiaojie Chu、Xiangyu Zhang、Jian Sun，*Simple Baselines for Image Restoration*。
- 正式发表：ECCV 2022，17–33；DOI `10.1007/978-3-031-20071-7_2`；CCF B。[出版社](https://link.springer.com/chapter/10.1007/978-3-031-20071-7_2)
- 方法：简化的编码解码恢复网络，以乘法门控等构件替代常用非线性激活；原始任务主要是自然图像去噪、去模糊，不是划线恢复。
- 官方代码、训练配置、权重均公开，MIT 许可。[官方仓库](https://github.com/megvii-research/NAFNet)

**适配决定：** 保留现有 width32、encoder `[1,1,1,28]` 版本作为固定配置。论文不同任务和宽度的结果不能混写；不拿 width32 的速度与 width64 的最佳精度拼成一个方法。官方 GoPro width32 配置为 200,000 iterations，而现有实验为 4,000；这证明预算不同，不表示本任务必须机械训练 200,000 步。[固定版本配方](https://github.com/megvii-research/NAFNet/blob/2b4af71ebe098a92a75910c233a3965a3e93ede4/options/train/GoPro/NAFNet-width32.yml)

### B4. Restormer

- 论文：Syed Waqas Zamir 等，*Restormer: Efficient Transformer for High-Resolution Image Restoration*。
- 正式发表：CVPR 2022，5728–5739；CCF A。[CVF 正式论文](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html)
- 方法：以适合高分辨率恢复的注意力和前馈模块建模图像信息；原论文覆盖去雨、去模糊、去噪，不能称为专门去划线的方法。
- 官方完整训练和测试代码公开，MIT 许可。[作者仓库](https://github.com/swz30/Restormer)

**适配决定：** 作为通用 Transformer 的核心比较对象，先固定一种公开单图恢复配置和原有目标函数，接入同一配对数据。不要把不同任务的配置混成一个“Restormer”。原始渐进裁块训练、大分辨率推理与固定 256 裁块适配必须分别记录。它和 NAFNet 覆盖不同架构，优先级高于再堆多个相似卷积网络。

### B5. DocDiff

- 论文：Zongyuan Yang 等，*DocDiff: Document Enhancement via Residual Diffusion Models*。
- 正式发表：ACM MM 2023，2795–2806；DOI `10.1145/3581783.3611730`；CCF A。正式 ACM 页面本轮抓取超时，发表信息由带 ACM 会议信息与 DOI 的作者论文以及官方仓库交叉确认。[论文](https://arxiv.org/abs/2305.03892)、[ACM DOI](https://doi.org/10.1145/3581783.3611730)、[官方仓库](https://github.com/Royalvice/DocDiff)
- 方法：粗预测器恢复主要内容，残差扩散模块细化高频；原论文研究文档去模糊、去噪、水印及印章去除。
- 训练、推理、默认配置、部分预训练权重公开。[官方说明](https://github.com/Royalvice/DocDiff/blob/main/README.EN.md)

**适配决定：** 保留文档残差扩散对照。现有先导训练没有 EMA；官方配置为 EMA 开启、100 扩散步、训练上限 1,000,000 iterations。正式比较应核对 EMA、频率分离损失、粗预测器与扩散模块的联合训练、采样方式和原生尺寸推理。允许制定适合本任务的预算，但必须标明偏离，不能把 4,000 步称为充分复现。[固定版本配置](https://github.com/Royalvice/DocDiff/blob/f422a3af10657eeabeca90f2e862d88c72725845/conf.yml)

### B6. DE-GAN

- 作者：Mohamed Ali Souibgui、Yousri Kessentini。
- 论文：*DE-GAN: A Conditional Generative Adversarial Network for Document Enhancement*。
- 正式发表：TPAMI 44(3)，1180–1191，2022-03 卷期；2020-09-07 首次在线；DOI `10.1109/TPAMI.2020.3022406`；CCF A。不能把预印本上传日期当首次发表日期。[IEEE 出版记录](https://ieeexplore.ieee.org/document/9187695)
- 方法：配对条件 GAN，从退化文档恢复干净图，覆盖清洁、二值化、去模糊及去水印；与 B1/B7 的非配对 CycleGAN 不同。
- 官方 Keras 实现及测试权重公开。[作者仓库](https://github.com/dali92002/DE-GAN)

**适配决定：** 保留官方生成器和判别器。现有适配沿用发布代码中的 MSE 对抗项与 100×BCE，并记录现代 TensorFlow/Keras 兼容修改。正式运行前核对论文公式与发布实现的差异、生成器/判别器交替更新及数据配对；不能只改成一个普通 U-Net 后继续叫 DE-GAN。[发布实现](https://github.com/dali92002/DE-GAN/tree/7281836455608337c940fa0de67cceffbc5c026d)

### B7. VCGAN：纳入候选，但存在明确的代码门槛

- 论文：Shivangi Nigam、Adarsh Prasad Behera、Shekhar Verma、P. Nagabhushan，*Deformity removal from handwritten text documents using variable cycle GAN*。
- 正式发表：IJDAR 2024，2024-05-07 online；DOI `10.1007/s10032-024-00466-x`；CCF C。仓库引用仍写 2022 预印本，正式参考文献应更新到 2024 期刊版。[出版社](https://link.springer.com/article/10.1007/s10032-024-00466-x)、[作者机构存档全文](https://dspace.networks.imdea.org/bitstream/handle/20.500.12761/1820/s10032-024-00466-x.pdf?isAllowed=y&sequence=1)
- 方法：非配对恢复，引入 Top-k Variable 内容损失，希望保护结构；覆盖划线、下划线等变形。我们必须保留正常下划线，因此其“去下划线”目标与本任务不完全相同，需要无涂改负样本检验。
- 作者代码：[shivii/VCGAN](https://github.com/shivii/VCGAN)，MIT；训练、生成器、判别器、配置均存在，尚未在我们的环境运行。

**本轮静态源码审查发现：** `train.py` 第 24 行对 top-k 差值重新构造 tensor，并调用 `clone().detach()`。这样得到的 Top-k 项没有通向生成器的梯度。当前第 98–99 行又注释了普通 cycle L1 项，配置中 identity 权重为 0，因而不能认为这份默认训练脚本实际施加了论文声称的 Top-k 内容约束。[固定版本问题位置](https://github.com/shivii/VCGAN/blob/56cb974cb9d0833f2846aa96f1f00bd16c58adb1/train.py#L19)、[损失组合](https://github.com/shivii/VCGAN/blob/56cb974cb9d0833f2846aa96f1f00bd16c58adb1/train.py#L95)、[配置](https://github.com/shivii/VCGAN/blob/56cb974cb9d0833f2846aa96f1f00bd16c58adb1/config.py)。

这是对公开版本的静态判断，不是复现跑分，也不代表作者发表时使用的代码一定如此。复现前需要核对论文 Top-k 的定义、聚合维度、k 的含义和 cycle/identity 权重；进行梯度与参数更新检查。若修复，应标记 `VCGAN-reimplementation / gradient fix`，公开最小差异。不能原样跑坏后以此宣称我们击败 VCGAN，也不能偷偷修好却写成官方代码原样复现。

## 5. 直接相关但暂不直接进入训练队列的论文

### Chaudhuri–Adak，Pattern Recognition 2017

完整标题为 *An approach for detecting and cleaning of struck-out handwritten text*，Pattern Recognition 61，282–294，DOI `10.1016/j.patcog.2016.07.032`，CCF B。它检测划线词，以骨架图上的约束路径识别划线，并用图像修补清洁词图，确实涉及保留原文字。[出版社](https://www.sciencedirect.com/science/article/pii/S003132031630190X)

本轮没有找到可确认的作者官方实现。搜索到的 [StrikOR](https://github.com/ankurguria/StrikOR) 采用 ResNet/U-Net 并将该文列为参考，不是该论文的官方骨架图实现。**必读并比较任务差异；获得完整细节后可做传统方法对照，不把第三方同主题项目冒充复现。**

### Zhong 等，Signal Processing: Image Communication 2025

*Struck-out handwritten word detection and restoration for automatic descriptive answer evaluation*，130，117214；2024 年在线、2025-01 卷期；DOI `10.1016/j.image.2024.117214`；CCF C。[作者机构发表记录](https://salford-repository.worktribe.com/output/3227736)、[出版 DOI](https://doi.org/10.1016/j.image.2024.117214)

它已经把划线词恢复与描述性作答评价联系起来：前段用 ResNet50 与对角线分割处理检测，后段用 U-Net 与 Bi-LSTM 预测原文字像素。因此“恢复文字用于答题评价”不能独立作为我们的首次创新主张。

本轮能确认正式发表、任务及方法概述，但未取得完整可检查的作者实现，全文细节也未完整获取。列为 **高优先级条件补充**：后续获取作者实现或充分的论文细节后，再决定忠实重实现。不能只凭摘要搭一个 U-Net+Bi-LSTM，冠以该论文名给出成绩。

### MathScrub，IEEE Transactions on Learning Technologies 2026

Sen Shen 等，*MathScrub: Single-Pipeline Strike-Out Removal and LaTeX Generation for Handwritten Calculus Expressions*，TLT 19，141–154，2026-02-24；DOI `10.1109/TLT.2026.3667787`。作者机构确认其正式发表，完整出版页本轮抓取未成功。[CUHK 发表记录](https://research.cuhk.edu.hk/en/publications/mathscrub-single-pipeline-strike-out-removal-and-latex-generation/)

其流程包括连通域聚合、删除检测、几何感知修补和 VLM LaTeX 生成，场景与我们非常接近。[Math-Strike 数据卡](https://huggingface.co/datasets/Incinciblecolonel/MathStrike) 定义的标签是 retained valid symbol 与 removed strike-out/noise；这提示它偏向有效内容清理，但**仅凭摘要和数据卡，不能断言它已经或没有解决“被划文字的底层内容恢复”**。需要全文确认输出及监督目标。

公开数据卡有真实考试与 20 类合成划线、区域坐标及页面来源，许可 CC-BY-NC 4.0；数据集发布不等于完整模型代码发布。本轮未确认完整训练实现，也未核验 TLT 的官方历史分区条目。因此列为 **必须讨论的新近相关工作，暂不计入等级与实现都已确认的主基线**，不把它包装成已就绪复现项目。

## 6. 通用文档恢复与修补的补充筛选

| 候选 | 正式发表核验 | 任务与实现判断 | 决定 |
| --- | --- | --- | --- |
| DocRes | CVPR 2024，CCF A | 五任务是展平、去阴影、外观增强、去模糊、二值化；官方训练、推理、权重可用；没有原生划线去除模式 | 优先级低于直接相关方法，作第八个补充。若加入自定义涂改掩膜提示，单独命名 adapted 版本 |
| MMDIR | CVPR 2026，8387–8396，CCF A | 正式论文确认；公开仓库当前只有一个 README，提供数据和预测结果链接，没有恢复模型训练/推理源码 | 纳入近期相关工作，等待可核验实现，不承诺马上复现 |
| MPRNet | CVPR 2021，CCF A | 作者提供去噪/去雨/去模糊训练测试代码，和已有通用恢复基线功能重叠 | 首轮暂不增加；主模型若强调多阶段设计，再考虑补充 |
| Partial Convolutions | ECCV 2018，CCF B | 掩膜条件修补；官方仓库可确认 partial-convolution 层，完整任务训练配方仍需核对 | 仅在论文明确研究遮挡处补全时增加；不能用文本整块掩膜先抹掉所有可见字再作不公平比较 |
| LaMa | WACV 2022 正式发表 | 是有影响力的通用修补方法，但 WACV 未在本轮 CCF 目录中，不能标成 CCF B/C | 当前严格等级清单外；可作技术参考，不为凑数量加入 |
| DocWaveDiff、2025 文档去污工作 | 检索到 WACV 2026/2025 会议原文 | 与文档恢复有关，但同样不满足已核验 CCF 场所路径 | 当前等级筛选下不进入主队列 |
| DocStormer | 本轮只确认到 arXiv 记录 | 未核验到正式发表场所；内容主要是彩色文档多退化恢复 | 暂不纳入已发表合格清单，不等同于断言它永远没有正式发表 |
| TexRGAN | 检索线索指向 ICVGIP 2021 | 任务相关，但正式全文、实现和符合要求的场所等级未完整核实 | 留在检索线索，不用作已确认合格基线 |

上述判断的直接出处：[DocRes 正式论文](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_DocRes_A_Generalist_Model_Toward_Unifying_Document_Image_Restoration_Tasks_CVPR_2024_paper.html)、[DocRes 作者代码](https://github.com/ZZZHANG-jx/DocRes)、[MMDIR 正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Li_MMDIR_Multimodal_Instruction-Driven_Framework_for_Mixed-Degradation_Document_Image_Restoration_CVPR_2026_paper.html)、[MMDIR 核查版本](https://github.com/xiaomore/MMDIR/tree/b1066ed4184b4c669b7105bd3620c42311e10602)、[MPRNet 作者代码](https://github.com/swz30/MPRNet)、[Partial Convolutions 正式论文](https://openaccess.thecvf.com/content_ECCV_2018/html/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.html)、[NVIDIA 实现说明](https://github.com/NVIDIA/partialconv/)、[LaMa 正式论文](https://openaccess.thecvf.com/content/WACV2022/html/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.html)、[DocStormer 作者预印本](https://arxiv.org/abs/2310.17910)、[DocWaveDiff 会议原文](https://openaccess.thecvf.com/content/WACV2026/papers/Marulli_DocWaveDiff_A_Predict-and-Refine_approach_for_Document_Image_Enhancement_with_Wavelet_WACV_2026_paper.pdf)、[2025 文档去污会议原文](https://openaccess.thecvf.com/content/WACV2025/papers/Li_High-Fidelity_Document_Stain_Removal_via_A_Large-Scale_Real-World_Dataset_and_WACV_2025_paper.pdf)。

## 7. 这些论文对数据制作的实际启示

“给干净手写图叠加划线，保留原图作监督”已经是直接相关工作的常见做法，不能作为本论文独有创新。2022 配对工作还比较了目标书写风格上的合成数据；2026 IJDAR 工作系统比较划线样本训练与恢复预处理。我们自己的真实划线素材、轨迹重绘和人工纠错，必须通过控制变量证明具体收益。

| 外部数据 | 能提供什么 | 使用限制与本项目决定 |
| --- | --- | --- |
| IAM Strikethrough | 七种合成划线、对应底稿，多书写者 | 可用于原方法管线自检或跨域测试；英文词图不替代中文数学数据；须遵守 IAM 与衍生数据许可和原始划分 |
| Dracula real | 同一书写者真实划线前后扫描及配准 | 有真正的前后图像，但书写者和规模有限；不把单书写者泛化结论扩展成学生总体 |
| Dracula synth | 在目标域干净词图上生成划线 | 与我们素材迁移实验相关，防止同一原词及其不同合成版本跨训练/验证 |
| Real-Strike-off | VCGAN 发布的真实评估数据 | 需逐项核查是否有配对底稿、转写和书写者信息；当前不假定具备 |
| 2026 LTU handwritten cross-outs | 多书写者、书写工具、划线类型、转写 | 论文有 12 位书写者，扫描样本大多不是同底稿多划线的配对；划分需检查是否书写者独立，不能照搬随机比例划分作泛化证明 |
| Math-Strike | 学生数学考试、真实及合成涂改、区域到整页映射 | 先确认“保留/删除”的标签语义，不能直接转换成保留底层文字的擦除真值 |

资源：[IAM 合成集](https://doi.org/10.5281/zenodo.4767094)、[Dracula real](https://doi.org/10.5281/zenodo.4765062)、[Dracula synth](https://doi.org/10.5281/zenodo.6406538)、[Real-Strike-off](https://github.com/shivii/Real-Strike-off-dataset)、[LTU 数据卡](https://huggingface.co/datasets/LTU-ML/handwritten_cross-outs)、[LTU 论文数据与划分说明](https://link.springer.com/article/10.1007/s10032-026-00603-8)、[Math-Strike 数据卡](https://huggingface.co/datasets/Incinciblecolonel/MathStrike)。本轮未下载这些图像数据，也没有将外部数据混入训练。

## 8. 正式实验协议：先冻结，再运行

### 8.1 四个必要对照与两条识别路线

这些是实验控制组，不占“已发表外部基线”的名额：

| 对照 | 用途 | 不能混淆的地方 |
| --- | --- | --- |
| 原图直接交给固定 VLM | 判断恢复是否真正改善读取 | 提示应读出可辨认的被划内容，不能再要求忽略作废文字 |
| 第一篇预测掩膜直接遮白 | 暴露粗掩膜删除文字的代价 | 它不是保护底层文字的恢复算法，属于破坏性参照 |
| 既有框内去划线模型 | 判断第二篇的新方法相对实际起点的增益 | 历史 57.4% 和当前微调模型来自不同训练历史，不能选高分拼接 |
| 人工擦除后的可确认图像 | 估计当前区域标注能带来的参考效果 | 不是完全遮挡位置真实底稿，也不保证是 OCR 理论上限 |

识别路线至少包括固定读者读原图、固定读者读恢复图。若论文要声称“视觉恢复优于直接适应划线”，还需加入相同文本监督预算下的直接识别适配组；这属于新增对照设计，不能把此前以“忽略作废内容”为目标训练的 LoRA 直接拿来代表恢复文字识别。2026 IJDAR 的三种英文识别器不要求全搬来中文数学任务，但它提出的路线比较应回应。

### 8.2 固定数据和监督信息

1. 保持按书写者划分，同一学生不同编号合并；原始宿主、划线素材、其变体均受划分约束。不能只排除目标底稿却允许测试书写者划线素材进入训练。
2. 第一个正式主比较使用同一个冻结的合成训练版本、真实纠错训练子集与负样本池。配对模型使用同一配对；非配对模型使用同源两域并遵循其原方法，明确监督差异。
3. 分开报告“仅合成”和“合成＋同一真实标注预算”。我们的模型若使用 50 处真实微调而对手没有，就不能归因于网络更优。
4. 素材方法消融先在我们的方法、NAFNet 和 COR-Generator 上做，不默认把所有模型乘以全部素材版本。控制素材数量、来源、宿主、叠加位置和增强；原金标与预测素材库大小不同的比较，不能只归因为清洁程度。
5. 使用已有清洁 PNG，固定灰度和残留处理。自动清理结果不能直接当作无争议人工真值；图像模型输出如加后处理，应对所有方法同样适用，并单列其贡献。
6. 掩膜来源与输入权限单独登记：image-only、预测 mask、人工 mask、划线类型属性。若我们用 mask，增加同骨干带 mask 的受控消融；不能把原版 image-only 模型输在少一个先验上，当成全部网络贡献。

### 8.3 局部恢复与整题系统分开评估

- **定位已知的恢复评测：** 所有方法使用相同人工区域，隔离定位误差。现有 81 处可读划线属于这一类。
- **整题自动评测：** 共享第一篇模型和已有行/区域切分，检测、恢复、回贴全部自动；报告漏掉的区域及错误处理的正常区域，不能只评检测成功的部分。
- 保持长宽比与原生细笔画；padding、缩放、重叠裁块和回贴规则冻结。必要的模型特定尺寸适配要披露。
- 整题输入、内部局部高分辨率恢复是合法系统设计，但局部人工框结果不能标为整题自动结果。

### 8.4 指标和选模型规则

主要读取指标沿用项目已经明确的区域口径：被划字符读回率、严格转写完全匹配、正常文字损伤，以及额外生成/误读字符。不能只看字符召回，让输出一长串候选字的模型占便宜；`whole` 允许额外字符，不等于严格完全正确。

像素指标成对报告划线去除召回与正常墨迹误擦，补充去除 precision/F1。可确认真实像素子集与有底稿的合成子集分开。PSNR/SSIM 仅在确有对齐底稿的数据上使用，并注意大片白背景的稀释；其他论文的 clean-foreground F1 与我们“擦掉的划线像素 F1”未必同一口径，不能直接排列比较。

模型选择建议用固定真实验证转写集上的读回表现，并预先确定文字误擦约束、同分决策、验证频率及停训规则；恢复-only 分析另报像素验证选版。若采用此建议，应在新实验 manifest 中明确改变了旧先导试验“16 区域像素 F1 选版”的规则。不得看测试集选素材、阈值、提示词或 checkpoint。

已有测试结果被多轮阅读，最终论文应考虑增加一份此前未用于开发决策的冻结审计子集，从合规保留书写者中选取；既有测试不改名、不悄悄改成验证。种子按当前项目规则仍为 42；书写者成组 bootstrap 可评估样本不确定性，但不能伪装成多随机种子训练稳定性。

### 8.5 训练预算与复现命名

不再用“所有模型 4,000 步”作为充分训练的依据。正式适配保留各自核心架构、损失、优化方式及 EMA 等关键机制，根据验证曲线设预算和停止条件，记录更新次数、有效见样本数、GPU 小时、参数量、推理耗时与峰值显存。

同时区分三种结果：

- **原配方验证**：在原任务或公开样例上检查实现能否表现出论文描述的行为；不宣称完整复现论文全部分数。
- **我们的任务适配**：使用冻结的共同数据和监督协议，偏离原配方处逐项记录。
- **固定计算预算比较**：资源受限下的效率实验，单独呈现，不能替代充分训练的主结论。

GPU0/3 是此前允许的设备范围，本调研没有实际调度 GPU，也没有给出未经测量的完成时间。后续执行仍需遵循现有锁卡、不可覆盖产物和同一 main 分支规则。

## 9. 现有三模型试验的定位与待改项

2026-10-07 运行了 NAFNet / DocDiff / DE-GAN × 三种素材，共九组，每组 4,000 步、batch 4、256×256；独立 16 人工区域按像素指标选 checkpoint，然后用固定 Qwen3-VL-8B 评估 81 处可读划线。它是可追溯的先导试验，已经揭示数据及实现问题，但不是原论文全预算复现。[项目结果记录](RESULTS.md)

| 项目 | 已有事实 | 正式比较前需要处理 |
| --- | --- | --- |
| 原图读回 | 48.88% | 原图必须随同一 OCR 版本、提示、裁图再次形成共同参照 |
| 本轮 NAFNet 最佳臂 | 原金标素材 54.54% | 不据此在测试上选择所有新模型的数据版本 |
| DocDiff | 末轮被选为最优；先导未使用 EMA | 核对官方机制并观察更充分的验证曲线 |
| DE-GAN | 本预算下误擦严重 | 保留失败记录，检查优化/损失/收敛，不据此排除 GAN 类方法 |
| 我们的历史 57.4% | 来自另一次训练和评测历史 | 需要同协议重评，不能直接与 54.54% 构成“已经胜出”的结论 |
| 真实微调 | 当前实际 50 训练＋16 验证区域 | 给对手相同可用真实监督，或将结果分层报告 |
| 评测范围 | 人工区域、81 可读划线、7 题像素标注 | 明确样本与书写者规模，不标为整题端到端或全遮挡恢复能力 |

现有源代码适配还应检查 DE-GAN 的现代 Keras 兼容与损失、NAFNetLocal 推理设置、DocDiff 采样与 EMA。已有数据的 darken-only 修复保留，不退回会使原本黑字变浅的合成版本。详见 [现有训练适配](revrender/baselines/restoration_torch.py)、[DE-GAN 适配](revrender/baselines/restoration_tf.py)、[数据制作](revrender/data/restoration_pairs.py)。

## 10. 后续执行顺序与完成条件

| 阶段 | 要完成的工作 | 通过条件 |
| --- | --- | --- |
| 0：冻结协议 | 确定真实标签可用范围、共同训练池、验证选版规则、输入权限和 OCR 提示 | 能列出每个方法看到了哪些图像、标签、掩膜与外部预训练数据 |
| 1：直接相关实现 | 接入普通 Strikethrough-CycleGAN 和 COR-Generator，检查 VCGAN 梯度问题 | 训练/推理入口闭合，损失可反传，保存重载一致，小批学习行为正常 |
| 2：统一正式比较 | 六个核心恢复系统＋我们的模型＋必要控制组；VCGAN 通过审查后加入 | 各方法验证选版和预算可追溯，不按测试优劣删除对手 |
| 3：数据与信息消融 | 配对素材复制/预测素材/轨迹重绘、真实微调、mask 先验 | 每个实验只改变预先声明的因素；同监督预算 |
| 4：整题与跨读者验证 | 共享自动定位流程回贴；换一个固定读者验证结论是否依赖单一 VLM | 含漏检和负样本损伤，有整题结果；不只展示挑选出的成功案例 |
| 5：必要补充 | 按证据决定 DocRes、2017 传统法、2025 U-Net+Bi-LSTM 是否补入 | 完整代码/方法与公平比较条件就绪，再增加工作量 |

目前不建议同时扩展所有候选，也不需要重新开发一个教育评分或动机分析模型。首先完成直接相关方法的公平比较，才能判断我们真正的技术空间。

## 11. 本次核查的可追溯记录与限制

本轮通过 Exa 执行 24 次检索，请求并返回 140 条结果（含重复），按原始 URL 去重为 112 个候选来源；URL 大小写/镜像差异未合并，这不代表 112 篇论文。进一步阅读正式论文页、可访问全文、作者仓库与关键配置。检索方向包括直接划线恢复、文档增强、通用恢复/修补、近期研究、等级、复现资源。没有声称已穷尽所有文献或把每篇付费全文读完。

关键源码版本如下。新仓库通过公开文件树及少量源码静态核查，未安装依赖或执行其训练；现有三模型版本来自此前保存的 source manifest。

| 仓库 | 核查版本 | 核查层次 |
| --- | --- | --- |
| RaphaelaHeil/strikethrough-removal-cyclegans | `91555b22cac6b6a379597aa94c23bdf02c9970a7` | 文件树、配置、训练、数据读取、辅助模型加载 |
| RaphaelaHeil/paired_strikethrough_removal | `19cdba071d82d3da1f69773123f4de86262e9519` | 文件树、配置、训练损失 |
| shivii/VCGAN | `56cb974cb9d0833f2846aa96f1f00bd16c58adb1` | 文件树、配置、训练损失与梯度断开问题 |
| swz30/Restormer | `68dc6ac472db26f16361150cb7a96a1bc87da93f` | 作者说明与训练/测试/架构文件树；未运行 |
| ZZZHANG-jx/DocRes | `d3e3a18c7c7ad10e4615ca2eb6a10f83128d8560` | 作者说明与训练/测试/架构文件树；未运行 |
| xiaomore/MMDIR | `b1066ed4184b4c669b7105bd3620c42311e10602` | 文件树仅一个 README；无训练源码 |
| Royalvice/DocDiff | `f422a3af10657eeabeca90f2e862d88c72725845` | 已有适配及本轮官方配置复查 |
| megvii-research/NAFNet | `2b4af71ebe098a92a75910c233a3965a3e93ede4` | 已有适配及本轮官方配置复查 |
| dali92002/DE-GAN | `7281836455608337c940fa0de67cceffbc5c026d` | 已有适配及本轮发表、源码复查 |

检索及代码摘录保存在本机忽略目录 `work/baseline_research20261008/evidence/`，不把第三方代码、模型或数据提交进本仓库。代码、分区或论文可用性随时间变化，第 1–11 节反映 2026-10-08 的核查状态。“未找到作者实现”是本次检索边界，不是断言不存在。

## 12. 2026-10-09 更新：近期论文与官方权重直接验证

### 12.1 有 2025、2026 年的工作，不能只列旧模型

本次检索时间范围为 2025-01-01 至 2026-10-09，分别查找直接划线恢复、文档去污、文字修补，以及这些论文的作者代码和权重。发表依据优先采用出版社、会议论文集、作者机构记录；代码和权重另查作者仓库。以下“未确认权重”只表示本次没有找到可核验的下载入口。

| 方法 | 已确认发表 | 实际任务与我们的关系 | 代码／权重现状与安排 |
| --- | --- | --- | --- |
| Zhong 等：Struck-out handwritten word detection and restoration for automatic descriptive answer evaluation | Signal Processing: Image Communication，2025 年 1 月卷期，CCF C；2024 年在线 | **直接相关**：检测划线词，用 U-Net 与 Bi-LSTM 恢复文字，用于作答评价 | 机构与 DOI 确认发表；未确认作者完整恢复实现及权重。优先补齐全文与实现，不用摘要自行拼网络冒充复现 |
| MathScrub | IEEE TLT，2026-02-24，19:141–154；等级仍待官方口径核验 | **场景直接相关**：数学考试、删除检测、几何修补、VLM 转 LaTeX；是否恢复被作废的底层内容仍需全文核对 | MathStrike 数据公开；未确认完整模型代码与权重。数据卡的标签区分保留有效符号和删除划线／噪声，不能据此当作我们的像素真值 |
| A study of handwritten text recognition with cross-out words | IJDAR，2026-07-02，CCF C | **直接相关的实验路线**：比较识别器适应划线、先恢复再识别及组合 | 公开项目；其中 COR 恢复器仍来自 DAS 2022 的 Generator，不能算成一个全新的 2026 恢复架构；未确认新发布的 COR 权重 |
| Uni-DocDiff | **ACM MM 2025，CCF A**；DOI 10.1145/3746027.3755362 | 新的统一文档扩散模型，任务提示和先验池处理多种退化。其 handwriting removal 明确是**删除手写、保留印刷字**，与我们保留手写底层文字的目标不同 | 已读作者全文并确认 ACM 出版信息；未确认公开可运行实现或权重。作为新的文档恢复候选，不能把原生去手写模式直接当去划线模式 |
| MMDIR | **CVPR 2026 主会，CCF A**，8387–8396 | 指令驱动的混合退化恢复，涵盖模糊、阴影、水印、印章 | CVF 正式论文可核验；作者仓库仍以 README、数据和预测结果为主，没有可验证的模型训练／推理代码及权重。预测图片不是模型权重 |
| MMTINet：Fusing Text Semantics for Text Image Inpainting | **ICDAR 2025，CCF C**，385–397，2025-09-16 | 从图像自身提取文字语义，恢复缺损／遮挡文字；原场景偏自然场景文字，不是手写数学划线 | 出版社摘要和发表信息已核验；完整内容受订阅限制，未确认作者代码或权重。比单纯去噪更接近“文字补全”，但复现条件未就绪 |
| Bridging Knowledge Gap Between Image Inpainting and Large-Area Visible Watermark Removal | **AAAI 2025，CCF A**，2025-04-11 | 用残留背景分支与门控融合，将 LaMa 修补先验用于去水印；处理不精确掩膜的思路值得参考，但原任务是自然图像水印 | 已读论文与作者页；作者声明源码在补充材料，本轮未取得可核验的完整补充包和权重。不能写成“无代码”，也不能说已就绪 |
| StainRestorer | WACV 2025，正式发表；本轮 CCF 目录未列 WACV | 文档污渍／印章去除，是较新的文档恢复对照 | 作者仓库有训练、推理和 **Weights release**。技术上值得试，但不擅自把 WACV 标为 CCF B/C；在用户原等级约束下列补充对照 |
| DocWaveDiff | WACV 2026，正式发表；同上不套 CCF 主会等级 | 小波预测与扩散细化，涵盖去模糊和手写删除 | 官方代码存在；README 未确认独立可下载的预训练权重。训练输出目录 `checksave` 不等于已发布权重 |

逐项主来源：

- Zhong 2025：[作者机构记录](https://salford-repository.worktribe.com/output/3227736)、[出版 DOI](https://doi.org/10.1016/j.image.2024.117214)。
- MathScrub：[作者机构正式记录](https://research.cuhk.edu.hk/en/publications/mathscrub-single-pipeline-strike-out-removal-and-latex-generation/)、[MathStrike 标签定义](https://huggingface.co/datasets/Incinciblecolonel/MathStrike/blob/main/README.md)。
- IJDAR 2026：[出版社全文](https://link.springer.com/article/10.1007/s10032-026-00603-8)、[作者项目](https://github.com/LTU-Machine-Learning/HTR_with_cross-out)。
- Uni-DocDiff：[含 ACM 出版信息的作者全文](https://arxiv.org/html/2508.04055v1)、[正式出版 DOI](https://doi.org/10.1145/3746027.3755362)。作者全文明确区分“删除手写、保留印刷字”，不是本文自行推断任务目标。
- MMDIR：[CVPR 2026 正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Li_MMDIR_Multimodal_Instruction-Driven_Framework_for_Mixed-Degradation_Document_Image_Restoration_CVPR_2026_paper.html)、[作者仓库](https://github.com/xiaomore/MMDIR)。
- MMTINet：[Springer 正式论文与摘要](https://link.springer.com/chapter/10.1007/978-3-032-04630-7_22)。
- AAAI 去水印：[会议论文页](https://ojs.aaai.org/index.php/AAAI/article/view/32484)、[作者全文](https://arxiv.org/html/2504.04687v1)、[作者项目页](https://chaoweifang.github.io/publication/aaai2025bridging_knowledge_gap/)。
- StainRestorer：[CVF 正式论文](https://openaccess.thecvf.com/content/WACV2025/papers/Li_High-Fidelity_Document_Stain_Removal_via_A_Large-Scale_Real-World_Dataset_and_WACV_2025_paper.pdf)、[官方代码](https://github.com/CXH-Research/StainRestorer)、[官方权重 release](https://github.com/CXH-Research/StainRestorer/releases/tag/Weights)。
- DocWaveDiff：[会议全文](https://openaccess.thecvf.com/content/WACV2026/papers/Marulli_DocWaveDiff_A_Predict-and-Refine_approach_for_Document_Image_Enhancement_with_Wavelet_WACV_2026_paper.pdf)、[官方代码](https://github.com/miccunifi/DocWaveDiff)。

另外检索到 DocRevive（CVPR 2026 **Workshop**）和 DocPure（2026 年 arXiv）。前者不能按 CVPR 主会 A 类计，后者本轮只确认预印本；因此没有为了年份新就塞进合格主基线。DocStormer 本轮仍只确认到 2023 年预印本，不能当作 2025/2026 新论文。

**后续清单建议：** 已能运行的六个系统保留；近期直接相关方法优先解决 Zhong 2025、MathScrub 的任务定义与实现可用性，文字修补补充关注 MMTINet。进一步核查后，建议新增 StainRestorer 作为 2025 年的补充实验，不将它冒充满足原 CCF 等级筛选的核心基线。Uni-DocDiff、MMDIR 先列相关工作和待复现，不因年份新就强行加入数值比较。四篇重点候选的资源证据及顺序见第 13 节。

### 12.2 旧方法确实发布了权重，本轮直接加载

| 系列 | 本轮下载并纳入的权重 | 官方出处 | 说明 |
| --- | --- | --- | --- |
| Strikethrough-CycleGAN | ORIGINAL 5 个、ORIGINAL_WITH_AUX_DISCR 5 个 | [v1.0 release](https://github.com/RaphaelaHeil/strikethrough-removal-cyclegans/releases/tag/v1.0) | 作者不同训练运行，共 10 个候选；推理不用人工类型。需要真实类型通道的 ATTR_GUIDED 两组排除 |
| DocDiff | deblur、seal 各一对粗预测器＋扩散器 | [官方 checksave](https://github.com/Royalvice/DocDiff/tree/main/checksave) | **更正：仓库不只有去模糊权重，也有去印章权重。** 两套都直接验证 |
| NAFNet | GoPro width32、SIDD width32 | [官方权重表](https://github.com/megvii-research/NAFNet) | 分别在自然图像去模糊／去噪上训练，保留对应网络配置 |
| Restormer | motion deblurring | [官方去模糊说明](https://github.com/swz30/Restormer/blob/main/Motion_Deblurring/README.md) | 真实去噪权重的官方 Google Drive 返回下载配额超限，未混用未经核验的镜像；本轮用成功下载的去模糊版本 |
| DE-GAN | watermark、deblur、binarization | [官方仓库及权重链接](https://github.com/dali92002/DE-GAN) | 对应 512／1024／1024 最大层宽，二值化版本保留作者 0.95 阈值 |
| COR-Generator | 未纳入预训练权重测试 | [作者仓库](https://github.com/RaphaelaHeil/paired_strikethrough_removal) | 有训练代码，但本次仍未确认官方已训练 Generator 下载入口；此前我们自行训练的权重不能冒充作者权重 |

合计 **5 个系列、18 个官方权重版本**。这是跨域直接推理，模型训练时使用作者各自的数据，**不是**重新用本项目的 10,788 对合成图训练；不能与前一轮同数据重训结果混为一张公平训练表。

本轮协议：转写验证使用既有验证划分中已完成的 11 题、41 处可读划线；另有 1 处可读涂黑仅记录，不参加划线选版。像素验证使用既有 16 处独立人工区域。使用清洁 PNG、相同裁图与固定 Qwen3-VL-8B 提示和解码；无优化、无测试集选版、无标注页自动残留清理。CycleGAN 按作者 128×512 输入处理后逆映射，其他方法使用原生分辨率 256 裁块；这些尺寸适配会影响跨域效果，已记录。

选版规则在推理前写入 manifest：先按 41 处被划字符读回率，再依次以严格全对、较低文字误擦和名称处理同分。同时报告划线去除、文字损伤及完整转写，避免把“擦得多”当作“恢复得好”。18 个候选在小验证集上择优有乐观偏差，所得仅是后续候选，不能称为最终泛化成绩或自动替换人工微调标注模型。

检索证据在本机 `work/pretrained_validation20261009/evidence/`；下载权重在本机 `work/official_weights20261009/` 与服务器 `outputs/paper2_official_weights20261009/`，权重哈希、具体来源和模型配置写入运行记录，不进入 Git。

## 13. 2026-10-09 追加核查：四篇近期候选是否具备复现条件

**建议：现有六个基线保留，新增 StainRestorer 补充实验；Zhong 2025 是最应争取补入的直接相关方法；Uni-DocDiff 与 MMDIR 暂列相关工作。** 以下区分论文发表、实际源码、模型权重与数据／预测图片。“未找到”仅表示这次公开检索和可访问资源的边界。

| 方法 | 可核验代码 | 可核验权重 | 是否加入实验 |
| --- | --- | --- | --- |
| StainRestorer，WACV 2025 | 官方训练、模型、数据读取、推理源码齐全；推理入口有待修正之处 | 官方 Release 有污渍、水印、印章三套 `.pth`；三条下载地址均实际返回文件字节 | **建议加入补充基线，优先验证**；WACV 不计入本轮 CCF 核心清单 |
| Zhong 等，SPIC 2025，CCF C | 机构记录和摘要可读；未找到可核验的作者完整恢复实现 | 未找到可核验下载入口 | **直接相关，优先争取加入**；先补齐全文与作者资源，不能把摘要拼装称为复现 |
| Uni-DocDiff，ACM MM 2025，CCF A | 可访问作者全文未给出方法仓库；方法名 GitHub 检索未找到实现 | 未找到可核验下载入口 | **暂缓实验**；先列相关工作，且其原生去手写任务目标与我们不同 |
| MMDIR，CVPR 2026，CCF A | 官方仓库当前只有 README；唯一分支 main，无 Release、Tag | 仓库未发布模型文件；README 外链标为数据集和预测结果 | **暂缓实验**；模型源码／权重发布后再考虑 |

### 13.1 StainRestorer：资源真实存在，但需要小幅推理适配

核查版本：[`CXH-Research/StainRestorer@f25239eb0ea817ce04bdfc2a3209157aa2e7998a`](https://github.com/CXH-Research/StainRestorer/tree/f25239eb0ea817ce04bdfc2a3209157aa2e7998a)。[Weights Release](https://github.com/CXH-Research/StainRestorer/releases/tag/Weights) 发布于 2024-11-28，早于 WACV 2025 卷期；以下是实际发布资产，不是“即将公开”的承诺。

| 官方文件 | 字节数 | 对应任务 |
| --- | ---: | --- |
| [StainDoc.pth](https://github.com/CXH-Research/StainRestorer/releases/download/Weights/StainDoc.pth) | 27,676,762 | 文档污渍去除 |
| [StainDoc_mark.pth](https://github.com/CXH-Research/StainRestorer/releases/download/Weights/StainDoc_mark.pth) | 27,679,426 | 文档水印去除 |
| [StainDoc_seal.pth](https://github.com/CXH-Research/StainRestorer/releases/download/Weights/StainDoc_seal.pth) | 27,679,426 | 文档印章去除 |

三条链接都通过 HTTP Range 取得前 64 字节，返回 206、总大小与 Release 一致，并具有 ZIP 文件头。这确认文件可下载，**尚不等于已完整下载、成功加载或在我们数据上验证**。

静态源码检查发现两个适配点：

1. [模型第 745 行](https://github.com/CXH-Research/StainRestorer/blob/f25239eb0ea817ce04bdfc2a3209157aa2e7998a/models/StainRestorer.py#L745) 返回 `[最终恢复图, 中间图]`；[训练验证第 96 行](https://github.com/CXH-Research/StainRestorer/blob/f25239eb0ea817ce04bdfc2a3209157aa2e7998a/train.py#L96) 正确取 `model(inp)[0]`，但 [infer.py 第 57–66 行](https://github.com/CXH-Research/StainRestorer/blob/f25239eb0ea817ce04bdfc2a3209157aa2e7998a/infer.py#L57) 把整个列表传给图像指标。后续入口应沿用作者验证代码取最终图；这是输出接口修正，不是改网络。本轮仅静态检查，未执行该脚本。
2. [仓库配置](https://github.com/CXH-Research/StainRestorer/blob/f25239eb0ea817ce04bdfc2a3209157aa2e7998a/config.yml#L18) 默认 256×256，而[论文](https://openaccess.thecvf.com/content/WACV2025/papers/Li_High-Fidelity_Document_Stain_Removal_via_A_Large-Scale_Real-World_Dataset_and_WACV_2025_paper.pdf) 描述训练 256、测试 1024。直接默认运行不能声称复现论文原测试协议；我们的长行适配应保持笔画长宽比，记录裁块／缩放规则。

论文同时验证污渍、水印、印章去除；后两类合成数据利用干净文档背景生成。其“去除覆盖层、保留文字”与我们有联系，但有色污渍／印章不等于黑色手写划线与文字交叉。因此值得作为新的文档恢复对照，不能预先认定效果优于专门去划线模型。

建议下一次执行先将三套权重在同一冻结验证集上直接推理，按既定读回规则选版并同时报告文字误擦；不按测试集挑任务权重。若随后进入正式同数据比较，再用共同训练对和相同真实标注预算训练／微调。跨域官方权重成绩与同数据重训成绩分表。

### 13.2 Zhong 2025：任务最接近，尚未拿到作者恢复实现

[作者机构记录](https://salford-repository.worktribe.com/output/3227736)与[出版 DOI](https://doi.org/10.1016/j.image.2024.117214)确认题名、四位作者、2025 年 1 月卷期及 2024 年在线发表。摘要描述先检测划线词，再用 U-Net 与 Bi-LSTM 恢复实际文字像素，并服务于描述题评价，直接对应我们的研究问题。

核查了题名、作者和方法相关代码检索；未找到可确认属于本文作者的完整恢复代码或权重。出版全文／补充附件本轮访问未成功，因此不能断言作者完全没有发布资源。[ScienceDirect 正确论文页](https://www.sciencedirect.com/science/article/pii/S0923596524001152)对应 PII `S0923596524001152`。

避免两个误认：搜索中的 `ankurguria/StrikOR` 不是这篇论文的作者实现；[HWG-SOW 2026 资源](https://zenodo.org/records/21560739)和[相关仓库](https://github.com/said702/hwg-strikeout-detection)提供的是划线检测／分类研究及标签，不是本论文的文字恢复器。后者明确说 SOW 原图需向原作者获取，不能把标签 CSV 当恢复模型。

建议将此文列为必讨论的直接相关工作和优先待复现候选。取得全文／作者资源后再决定使用官方实现或明确标为自行复现；本轮未向作者发送消息。

### 13.3 Uni-DocDiff：尚未确认公开实现，去手写模式不能直接套用

[可访问作者全文 v1](https://arxiv.org/html/2508.04055v1)与[正式出版 DOI](https://doi.org/10.1145/3746027.3755362)确认 ACM MM 2025。检查作者全文／摘要外链、Crossref 元数据、GitHub 的 `Uni-DocDiff`／`UniDocDiff` 仓库检索与 Hugging Face 模型检索，没有取得可核验的方法代码或模型文件。ACM 附件访问未完成，不能把这次未找到写成永久无代码。

全文明确：handwriting removal 是删除手写、保留印刷内容；我们需要保留被划掉的手写字。**原生去手写模式的任务目标相反。** 即使未来拿到权重，也应先确认具体任务头／提示及适配办法，不能把删除所有手写的结果直接当去划线能力。[SyntheticDoc 作者论文](https://igl.ethz.ch/projects/SyntheticDoc/syntheticdoc-eccv-2026-woortmann-et-al.pdf)也记录了其研究当时 Uni-DocDiff 未发布代码／结果的复现限制，这只作旁证，不替代当前仓库核查。

建议在相关工作中讨论统一文档扩散恢复；暂不增加为可运行实验。检索返回的 2023 DocDiff 仓库不是 Uni-DocDiff，不能混用名称或权重。

### 13.4 MMDIR：有正式论文，但公开项目仍不具备模型推理条件

[CVPR 2026 正式论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Li_MMDIR_Multimodal_Instruction-Driven_Framework_for_Mixed-Degradation_Document_Image_Restoration_CVPR_2026_paper.html)与[作者仓库固定版本](https://github.com/xiaomore/MMDIR/tree/b1066ed4184b4c669b7105bd3620c42311e10602)可核验。通过公开 GitHub API 检查文件树、所有公开分支、Release 和 Tag：只有 main 分支的一个 README 文件，Release／Tag 均为空。

README 两个网盘链接分别标为 **MixedDoc 数据集**与**模型预测结果**；预测图片不能供我们输入新答题卡推理。本轮未检查网盘内部全部文件，不对其中是否另藏未注明的附件作断言；公开页面目前没有可核验的源码或 checkpoint 入口。CVF 论文页也未列出补充包链接。

它针对模糊、阴影、水印、印章及混合退化，适合讨论多退化文档恢复，但尚不能据此加入实际成绩表。等作者发布训练／推理模型与权重后再核验，不依据摘要重搭一个网络冒充官方复现。

### 13.5 本轮检查范围

围绕上述四篇额外执行 7 次 Exa 检索，请求 46 条结果（含重复），并检查作者论文、机构页、GitHub API、关键源码和发布附件。46 是请求的检索结果数量，不是已完整阅读的论文数量。主要限制是 Zhong 的出版全文／附件与 Uni-DocDiff 的 ACM 附件未成功取得；对这两者使用“未找到可核验资源”的口径。

公开证据、下载探测与源码摘录保存在本机忽略目录 `work/baseline_research20261009/followup/`。本轮没有训练、推理或新增实验成绩；第 12 节的 18 份权重验证属于此前已经完成的实验，未因这次文献核查改写。

## 14. 2026-10-09 数据集专项核查：原论文究竟训练、测试了什么

### 14.1 结论和证据范围

**“已有数据都是合成的，所以我们提出第一个真实数据集”不成立。** 2021 年已公开真实笔划前后配对数据；之后已有学生笔记、学生答卷、多书写者划线及数学考试涂改资源。真正需要研究的是：这些数据能否支持我们要做的“自然发生的中文数学作答涂改中，保真恢复被划掉的内容”，以及缺少什么监督与评测。

同样，**不能由通用模型在我们答题卡上的迁移效果较差，推出其原论文设计不好**。部分模型原本解决自然图像噪声、模糊、彩色水印或污渍，训练目标并非分离同色、交叉的手写字与划线。需要先公平适配，再报告具体失效模式。

本轮围绕原训练集、原测试集、真实／合成定义、干净底稿、数据粒度和近期学生作答数据，执行 17 次 Exa 检索、请求 85 条候选结果（含重复，其中一次查询报错）；这不是读完 85 篇论文。已下载并核对 **12 篇可获取全文的数据与实验章节**，另读取作者仓库、数据 README 和机构记录。下载及文本保存在 `work/baseline_dataset_audit20261009/`，清单 `download_manifest.json` 记录来源、页数和 SHA-256；PDF 和原始摘录不进入 Git。

已下载全文：COR 2022、DE-GAN、NAFNet、Restormer、DocDiff、DocRes、VCGAN、StainRestorer、Uni-DocDiff、MMDIR、2026 IJDAR 划线识别研究、HWG 2026。以下页码指本轮下载 PDF 的物理页码。

**仍未取得原论文全文：** CycleGAN 2021、Zhong 2025、MathScrub 2026、Chaudhuri–Adak 2017。其结论分别限定为出版社摘要、作者代码／数据、机构记录或明确注明的后续论文证据；没有把这些当成已全文核验。没有下载大规模数据集，没有启动新的实验。

### 14.2 先区分三种“真实”

| 数据来源 | 例子 | 有什么价值 | 不能等同于什么 |
| --- | --- | --- | --- |
| 真实手写底稿，加数字合成划线 | IAM-synth；我们目前主要合成训练对 | 原字确定，能提供严格配对监督 | 划线行为本身不是真实发生的 |
| 人按要求写字、划线，或在纸上施加污渍 | Dracula；LTU 多书写者；StainDoc 污渍 | 覆盖真实笔触／成像；部分能获得处理前后配对 | 不必然代表学生做错后自然修改的分布 |
| 自然作答中产生的修改 | 学生笔记、SOW、Math-Strike、BRESSAY 的删改 | 更接近实际应用 | 通常没有修改前的同位置干净扫描图 |

“有原文转写”“有划线区域”“有可擦除像素”“有同一图像修改前的干净底稿”也是四种不同的标注，不能互相替代。

### 14.3 已运行或已纳入的七个基线：原论文数据

下表是**作者原实验**，不是我们适配实验的数据。不同任务的子实验分别训练，不能把全部数据相加后说每套权重都见过这些数据。

| 方法 | 原训练数据 | 原测试数据／规模 | 数据性质及对我们意味着什么 |
| --- | --- | --- | --- |
| **Strikethrough-CycleGAN，2021** | 出版摘要明确：模型只用合成数据训练；公开 IAM 划线数据含训练 3,066、验证 273、测试 819 个划线词 | IAM 合成测试；真实单书写者 Dracula 也用于评估。Dracula 官方测试分割为 378 个划线词 | **已经测试真实划线**。Dracula 是蓝色圆珠笔、受控七种划线、有配准后的干净底稿；不是自然数学答题卡。依据：[出版社摘要](https://link.springer.com/chapter/10.1007/978-3-030-86337-1_38)、[作者项目](https://github.com/RaphaelaHeil/strikethrough-removal-cyclegans)、[真实数据 README](https://zenodo.org/records/4765063/files/README.md?download=1)；IAM 分割另由 COR 表 2 核对，2021 全文未取得 |
| **COR-Generator，DAS 2022** | 分别用 IAM-synth 3,066；或 Dracula 干净训练词 126 个生成划线，单个随机种子 126、五种种子合并 630；配对重建训练 | IAM-synth 819；Dracula-real 378，真实验证 126 | **用合成训练、真实划线测试的路线已经存在**。不能把表中 Dracula-real 的 train 126 误写成模型直接用 126 个真实划线训练，正文 §4.2 指定的是合成训练。来源：[全文](https://arxiv.org/abs/2201.09633)，PDF pp.6–7，表 2、§4.1–4.3、§5.1 |
| **DE-GAN，TPAMI** | Noisy Office：112 页、切成 1,356 块；二值化：其他年份 DIBCO 共 80 页、6,824 块；自制数字水印：800 页、7,658 块；去模糊：4,000 块 | Noisy Office 32 页；DIBCO 2013、2017、H-DIBCO 2018；数字水印 200 页；去模糊 932 块 | 文档去噪、二值化、水印与模糊，**不等于真实手写划线恢复基准**。真实劣化文档与人工合成任务并存。来源：[全文](https://arxiv.org/abs/2010.08764)，PDF pp.5–10，§4.1–4.5 |
| **DocDiff，ACM MM 2023** | 去模糊：Document Deblurring 中 30,000 对；二值化：其他年份 DIBCO 加 Noisy Office、Bickley Diary、PHIDB、S-MS；水印：3,000 个 512² 训练块；印章：3,000 个 512² 样本 | 去模糊 10,000 对；H-DIBCO 2018、DIBCO 2019；水印 100 张整图；印章 500 个样本；另展示真实中文票据 | 水印、印章测试主要为作者合成，背景和覆盖素材的训练／测试不重叠；真实票据展示不能当作真实学生划线大规模定量实验。来源：[全文](https://arxiv.org/abs/2305.03892)，PDF pp.5–6、10–11，§4.1、附录 B/C |
| **NAFNet，ECCV 2022** | 主要为 SIDD 真实相机噪声、GoPro 去模糊；附加 RAW 去噪沿用 PMRID 方案 | SIDD、GoPro；还测试 4Scenes RAW、REDS-val 去模糊加 JPEG | 自然图像恢复；我们下载的 SIDD32、GoPro32 权重没有因此接受划线监督。来源：[全文](https://arxiv.org/abs/2204.04676)，PDF §5 及扩展实验 |
| **Restormer，CVPR 2022** | 按任务使用 GoPro、SIDD、DPDD、去雨／高斯去噪数据 | GoPro 训练还在 HIDE、RealBlur-R/J 测试；SIDD 训练测试 SIDD 与 DND；另有 DPDD、去雨和标准去噪测试集 | 自然图像恢复；**原论文已经做跨数据集测试**。我们已下载的 motion 权重对应 GoPro，不是手写文档预训练。来源：[正式全文](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf)，§4，表 2、6 |
| **StainRestorer，WACV 2025** | StainDoc：4,502 对；另分别训练数字合成的 StainDocMark、StainDocSeal | 各任务 558 对；主 StainDoc 共 5,060 对 | **主数据不是全数字合成**：在纸上真实施加红墨、蓝墨、绿茶、红茶等污渍，再拍摄配对；Mark／Seal 才是数字覆盖。仍与同色手写划线不同。来源：[正式全文](https://openaccess.thecvf.com/content/WACV2025/papers/Li_High-Fidelity_Document_Stain_Removal_via_A_Large-Scale_Real-World_Dataset_and_WACV_2025_paper.pdf)，PDF p.3 §3、p.6 §5.1 |

**Dracula 数量的澄清：** 官方所说 756 个词，不是 756 个划线正例。README 列出真实划线 train/val/test 为 **126/126/378，共 630**；此外 train 有 **126 个不与划线词重叠的干净词**，供非配对训练使用。因此 630 与 756 并不冲突。[官方 README](https://zenodo.org/records/4765063/files/README.md?download=1)

**指标也不能混用：** COR 的 F1 是对输出干净字图做 Otsu 二值化后，与干净底稿比较墨迹前景的 F1；它不是第一篇的涂改区域掩膜 F1，也不是我们的“被划字符读回率”。跨论文不能直接排大小。

### 14.4 近期／直接相关数据：会限制我们哪些创新表述

| 工作／数据 | 本轮核验到的内容 | 对本论文的影响与证据边界 |
| --- | --- | --- |
| **VCGAN，IJDAR 2024** | 主体使用 IAM 合成划线；§6.3 另收集真实学生笔记中的划线词，文中用于丰富训练并展示真实结果。官方有 [Real-Strike-off 数据](https://github.com/shivii/Real-Strike-off-dataset) | 不能说以前没有自然学生笔迹。正文未给出足够清晰的真实子集规模、独立划分与完整配对说明，不能把所有结果表都解读为严格真实测试。[作者存档全文](https://dspace.networks.imdea.org/bitstream/handle/20.500.12761/1820/s10032-024-00466-x.pdf?isAllowed=y&sequence=1)，PDF p.6 §6.3、§7 |
| **A study of handwritten text recognition with cross-out words，IJDAR 2026；LTU 数据** | 12 位书写者；总共 **27,961** 个词图，其中 **3,496 干净、24,465 有划线**；真实纸笔与人工数字书写并存，按要求施加七种划线；70/15/15 划分，多数没有同图干净配对 | 多书写者真实划线已经存在。另有 IAM-synth：有效 clean word train/val/test **47,950/7,558/20,306**，每词生成七种划线变体。研究比较直接识别适配、COR 恢复后识别、组合路线。**主要 COR 路线比较在 IAM 合成数据；真实集 §4.7 主要研究 CRNN 适配**，不能混成所有路线都做了大规模真实比较。真实集是否按书写者隔离需进一步查分割 ID；原文明示的 writer-independent IAM 分割不自动适用于真实集。[全文](https://link.springer.com/article/10.1007/s10032-026-00603-8)，§4.1、4.7；[数据](https://huggingface.co/datasets/LTU-ML/handwritten_cross-outs) |
| **Zhong 等，SPIC 2025；SOW** | 作者摘要明确研究划线词恢复及描述题评价；HWG 2026 作者从原作者取得 SOW，记录来源为 **66 张学生答卷**，排除合成划线后剩 **295 个真实划线词** | “恢复被划内容用于教育评价”已有直接前作。66/295 是后续作者对 SOW 的核查，**不是本轮取得 Zhong 全文后的复述**；295 也不是 SOW 全部数据规模。尚未确认作者恢复源码／权重。[Zhong 机构记录](https://salford-repository.worktribe.com/output/3227736)、[HWG 全文 §3.3](https://www.korpuslab.uni-hamburg.de/en/konvens2026/46.pdf) |
| **MathScrub，TLT 2026；Math-Strike** | 官方数据卡说来自 **2024–2025 学年大学多元微积分期中／期末考试**；含真实与 20 种合成划线，保存整页、连通域位置与标签，相关论文摘要描述数学转写流程 | **数学考试、真实划线、整页和识别链路都不是空白**，是方案二必须认真比较的近邻。标签区分保留符号与待移除划线／噪声，但是否恢复“被作废文字本身”、是否有划线前配对底稿，尚不能由摘要／卡片确定；不得擅自宣称与我们完全相同或完全不同。CC-BY-NC-4.0。[官方数据卡](https://huggingface.co/datasets/Incinciblecolonel/MathStrike)、[作者机构摘要](https://research.cuhk.edu.hk/en/publications/mathscrub-single-pipeline-strike-out-removal-and-latex-generation/) |
| **BRESSAY，ICDAR 2024** | 葡语真实作文，**1,000 页／1,000 书写者、30,090 行、416,826 词**；提供页、段、行、词对应关系。转写用 `--text--` 标可读删改，`--xxx--` 标不可读删改；按页分割，同页裁块保持同分区 | 自然修改和多粒度数据早已有之；上述词数不是划线正例数。适合作为外部转写／可读性候选，不等于像素恢复配对集。未见修改前干净图说明。CC-BY-NC-SA-3.0。[IAPR TC11 官方数据说明](https://tc11.cvc.uab.es/datasets/BRESSAY_1)、[作者发布](https://zenodo.org/records/11637681) |
| **HWG，KONVENS 2026** | 划线检测资源包括 collected **402** 个自然划线词、written **1,848** 个人工书写划线词、synthetic **37,010** 个合成划线词、SOW **295** 个真实划线词；另含干净负例 | 已研究合成到真实的落差及加入不同真实数据的收益，但任务是**识别／过滤作废词，非恢复下面的文字**。不能只拿“合成有落差”当新发现。IAM 原图不随包再分发，SOW 部分只发布标签。[全文](https://www.korpuslab.uni-hamburg.de/en/konvens2026/46.pdf)，§3–5；[资源说明](https://zenodo.org/records/21560739) |
| **Chaudhuri–Adak，Pattern Recognition 2017** | 机构摘要描述英文、孟加拉文共 500 页，检测、骨架分析和图像修补 | 真实／合成细节、恢复监督和具体分割尚未全文核实。不能把它归为“只有合成训练／测试”。[作者机构记录](https://research-repository.griffith.edu.au/items/62aec45f-e43b-5cc5-8750-478674b49fa1) |

上述数据的“公开可读”不等于原图、许可和干净底稿都满足我们的使用条件。后续选择外部测试前，仍要核查图像可获取性、身份去重、标注语义和许可。

### 14.5 三篇统一文档恢复的原测试，不能误认成划线恢复

| 方法 | 原训练与测试 | 与我们任务的关系 |
| --- | --- | --- |
| **DocRes，CVPR 2024** | 去形变：Doc3D 100K → DIR300；阴影：FSDSRD 14,200 合成 + RDD 4,371 真实 → Jung 87、Kligler 300、OSR 237；外观：Doc3DShade 90K + RealDAE 450 → RealDAE 150、DocUNet 130；去模糊：TDD 40K → 1.6K；二值化：其他 DIBCO 与文档集合 → DIBCO 2018 | 多任务且有真实测试，但原生五任务没有划线文字恢复。[正式全文](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_DocRes_A_Generalist_Model_Toward_Unifying_Document_Image_Restoration_Tasks_CVPR_2024_paper.pdf)，PDF p.5 §4.1 |
| **Uni-DocDiff，ACM MM 2025** | 去形变：Doc3D + UVDoc → DocUNet／DIR300；去模糊：TDD 40K → 1.6K；二值化以 DIBCO 2018 为测试；另做阴影、光照恢复。去手写：比赛数据与 EnsExam 共 **26,895** 训练图，EnsExam **115** 测试图 | 去手写原任务是删掉手写、留下印刷文本，目标与保留底层手写字不同。它也不是所有任务都只在合成数据上测试。[全文](https://arxiv.org/abs/2508.04055)，PDF p.6 §4.1；不将去阴影段落中 SD7K／FSDSRD 命名歧义强行解释为已核实的统一计数 |
| **MMDIR，CVPR 2026** | 主要混合退化训练 **110K**，来自干净 CDLA／CDDOD／FSDSRD 后数字合成模糊、阴影、水印、印章；MixedDoc 测试 **1,837**，来自 M6Doc，使用分离的覆盖素材；另有单任务测试 | 混合退化主测试是合成，不能等同于真实黑色手写交叉；其单任务训练／评测又包含真实阴影文档，不能概括为整篇“没有真实数据”。[正式全文](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_MMDIR_Multimodal_Instruction-Driven_Framework_for_Mixed-Degradation_Document_Image_Restoration_CVPR_2026_paper.pdf)，PDF p.5 §4 |

### 14.6 我们已经跑过的数据，和上面的原论文实验是两回事

1. **我们重训六个基线**：共同使用本项目 `paper2_restoration_pairs20261007_v4` 的 **10,788 对合成训练图、596 对合成验证图**，以独立 **16 处真实人工区域**按既定规则选版；后续读回测试为 **81 处可读划线**，像素测试为 **7 题**。这是“作者架构在我们的任务上适配”，不是完成原论文全部 benchmark 复现。
2. **作者权重直接迁移**：此前 18 份加 StainRestorer 3 份，共 21 份，权重来自各自原任务；在我们的 **11 道验证题、41 处可读划线**及独立 **16 处像素区域**验证。不同批次原图转写存在变化，因此按同批原图配对解释，不能不加处理混排。
3. **我们的历史版本**：基础合成、框内合成、重绘合成、人工真实微调的训练历史和监督量不同。历史 57.4% 不能与上述数字拼起来宣称公平击败基线。

这些事实来自 [RESULTS.md](RESULTS.md) 的 2026-10-07 至 10-09 记录。本轮未重跑上述实验、未改变数值，也未重新查询正在执行的训练状态。下一次正式主表要冻结共同数据、真实标注预算、预训练条件、额外掩膜输入和验证选版规则。

## 15. 两种论文方案：推荐共同主线，明确研究与系统的边界

### 15.1 方案比较

| 问题 | 方案一：行／局部恢复为核心 | 方案二：整道题输入为核心 |
| --- | --- | --- |
| 核心目标 | 从划线与原字重叠的局部图中恢复原字和公式 | 在完整作答中自动定位、恢复、拼回，再识别 |
| 优点 | 细笔画分辨率高，监督较容易，适合先验证文字保护和交叉笔画恢复 | 能验证切行遗漏、跨行干扰、上下文、正常内容保护和实际识别收益 |
| 主要风险 | “英文词换成中文行 + 新模型”还不足以自然构成强贡献；数学公式不总能切成单一文本行 | 把定位、切行、恢复、VLM 和教育分析都设为创新点会分散；直接缩整页会损失小笔画 |
| 创新不能只靠 | 真实数据、合成到真实落差、模型比跨域权重高几分 | 整题输入、恢复后 OCR／VLM、数学考试数据或教育应用这些名称 |
| 合适的角色 | **主要科学问题、恢复监督、方法对比和消融** | **同一系统的完整输入接口、整题测试和有限应用验证** |

**建议选“方案一的研究核心 + 方案二的整题验证”。** 不是要求同时做两篇规模的工作：训练可按带适量上下文的行／局部公式区域进行，系统仍接收完整题图，自动定位并处理、保持坐标拼回。若相邻行相交或是二维公式，可合并区域而不是强制水平切断。所有模型采用同一裁切／拼回策略，避免把预处理优势误算成新模型优势。

“输入整题”不要求一个网络把整页一次性缩小后预测；内部裁块是实现选择。需要向审稿人证明的是保持分辨率的必要性、定位失败的真实代价、恢复后的完整作答是否更好读。**整题接口本身不包装为模型创新。** 如果要提出利用全题上下文的新模块，应在固定局部恢复器和输入预算下，先证明它对局部歧义有独立收益。

### 15.2 一句话目标与三个可检验的挑战

目标建议写成：**面向真实学生数学作答，分离作废划线并恢复可辨认的原始文字与公式，在保护未修改内容的同时，提高被划内容的识别可靠性。** 恢复出的字仍带“已作废”属性；模型不能把学生写错的式子自动改成正确答案。

1. **同色墨迹混叠。** 划线和文字相交，仅按深浅、区域或擦除面积不能区分。要验证哪些交叉结构被保住，哪些被截断。第一篇的区域掩膜 F1 约 0.7 也不能保证划线像素纯净：区域掩膜与墨迹相交后，原文字笔画仍在里面。
2. **真实修改与合成监督之间存在差异。** 自然划线位置和字形、被改内容有关，反复涂抹不等于随机贴一根线。需要用受控实验量化差距，不能预先说全部已有合成方法都无效。
3. **真实干净答案缺失，部分内容不可恢复。** 已经划掉后扫描的交叉像素，人工也未必能确认其原样。要区分确定可擦除、确定要保留、遮挡不确定；全部涂黑区域不能用猜测当真值。

第一篇侧重找到涂改；第二篇应在**内容恢复和保真**上形成独立问题。仅复用第一篇网络、多训练一轮并换数据，不自动构成新的方法贡献。方法设计应由验证集错误证据驱动，例如如何在擦除划线时保护细笔画、如何处理交叉处的不确定性；暂不先承诺某个 GAN／扩散架构一定有效。

### 15.3 数据集的卖点应该怎样表述

不建议写：“现有数据都是合成的／效果不好，所以提出真实数据集。”

更准确的立论候选是：“既有资源分别覆盖合成词图、受控真实划线、自然作答或文档污渍，但对**真实中文数学修改中，被划内容保真恢复的监督和评测**仍需专门核查与补充。”这是本轮提出的研究定位，**不是已经证明‘此前没有任何同类数据’**；尤其需要补齐 MathScrub 和 Zhong 的全文、样本及标签语义再定独创性表述。

我们的数据若要有明确价值，优先补齐以下组合，而不是只增加页数：

- 自然发生的学生修改，保留整题—行／公式块—涂改区域的坐标和书写者归属。
- 可读的被划内容转写、未划内容转写和作废属性；不能恢复的区域单独标记。
- 可确认的划线／文字像素，以及交叉处未知标记；人工擦除图不能直接命名为“完整干净真值”。
- 一小组**划线前后真实扫描配对**，专门检验交叉笔画恢复；与自然作答子集分开描述，不冒充自然修改。
- 按学生划分训练、验证、测试；同一学生的不同题、同一页的不同切块和同一干净底稿的合成变体不跨分区。

现有精细擦除标注可继续服务于划线分离与文字保护，不必废弃；但要声称“重建被覆盖的原字”，还需要上述配对证据或可靠文字级真值。数据量是否足够应由学习曲线和书写者覆盖判断，不由“满 500 条”单独决定。

## 16. 支撑论文结论所需的实验及讨论顺序

### 16.1 最小实验组合

| 要证明的结论 | 必须比较什么 | 防止误读 |
| --- | --- | --- |
| 新模型恢复能力更好 | 同数据、同真实标注预算、明确预训练与掩膜信息条件下比较专用划线模型、通用恢复模型和我们的方法 | 官方跨域权重是迁移实验，不能代替公平适配；新模型不能独享更多真实监督后只归功于架构 |
| 真实数据有必要 | 固定模型，合成训练／真实训练／合成预训练加真实适配；既报完整预算，也做相同样本量或相近训练预算控制 | 区分数据质量、数量、语言和书写者变化，不能把中文与英文差异全归为合成不真实 |
| 跨数据集能泛化 | 我们训练 → 完全未见的外部真实集；外部训练 → 我们真实测试；分开报告零样本迁移和少量适配 | Dracula 适合有配对的外测但只有一个书写者；LTU／BRESSAY 更适合文字识别外测，不能假装都有像素真值 |
| 恢复有助于识别 | 原图直接读、仅去划线、恢复后读；另有“识别器直接适配划线”这一强对照；各路线同一转写目标 | 先恢复再识别并非天然最好，2026 IJDAR 已比较过这些路线。固定 VLM 提示和解码；提示必须要求读取被划内容，不能再混入“忽略作废字” |
| 系统适用于完整题图 | 同一测试题的原图与处理图、自动裁切流程与人工区域上限；保留无涂改页／行作保护性对照 | 只挑成功定位的划线区域会隐藏漏检；整题结果需包含未检出涂改和正常内容损伤 |

专用基线继续以 CycleGAN、COR 为核心；VCGAN 在实现问题解决后加入，Zhong 取得资源后优先评估。DE-GAN、DocDiff、NAFNet、Restormer、StainRestorer 作为不同恢复能力的对照，不称为全都专门去划线。**不因文献清单长就继续扩张训练队列。**

### 16.2 指标应回答“恢复对了什么”，而非只看干净程度

- **被划内容：** 可确认字符／公式的读回率、严格完整正确率，并统计新增／替换的错误符号；单纯读回更多字符可能伴随更多臆造。
- **未划内容：** 正常墨迹误擦、有效字符识别损失，以及整题未修改区域的变化。无涂改对照不能从正式系统评测中删除。
- **交叉与不可读区域：** 按划线密度、交叉严重度、可恢复性分别报告；“不知道”与错误补字分开计，不给不可确认原字虚构正确答案。
- **图像级恢复：** PSNR／SSIM 和文字前景指标只用于真正有对齐干净底稿的部分；人工擦除标签只支撑它实际确认的像素，不给未知交叉像素强行打分。
- **不确定性与样本独立性：** 按学生／题目聚合，给出配对差与置信区间；增加同页切块不等于增加独立书写者。最终方案冻结后使用未用于反复选方案的测试集。

### 16.3 后续优先级

1. **先定任务边界和标注语义。** 主张“恢复可辨认的被划内容”，保留作废属性；教育部分限定为修改前后内容的可观测展示／识别收益，不推测学生心理原因。
2. **先审计现有标注能提供什么真值。** 区分自然作答、可确认擦除、未知交叉、可读转写；检查按学生隔离及与第一篇训练来源的关系，再规划扩标。
3. **优先补齐最相关前作的比较条件。** Dracula 配对集、LTU 多书写者集，以及 MathScrub／Zhong 的标签目标，比再加入一个通用恢复网络更影响论文定位。公开资料不足的部分继续明确保留未知；需要联系作者时另行取得发送授权。
4. **固定共同训练与评测协议后，再决定模型改进。** 先从验证集证实误擦、残留、交叉缺笔和幻觉哪项是主要瓶颈，再设计对应模块与消融。整题实验沿用高覆盖切行并保留失败统计，不必先改成整页网络。

本节是方案与证据要求，不代表这些新增实验已经执行，也不改写现有训练结果。
