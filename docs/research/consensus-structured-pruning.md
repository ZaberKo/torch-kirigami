# 通用结构化剪枝：Consensus 调研

调研日期：2026-09-06。本报告使用 Consensus 搜索并对下列 10 篇论文逐篇 fetch，随后打开原始 arXiv 页面校对相关出版信息。检索覆盖经典方法和 2024–2025 年扩展，不声称穷尽全部近期论文。API 建议是根据文献与当前依赖图目标作出的工程推论，不是论文给出的库接口。

## 流程共性与必须区分的概念

常见的训练后流程是：

```text
定义可变化结构与预算
→ 建立依赖关系、提出候选删除请求
→ 可选的统计采集/校准
→ 评估候选影响
→ 在结构约束与预算下选择组合
→ 联合传播并检查最终请求
→ 生成可审查的修改计划、执行物理收缩
→ 验证并可选恢复训练
→ 下一轮重新建图/统计/评分
```

DepGraph 支持“由一个结构位置出发，通过依赖关系找到联动变化”这一理解 [1]，但起点只是定位方式。真正删除的对象是传播后的多个张量区域。一个候选也可能由多个起始选择组成，例如一个 head、固定分组中各组不同的通道集合。

不能把以下三者统一叫作 group：

| 概念 | 回答的问题 | 谁负责 |
| --- | --- | --- |
| 结构联动范围 | 必須随某个请求一起变化的张量区域有哪些？ | 现有依赖图 |
| 评分对象与聚合 | 使用哪些参数/激活证据，如何得到候选分数？ | Importance |
| 比较域与预算范围 | 哪些分数可以相互排名，每个范围允许改变多少？ | Selection policy |

Isomorphic Pruning 的关键恰是最后一项：异构结构不一定具有可比较的分数分布 [2]。这不意味着把原有依赖组重新拆开，更不等同于 Conv 的 groups。

## 并非所有方法都是 metric → top-k → slice

- 范数基线可以直接评估候选、排序、物理收缩，再微调 [3]。
- Taylor 方法依赖损失与梯度的统计定义，且可迭代重估 [4]；LLM-Pruner 将此类评估和 LoRA 恢复结合 [7]。
- Network Slimming 先通过训练形成结构稀疏性，最后的幅值排序只是流程的一部分 [5]。
- LASSO + 重建同时涉及选择和剩余权重拟合 [6]；二阶后训练方法也可能联动更新未删除权重 [9]。
- SPA 将依赖组与不同重要性估计连接起来 [8]；近期 Optimal Brain Connection 进一步强调结构内部和层间交互，以及恢复机制 [10]。

因此基本实现可以从 magnitude / 明确定义的 Taylor 开始，但公共接口不应把所有方法强制表达为“每层给一个 channel 分数向量”。高级方法可能直接评价候选集合、在组合中重新评分，或产出补偿所需信息。

## 对 API 边界的建议

这是供讨论的职责划分，不要求一次实现所有插件系统。

1. **候选请求**：保存一个或多个原坐标 Selection、来源名称与稳定 ID。根节点可以是默认候选生成器的入口，但不成为依赖核心的必需字段。
2. **评估**：接受候选及其 Impact、只读模型绑定和可选统计结果；返回按候选 ID 对齐的分数、可用性及语义说明。支持批量评分；不要求逐参数可加性。
3. **选择策略**：接受候选、评分器/已有分数、预算及约束，输出请求组合和选择解释。简单实现是确定性排序；高级实现可做联合搜索，并调用评分器重新评价组合。
4. **执行准备与执行**：将最终联合 Impact 转成显式修改计划并验证执行器能力。分析已解决与执行器已支持是不同状态。补偿/微调允许作为独立步骤，不嵌入依赖规则。
5. 统计采集、代价评估、轮次调度可以先作为小型辅助接口或调用方流程。无需为了可扩展性立即搭建另一个训练框架。

建议基础实现先支持显式候选选择、magnitude、从明确梯度快照计算的一阶 Taylor、确定性的局部排序，以及最终组合的联合验证。激活评分、二阶统计、稀疏训练、复杂硬件预算、同构划分和恢复算法可逐项加入，但不能在类型契约上被排除。

## 组合约束与去重

以下是从现有一般区域表示推导出的正确性要求：

- 单个请求 unresolved 不一定意味着永远不能选择。例如固定分组平衡约束下，单独删一个通道不成立，但和另一组的一个通道组合后可以成立。候选层不能过早把所有 unresolved 候选丢弃；最终组合必须重新传播验证。
- 结构联动闭包相同的入口需要识别为等价请求，避免重复选择。只共享部分删除区域的候选并不等价。
- magnitude 或权重 Taylor 的参数贡献，应按 Parameter 实体和原坐标区域求并集；参数别名、多次模块调用、多条传播路径不能重复计入同一个物理参数位置。
- 成本同样要按最终删除并集计算。两个候选共同删除同一参数时，节省参数量不能相加；矩阵同时减少输入和输出宽度时，FLOPs 的组合收益也通常不是单项之和。
- 激活统计与参数统计的去重单位不同。共享参数只有一个实体，但同一模块的不同调用确实有不同激活和数据分布，不能简单按模块对象合并。
- 组评分的 sum、mean、norm、先分块再平均等应由方法明确指定。可配置的归一化也不能声称自动解决任意结构之间的可比性；比较域仍属于策略 [2]。

## 梯度统计的契约

“基于 grad”不是充分的方法名称。至少应区分以下一阶代理量，其中 U 是去重后的删除参数区域，g 是梯度：

```text
sum_U abs(w * mean_batch(g))
abs(sum_U w * mean_batch(g))
mean_batch(abs(sum_U w * g_batch))
mean_sample((sum_U w * g_sample)^2)
```

这些不是同一个量；最后一种涉及平方统计，不能从最终累计的 .grad 恢复。也不能把 minibatch 梯度平方默称为 per-sample Fisher。这里的公式仅解释 API 应区分的统计语义，不宣称它们是某篇论文的同一种实现。

统计结果应标记：模型/图版本、损失定义及归约单位（样本或 token）、采样权重、batch 或 sample 粒度、累加次数、绝对值/平方发生的位置、模型模式。AMP 梯度需先去除缩放因子；调用方梯度的是否保留应明确。不能默默使用未知来源或过期的 .grad。

共享参数的 .grad 已汇总其参与的各次调用贡献；再次按模块调用累加同一个 .grad 会重复计数。训练更新可以不使依赖图过期，却会使先前 importance/统计快照过期。因此图有效性与评分有效性需分开。

## 文献与证据

Consensus 部分记录未更新会议名称，且第 6、9 篇作者有遗漏；相关作者及出版信息已按原始论文页面校正。

1. [DepGraph: Towards Any Structural Pruning](https://arxiv.org/abs/2301.12900) — Gongfan Fang, Xinyin Ma, Mingli Song, M. B. Mi, Xinchao Wang；2023；2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)。结构耦合决定必须共同变化的参数；范数评分仍须面对整个联动结构。 DOI: 10.1109/cvpr52729.2023.01544。[已 fetch 的记录](https://consensus.app/papers/depgraph-towards-any-structural-pruning-fang-ma/c1c65150e3255fe0816f19e40e2525bb/?utm_source=chatgpt)。

2. [Isomorphic Pruning for Vision Models](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4364_ECCV_2024_paper.php) — Gongfan Fang, Xinyin Ma, Michael Bi Mi, Xinchao Wang；2024；ECCV 2024。异构结构的分布、规模和拓扑使统一全局排序失真；同构结构分别比较。比较域是策略配置，不改变依赖关系。 DOI: [10.1007/978-3-031-73404-5_14](https://link.springer.com/chapter/10.1007/978-3-031-73404-5_14)。[已 fetch 的记录](https://consensus.app/papers/isomorphic-pruning-for-vision-models-fang-ma/bae161672c77507b8f9f89d2c74ac277/?utm_source=chatgpt)。

3. [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) — Hao Li, Asim Kadav, Igor Durdanovic, H. Samet, H. Graf；2016；ICLR 2017（预印本 2016）。整滤波器删除及相连特征图收缩，为无校准数据的 magnitude 基线提供依据；2016 是预印本年份，ICLR 2017 正式发表。 DOI: 10.48550/arxiv.1608.08710。[已 fetch 的记录](https://consensus.app/papers/pruning-filters-for-efficient-convnets-li-kadav/ea9e79e6a39b5b3cb3981c9273ed0bf1/?utm_source=chatgpt)。

4. [Importance Estimation for Neural Network Pruning](https://arxiv.org/abs/1906.10771) — Pavlo Molchanov, Arun Mallya, Stephen Tyree, I. Frosio, Jan Kautz；2019；2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)。以一阶、二阶 Taylor 估计结构对损失的贡献并迭代删除。统计采集及聚合顺序属于方法定义。 DOI: 10.1109/cvpr.2019.01152。[已 fetch 的记录](https://consensus.app/papers/importance-estimation-for-neural-network-pruning-molchanov-mallya/b01b628515815eecb070006b0c08a242/?utm_source=chatgpt)。

5. [Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519) — Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, Changshui Zhang；2017；2017 IEEE International Conference on Computer Vision (ICCV)。训练中施加通道稀疏性，再删除并微调；训练正则化并非普通只读评分函数。 DOI: 10.1109/iccv.2017.298。[已 fetch 的记录](https://consensus.app/papers/learning-efficient-convolutional-networks-through-liu-li/49d1bf2ffedf53c4aebdd13d901d416d/?utm_source=chatgpt)。

6. [Channel Pruning for Accelerating Very Deep Neural Networks](https://arxiv.org/abs/1707.06168) — Yihui He, Xiangyu Zhang, Jian Sun；2017；2017 IEEE International Conference on Computer Vision (ICCV)。LASSO 选择与最小二乘重建交替执行；需要联合选择和保留权重补偿接口。 DOI: 10.1109/iccv.2017.155。[已 fetch 的记录](https://consensus.app/papers/channel-pruning-for-accelerating-very-deep-neural-he-zhang/38089466bb44503d8f8bd17873fb9eb0/?utm_source=chatgpt)。

7. [LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627) — Xinyin Ma, Gongfan Fang, Xinchao Wang；2023；NeurIPS 2023。发现耦合结构、利用梯度评估、剪枝后 LoRA 恢复。恢复策略应独立于结构依赖分析。 DOI: 10.48550/arxiv.2305.11627。[已 fetch 的记录](https://consensus.app/papers/llmpruner-on-the-structural-pruning-of-large-language-ma-fang/077991c6632753c1938962aad1960c6f/?utm_source=chatgpt)。

8. [Structurally Prune Anything: Any Architecture, Any Framework, Any Time](https://arxiv.org/abs/2403.18955) — Xun Wang, John Rachwan, Stephan Günnemann, Bertrand Charpentier；2024；ArXiv。ONNX 计算图上的依赖分组与组评分，适配不同训练时机；可借鉴分层边界，不建议据此引入新的捕获前端。 DOI: 10.48550/arxiv.2403.18955。[已 fetch 的记录](https://consensus.app/papers/structurally-prune-anything-any-architecture-any-wang-rachwan/cc34654d653053369b653138541464da/?utm_source=chatgpt)。

9. [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://arxiv.org/abs/2208.11580) — Elias Frantar, Sidak Pal Singh, Dan Alistarh；2022；NeurIPS 2022。校准数据支持的二阶/OBS 后训练压缩，不要求完整重训练；作为补偿接口证据，不把所有结果都当成通道结构剪枝结果。 DOI: 10.48550/arxiv.2208.11580。[已 fetch 的记录](https://consensus.app/papers/optimal-brain-compression-a-framework-for-accurate-frantar-alistarh/7feb31f6bbb75c739c65864165098c8e/?utm_source=chatgpt)。

10. [Optimal Brain Connection: Towards Efficient Structural Pruning](https://arxiv.org/abs/2508.05521) — Shaowu Chen, Wei Ma, Binhua Huang, Qingyuan Wang, Guoxin Wang, Weize Sun, Lei Huang, Deepu John；2025；ArXiv。2025 年预印本提出考虑参数交互和层间依赖的 Jacobian 评分，以及微调期补偿；可作为将来联合评分/恢复接口的压力案例，不以少量引用认定其普适优越。 DOI: 10.48550/arxiv.2508.05521。[已 fetch 的记录](https://consensus.app/papers/optimal-brain-connection-towards-efficient-structural-chen-ma/253becd8cefd52fd9a08aaa75fb47151/?utm_source=chatgpt)。

