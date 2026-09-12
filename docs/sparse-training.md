# 稀疏训练组件

库提供可组合组件；算法的选组、强度、阶段切换和训练循环放在 `examples/workflows/`。所有稀疏正则统一返回标量 loss，经 autograd 求导，没有直接修改 `.grad` 的第二套入口。用户负责任务 loss、数据和 optimizer。

## 结构组与正则

```python
from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import CandidateSpace
from torch_kirigami.sparsity import GroupLasso

graph = DependencyGraph.build(model, args=(example_input,))
space = CandidateSpace(graph)
regularizer = GroupLasso(space.parameter_groups())

optimizer.zero_grad()
task_loss = criterion(model(inputs), targets)
sparse_loss = regularizer()
(task_loss + strength * sparse_loss).backward()
optimizer.step()
```

`CandidateSpace(graph, candidates=None, axes=None, preserve_io=True, constraints=())` 与 Pruner 共用候选发现和默认输入输出保护。显式 candidates 必须配显式 axes；显式轴保留保护域的预算分母。`protected_axes` 记录自动发现时因 IO 保护而排除的域，显式轴模式下为空。`impact(candidates)` 查询联合闭包；`parameter_groups(candidates=None, parameter_filter=None)` 返回逐候选完整参数组。filter 接受 (TensorRef, Parameter)。

同组区域与别名去重，完全等价的组规范化；不同重叠组仍分别贡献正则。过滤为空或影响不完整明确报错。可补全的计数约束仍由最终 plan 检查。也可用 `ParameterGroup(graph, selections, key="...")` 显式指定参数区域。

| 模块 | 标量定义 |
| --- | --- |
| `ScaleL1(graph, parameters)` | 显式一维参数区域并集的绝对值之和 |
| `GroupLasso(groups, coefficients=None)` | sum(a_g * norm(W_g, 2)) |
| `GroupSquaredL2(groups, coefficients=None)` | 0.5 * sum(a_g * sum(W_g**2)) |

coefficients 是不求导的非负有限 Python 标量，默认一，无隐式组大小归一化。等价组系数冲突会拒绝。逐组系数改变时重新构造正则对象；总强度在外部乘入。ScaleL1 接受参数路径或 TensorRef，不自动选择 BN 或 gate。

每次调用读取最新权重，不保留跨步计算图，不修改参数、梯度、BN buffer 或 optimizer。低精度归约至少 float32，float64 保留；零点采用零次梯度，不以平滑改变公式。当前稀疏组件支持单图、单设备的稠密实数参数，不自动收集分布式或分片参数。非有限选中值和结果明确报错。

结构组不是零不变组，不保证置零与物理删除等价。LayerNorm、GroupNorm 或 attention 归约域变化要按紧凑模型验证。

## 训练生命周期

梯度累积对整个组合 loss 按有效 batch 归一化，避免正则强度随微批次数增加。AMP 使用普通 GradScaler 流程，累积完成后去缩放、裁剪和 step，不额外添加正则梯度。WeightTaylor 的任务梯度另外采集，示例不将稀疏正则梯度混入评分。

数值更新可以继续使用图；结构、绑定或建图模式等前提改变后，重新构造图和组。每次 apply 后重建 optimizer；普通微调关闭稀疏正则。库不迁移动量、scheduler 或梯度。

## 整数与累计预算

`ChannelCount(counts, axes, scope="local")` 的 local counts 是逐轴整数上限，global counts 是一个总整数上限。两种预算共用现有联合计量、约束补全、保护和欠达报告。

```python
from torch_kirigami.pruning import Magnitude, Pruner
from torch_kirigami.sparsity import CumulativeChannelBudget

accounting = CumulativeChannelBudget(space)
budget = accounting.budget(space, 0.5)
model, result = Pruner(model, graph=space.graph).prune(metric=Magnitude(), budget=budget)
space = CandidateSpace(DependencyGraph.build(model, args=(example_input,)))
accounting.update(result, space)
```

累计比例始终相对于首次绑定的逻辑域宽度：先 floor 原始目标，再扣除实际删除数量。global 使用原始总宽度，不附加 local 比例。budget() 不推进状态，update() 核对结果前后结构和重建图；未知变更、增宽、域声明改变会拒绝。

最后一个通道或 head block 可能使某域变为整体 IO 保护。累计预算使用新图的路径与维度重新绑定这个初始域，保留分母和实际剩余宽度，即使它不再出现在自动候选轴中；手动丢弃域或声明变化仍会拒绝。恢复训练时同样保留这些已受保护的初始域。

`state_dict()` 保存初始/当前宽度、稳定域声明及结构前提；`load_state_dict(state, space)` 先验证，再绑定等价恢复模型。这里统计逻辑结构数量，不等于参数量、FLOPs 或延迟比例。

## 调度与统计

Constant、Linear、Polynomial 和 Piecewise 都接受显式非负 step，不维护内部进度。Polynomial 公式为 start + (end-start) * progress**power；区间外钳位，Linear 的 power=1。Piecewise 在 milestone 当步使用新值，首 milestone 为零。训练恢复保存构造配置和外部 step。

`selection_similarity(left, right)` 接受“域字符串 → 保留位置整数集合”的映射，计算逐域 Jaccard 后平均，空集对空集为一。SelectionWindow(size) 记录相邻比较，窗口未满返回 None；物理剪枝后须 reset，不能跨坐标阶段比较。阈值、截止时间和触发政策留在示例。窗口保存恢复先验证再提交。

## 门控与参数操作

ChannelGate(size, axis, trainable=True) 沿激活轴逐位置缩放，weight/mask 初始为一。set_mask() 只接受匹配尺寸的二值 mask。创建 gate 后再创建 optimizer。

先用 register_gate_operators(operators) 显式注册规则，再建图。GateBinding(graph, module_path).candidates(space) 找出联动到门控的候选；GateMagnitude(bindings) 对受影响的 abs(weight*mask) 并集评分。共享 weight 和 mask 的别名不重复计算；共享 weight、具有不同 mask 的门控分别贡献分数。无门控候选应显式排除，不能默认为零分。

Gate 不新增预算域，不用 forward hook。其乘法产生独立输出，规则显式声明该分配行为，因此后接单消费者的原地 ReLU 可以通过剪枝验证；多消费者的原地别名检查仍保留。物理剪枝同步收缩关联结构、gate 和 mask，保留剩余非一缩放。恢复工厂含相同 gate 定义。opaque 融合模块内部的门控必须由该模块规则声明，GQA 示例给出了完整做法。

scale_groups_(groups, factor) 和 zero_groups_(groups) 对区域并集只更新一次。set_group_norms_(groups, targets) 将组缩放到指定 L2 范数；等价组同目标去重，其他重叠拒绝；零向量到正范数拒绝。操作准备全部结果再提交，不改变梯度或 optimizer state。

不同注册张量共享底层存储时，参数操作在提交前拒绝，即使其中某个别名没有被选中；同一 Parameter 的多个注册路径仍可去重使用。范数投影先稳定归一化再乘目标值，避免对极小非零参数计算溢出的缩放比例；最终值不能由目标参数 dtype 有限表示时仍会拒绝。

这些操作在完整 backward 后、没有待反传图时调用，示例放在成功 optimizer step 后。AMP 跳步时，调用方也应跳过相应操作和进度。置零不安装永久 mask，但不保证重新生长：完整组两端归零可能同时消除任务梯度。软剪枝示例显式建立并保留非零动量，安排不带投影的恢复步骤；永久约束和状态政策由算法定义。

## 示例、保存与边界

参见[算法示例](../examples/workflows/README.md)。最终模型沿用 save_checkpoint/load_checkpoint；公共示例工具将模型与 optimizer、算法、调度及 RNG 状态分别保存。restore_training 先恢复紧凑模型、创建 optimizer，再加载训练状态；有状态组件用新图重新绑定。

当前不提供期望 L0、完整 D-Gating/OTO 优化器、控制网络、FLOPs/延迟预算或自动网络改写。示例展示组件组合，不宣称论文精度复现。

方法来源：[Network Slimming](https://arxiv.org/abs/1708.06519) 的 BN 缩放 L1、[Growing Regularization](https://arxiv.org/abs/2012.09243) 的递增惩罚、[DPM](https://arxiv.org/html/2406.03879v2) 的平滑收缩，以及 [OCSPruner](https://arxiv.org/html/2501.13439v2) 的组选择稳定性。示例使用本库的依赖区域分组和普通优化器；范数衰减作用于选中区域并集，不实现 DPM 的梯度纠错，稳定性阈值和阶段切换也采用示例自己的配方。
