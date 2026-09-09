# torch-kirigami

基于 PyTorch FX 的通用结构化剪枝库。依赖核心独立分析结构位置、传播选择和检查约束；剪枝层提供自动选择、原 Module 的物理收缩、静态计划和最终结构 checkpoint。

首次审阅建议阅读[当前完整架构说明](docs/architecture.md)，其中包含各层职责、索引传播、统一扩展、规划执行、保存恢复及支持边界。

## 使用

Python 3.10+、PyTorch 2.6+。仓库使用 uv 管理环境；开发锁使用 PyTorch 2.14 CPU，安装后的库依赖为普通的 `torch>=2.6`。

```bash
uv sync --locked
uv run --locked python examples/dependency.py
uv run --locked python examples/pruning.py
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
```

```python
import torch
from torch import nn
from torch_kirigami import DependencyGraph

model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))

selection = graph.parameter("0.weight").axis(0).select([1, 4])
impact = graph.propagate(remove=[selection])

assert impact.status == "resolved"
print(graph.explain(impact))
# 联动到第一层 bias，以及第二层 weight 的输入列；不会修改 model。
```

`resolved` 表示当前请求的结构关系与约束已解决，不表示现有执行器能够执行，也不表示与未剪枝模型数值等价。分组不均衡、未知算子或无法紧凑表示的选择会给出诊断。

## 结构化剪枝

```python
from torch_kirigami.pruning import ChannelRatio, Magnitude, Pruner

pruner = Pruner(model, graph=graph)
plan = pruner.plan(metric=Magnitude(p=2), budget=ChannelRatio(0.34))
print(plan.explain())
model, result = pruner.apply(plan)
# 不需要预览计划时：model, result = pruner.prune(metric=..., budget=...)
```

手工请求使用 `pruner.plan(remove=[selection])`；默认保护输入输出，必要时显式设置 `preserve_io=False`。自动预算允许欠达，但返回的计划必须通过完整执行检查。plan 是纯静态数据，可导出并在兼容原模型上重放；权重更新后重放不会重新评分。结构、模式、配置和布局必须符合计划前提。

```python
from torch_kirigami.pruning import load_checkpoint, save_checkpoint

save_checkpoint(model, "pruned.pt")
# 在另一个脚本中提供兼容的原始模型定义，不需要依赖图或剪枝历史：
restored = load_checkpoint(make_original_model(), "pruned.pt", map_location="cpu")
```

## 能力与边界

- 固定使用 FX symbolic tracing 和 ShapeProp；没有可选前端或捕获降级。
- 覆盖 Linear、普通/分组/深度及转置卷积、池化、归一化、Embedding、常用轴操作、矩阵运算、SDPA/GQA 和 MHA 的已声明剪枝形式，详见[覆盖矩阵](docs/operator-coverage.md)。
- 通过区域选择保留每组不同的参数切片，通过工作队列求联动闭包；同一参数的别名和多次调用共同参与分析。
- 用户扩展与内置规则使用统一 `OperatorRegistry` 和 `OperatorRule`；共享描述贯通依赖分析、候选发现与执行，无独立执行注册表。
- 普通 Python 的 Tensor 数据/shape 条件分支与动态循环不支持；配置常量分支和固定循环遵循 FX 的捕获能力。
- 建图隔离样例输入及注册 buffer，并恢复 CPU/已初始化 CUDA 的随机数状态。模型 forward 必须不写参数、不产生外部副作用；不要与同一模型的训练/执行并发建图。
- 图绑定本次输入结构、元数据、模式和相关配置；结构剪枝后重新建图。更换样例 shape 时不能直接沿用旧图的分析结论。
- 剪枝层支持 L1/L2 magnitude、两种 WeightTaylor、局部/全局通道预算和有界贪心补全；默认保护全部输入输出轴。
- 物理执行保留原 Module，先检查配方再统一提交；剪枝后重新建图并创建 optimizer。复杂索引、未知融合操作、必须改写原 forward 的变化需要额外支持，不能保证任意 Module 自动兼容。

[架构与 API 契约](docs/dependency-graph-design.md) · [扩展示例](examples/custom_rule.py) · [测试与兼容性](docs/testing.md) · [结构化剪枝设计与扩展 API](docs/pruning-design.md)

[逐类契约与组合测试](docs/testing-coverage.md) · [全部注册入口的测试清单](docs/operator-test-coverage.md)

[融合 GQA 的完整扩展示例](examples/fused_attention.py) 展示一处定义完成自动剪枝与 checkpoint 恢复。

`experiments/capture_probe.py` 保留为早期技术调研材料，其中 export/JIT 实验不属于当前库的实现流程。
