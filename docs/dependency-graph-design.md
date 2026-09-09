# 依赖图：固定 FX 流程与独立结构语义

当前实现采用自审修订方案。依赖模块独立于评分、候选选择和模型修改；内部明确使用 FX，不设置前端适配器、不建立第二套计算 IR。

## 1. 从模型到结构关系

1. 在 tracing 前收集原模型参数、buffer、模块对象和全部注册路径。同一个参数对象对应一个实体；同一模块的不同调用对应不同调用节点。
2. 使用官方 FX Tracer，扩展只通过公开叶子钩子和 autowrap 配置接入。根模块需要作为叶子时，用公开 Graph API 构造一个 call_module。
3. 根据原 forward 签名绑定 args/kwargs 和默认值，ShapeProp 通过 placeholder 获取绑定后的值；支持普通 tuple/list/dict 容器和已捕获的变长参数。
4. 保留形状、stride、dtype、device 与有限的尺寸表达式，不保留中间激活。FX 图是计算事实来源；结构关系作为独立注解保存。
5. 按明确的算子身份调用局部语义规则。捕获失败抛出 CaptureError；捕获成功但缺少规则时保留图，相关传播结果为 unresolved。

样例执行只用于元数据。它不会替 symbolic tracing 决定数据分支，也不能证明其他样例 shape、配置或路径的有效性。没有自研具体执行 tracer，没有 autograd/export/JIT 降级。FX 公开接口及其兼容性边界参考 [PyTorch 2.6 文档](https://docs.pytorch.org/docs/2.6/fx.html)。

## 2. 选择与关系

TensorRef 表示参数、buffer 或 FX 值；AxisRef 表示物理轴。IndexSet 使用规范化的半开区间，Selection 使用 Cartesian Region 的并集。所有位置均采用建图时的原始坐标，保留顺序固定为原始顺序。

轴选择是删除整条轴截面的便利接口。一般参数变化需要多个区域，例如 Conv2d(6,4,groups=2) 删除输入通道 0、4：

| 输出行 | 每行删除的局部输入列 |
| --- | --- |
| 0、1 | 0 |
| 2、3 | 1 |

这两个区域不能合并成对整个权重统一删除输入列 0、1。AxisRelation 的 Port 可限定张量分区；BlockMap 表达偏移、重复对应，并明确区分触及块与完整移除块的传播条件。相同表达支持普通 Conv、grouped Conv 和 depthwise multiplier，没有特殊剪枝 callback。

其他关系包括静态 SliceRelation、PermuteRelation 和基于原逻辑元素顺序的 ReshapeRelation。只物化必要的索引区间，不为激活建立标签张量；超过 4096 个区间/区域的复杂映射明确报 analysis_limit，不近似成 identity。

区域并集先去重并保持互不重叠。传播使用增量工作队列，节点收到新增选择后重新传播，直至闭包。Selection 的等价性按选中位置判断，不依赖矩形分解方式。Provenance 记录源选择、实际新增目标区域与关系原因，避免同一位置被多条路径重复统计。

约束与映射分离：

- NonEmpty：当前操作要求保留非空结构。
- Balanced：多个分区保留数量相同，不要求每组删除相同局部索引。
- BlockBalance：剩余逻辑组具有相同的成员数；允许整组消失或均衡减少每组成员。
- Divisible：保留尺寸需要整除指定因子。
- Fixed：调用方额外保护某个轴。
- Layout：选择必须能构成受支持的紧凑或分区布局。
- Barrier/AxisBarrier：相关操作或轴缺少可证明的语义。

Balanced、BlockBalance 和 Divisible 不替策略选择补充位置；结果保留未满足的约束。Depthwise 删除输入通道会删除其全部输出；只删除部分输出时，可以继续均衡降低 multiplier，也可以补全整组删除，分析器不会代选。

同一权重被多个操作使用时，每次调用分别检查布局，不能把不同调用允许的布局取并集。对分区参数自身物理轴的额外 Fixed/Divisible/Balanced 约束，无法证明时返回 partitioned_constraint；可以改为约束其对应的逻辑输入/输出轴。

## 3. 查询契约

```python
graph = DependencyGraph.build(model, args=(x,), kwargs={})
weight = graph.parameter("encoder.conv.weight")
impact = graph.propagate(remove=[weight.axis(0).select([2, 5])])
```

- parameter(path)、buffer(path)：按原模型路径查询；别名返回同一个对象。
- calls(path)：返回该模块对象的所有调用。FX 可能把多个别名归一到同一个 target，不能据此声称恢复了每次调用原本使用的 Python 属性名。
- CallRef.input()/output()：按展平后的 Tensor 端口顺序查询；容器结构仍保留在 FX 图及 OperationContext 中。
- values()、metadata(ref)、relations、constraints、shape_expressions：检查分析依据。context 记录 PyTorch 版本、梯度/推理模式、模块模式与规则快照身份。
- fx_graph：只用于检查的副本，修改它不会改变当前依赖结果。
- propagate(remove=[...], constraints=[...])：联合处理多个删除请求，不修改模型。
- explain(impact)：解释受影响区域、传播原因、阻碍和修改要求。

Impact 包含 requested、selections、parameters、buffers、interfaces、diagnostics、requirements、provenance 和参与检查的 constraints。complete 单独标记影响范围是否完整，不能把尚未满足的平衡约束与未知影响混为一谈。

| 状态 | 含义 |
| --- | --- |
| resolved | 当前图和规则前提下，确定联动及约束已解决 |
| unresolved | 尚需补充选择、语义规则或更强的布局分析 |
| conflict | 请求违反明确约束，例如删空固定分组或改变保护轴 |

这些状态不表示具有执行能力，也不证明模型精度或与原模型的数值等价。没有自动评分、自动补选或权重 surgery。

## 4. 一个语义扩展接口

OperatorRegistry.default() 提供内置规则；用户注册明确的模块类型或函数对象。Tensor 方法通过 register_method 注册。重复注册直接报错，每次建图复制注册表，不使用全局规则状态。注册对象是 OperatorRule，不另设执行注册表。

OperationContext 提供规范化参数、TensorRef、张量 metadata、模块及参数/buffer 绑定、FX 节点和尺寸表达式。OperatorRule.analyze 返回 OperatorSpec，包含关系、约束、Requirement、共享分区描述和可选候选轴。扩展应该是确定、无副作用的语义分析函数；内置规则与用户规则调用路径相同。

opaque=True 的模块/函数通过官方 FX 扩展机制保持为叶子，内部结构由规则负责。自定义子类不会自动继承基类规则。根叶子与嵌套叶子均受支持；根叶子的变长签名目前明确拒绝。任意 Python callable 的包装仍受官方 FX autowrap 能力限制。

[可运行的外部模块规则](../examples/custom_rule.py) 展示不修改核心、不实现执行器即可加入结构语义。

## 5. 尺寸与后续执行层

ShapeExpr 记录常量、输入维度读取、numel 和受支持的整数运算、reshape 的 -1。来源无法证明时不能只根据最终 shape 猜测。尺寸表达式单独调度其消费者；即使该调用的张量没有删除位置，尺寸变化也必须触发语义参数检查。

Requirement 保存目标、关联张量及结构化数据，包括属性轴绑定、图中尺寸表达式、旧分区大小和布局要求。对应的保留索引可从 Impact 获取；各项要求仍须后续执行器检查和具体化。

例如 LayerNorm 的 normalized_shape、Linear 的 in_features/out_features、Unflatten 的 unflattened_size 具有规则明确建立的绑定。Squeeze 遇到新产生的单例轴时会要求保持捕获时的输出 rank，必要时需用明确 reshape 替换；普通 Python 闭包和硬编码整数没有自动属性来源。图操作数可以需要修改，但不能因此保证原 Python forward 可自动修复。

结构关系按保留当前操作及已声明布局要求分析，不自动引入任意 gather、重排或删除整段程序。输出接口受影响会被报告，是否保护由消费者提供 Fixed 约束。

torch.nn.utils.prune 可以在后续用于 mask 训练/验证；它不会物理缩小 Parameter，因此不作为依赖求解器或物理执行器。

## 6. 状态、有效期与当前边界

建图隔离 args/kwargs 和注册 buffer，尽可能通过共同 deepcopy 保留输入及 buffer 的共享关系；恢复原 buffer 绑定、模块模式和 CPU/已初始化 CUDA RNG。普通属性及 plain list/tuple/dict 中对注册 Tensor 的引用也通过共享绑定工具临时重定向到 clone，成功和失败均恢复原容器及绑定。参数不整体复制，forward 必须不写参数、不修改外部 Python 状态。同一 OperatorRule 的 preflight/effects 入口在 ShapeProp 前检查写入，包含 Embedding max_norm 和已识别的参数/别名写操作；这不是任意 Python 代码的副作用沙箱。

同一绑定描述记录普通容器里的注册 Tensor/Module 引用和相邻标量配置，供 graph、静态 plan、apply 和 checkpoint 共用。不同 Tensor 对象即使共享注册 storage，也不能当作可直接重绑定的对象别名；容器中这类视图在执行前拒绝。任意自定义对象、闭包、外部全局容器中隐藏的 Tensor 引用不在扫描范围内，调用方不得通过它们绕过注册绑定；不承诺通用 Python 对象迁移。

非叶 Tensor 样例等无法 deepcopy 的输入明确失败；输入或 buffer 共享参数 storage 也拒绝。不同 Parameter 对象共享 storage 不合并，涉及它们的重写返回 unresolved。不要对同一模型并发建图或训练。

当前区域表示不能保留零元素张量的独立轴删除意图，因此样例或中间执行结果含零元素张量时明确拒绝建图。原模型注册的 forward/pre-forward hooks（包括根模块、被 FX 展开的父模块及 PyTorch 全局 hooks）也在执行前拒绝；它们的任意代码不属于当前算子描述。建图后增加这类 hook 会使图过期，静态 plan 的 apply 也重新检查。PyTorch 无公开的全局 hook 查询接口，对其稳定注册表的读取集中在一个辅助函数内，并在最低/开发版本测试。

快照记录注册结构、对象身份、形状、dtype/device/stride、模块模式与可识别的标量及嵌套 list/tuple 配置。依赖图和静态计划共用配置冻结逻辑，区分 list/tuple 且不保留可变列表引用。普通权重数值更新不使图过期；结构、对象替换、模式、已记录配置改变会抛出 StaleGraphError。任意外部状态变化不能由此得到完整检测，修改结构后必须重新建图。

标量 guard 保留精确类型，True、1、1.0 不等同；浮点数使用稳定表示，NaN 配置在结构 guard 中可匹配。tensor_bindings() 提供经过一次完整校验的批量注册绑定读取；用户回调或模型可能变化后必须再次验证，不把返回值当作锁。

grad/inference 上下文记录仅用于说明捕获前提，不是完整的 Python 路径 guard。forward 若依赖 torch.is_grad_enabled()、is_inference_mode_enabled() 或外部全局状态选择不同结构，切换后必须重新建图；跨这些上下文复用结构分支不属于支持契约。普通模型可在 no_grad 中查询或 inference_mode 中 apply，不因上下文本身不同而一律拒绝。

当前覆盖卷积及转置卷积、常用归一化/池化/Embedding、矩阵与轴操作、SDPA/GQA/MHA 等已声明形式；每类可剪轴、自动候选和执行边界见[覆盖矩阵](operator-coverage.md)。小型静态整数索引记录值 guard，修改这些值必须重新建图。

明确边界包括：数据相关索引、未知融合算子、卷积空间/卷积核裁剪、不能保持紧凑布局的 reshape，以及删掉 unbind 输出端口。未登记算子不会按名称或同 shape 猜测为逐元素操作。CPU 及 RTX 5070 Ti 上的 PyTorch 2.14 CUDA 已完成测试，覆盖算子、独立数值参考和状态恢复，详见[测试与兼容性](testing.md)。这些结果不代表其他设备或自定义 kernel 已验证。

早期的 export/JIT 实验只作为调研记录保留，不参与当前实现。
