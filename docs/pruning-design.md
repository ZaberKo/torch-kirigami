# 结构化剪枝：统一算子定义、静态计划与持久化

依赖核心独立于评分、选择和模型修改。固定复用 FX symbolic tracing 与 ShapeProp；保留原 Module，不改写 Python forward，不引入新 tracer 或计算 IR。前期调研见 [Consensus](research/consensus-structured-pruning.md) 和 [原生搜索](research/web-structured-pruning.md)。

## 1. 一次调用与可检查的两步流程

```python
graph = DependencyGraph.build(model, args=(x,))
pruner = Pruner(model, graph=graph)
model, result = pruner.prune(metric=Magnitude(p=2), budget=ChannelRatio(0.2))

# 或者，在模型尚未修改时检查和保存决策：
plan = pruner.plan(remove=[selection])
print(plan.explain())
model, result = pruner.apply(plan)
```

上面两种写法是替代关系；一次结构修改后需要重新建图。`prune()` 只封装 `plan() → apply()`，没有另一套选择逻辑。plan 做决策，apply 只验证和执行已确定的配方；静态数据设计不会取消这个职责边界。

- 手工 remove 与自动 metric/budget/candidates/strategy 互斥。手工请求整体成功或抛 PlanningError，不暗中调整请求。
- 默认用 Fixed 保护输入输出的全部张量轴；可设置 preserve_io=False 并追加 constraints。改变输入结构后，调用方负责提供兼容的数据和布局。
- 自动选择可以欠达，但接受的联合请求必须整体可执行。模型结构修改后，由调用方重建图和 optimizer。
- 返回的 model 始终是原对象。PruningResult 包含 plan、最终 structure、旧新 parameter_map、分段 coordinate_maps 和 report，不重复包含 model。
- Pruner(model) 可独立执行已恢复的 plan；plan/prune 必须提供 graph。

## 2. 一个特殊场景定义一次

```python
operators = OperatorRegistry.default()
operators.register(MyFusedModule, OperatorRule(analyze=my_analysis))
graph = DependencyGraph.build(model, args=(x,), operators=operators)
```

可以继承 OperatorRule，也可以提供纯回调。局部注册表精确匹配模块类型、函数对象和 Tensor 方法；重复注册报错，子类不自动继承语义。不透明模块和函数使用 FX 官方叶子/包装机制，包含根模块。

同一规则具有不同生命周期入口：

| 入口 | 职责 |
| --- | --- |
| preflight(node, module) | 在 ShapeProp 前检查配置相关写入，例如 Embedding max_norm |
| effects(node, module) | 返回 CallEffects，统一描述输入写入和确定的新存储 |
| analyze(OperationContext) | 返回 OperatorSpec，声明结构语义 |
| lower(RewriteContext) | 可选特殊 lowering；默认消费共享描述生成修改配方 |

OperationContext 提供原 FX 调用、规范化参数、输入输出元数据、参数/buffer 绑定、尺寸表达式和小型整数常量。OperatorSpec 包含关系、约束、requirements、候选轴、分区布局、调用契约及常量值 guard。依赖核心只分析这些事实，不调用 lower。

复用层次为：索引/布局/约束原语 → 算子家族 → 复合扩展。PartitionedLayout 只声明一次原权重分区；执行层直接在这些分区内计算保留区域，不重新根据 Conv 类型推导 groups。CandidateAxis 显式声明稳定域 key、逻辑预算轴和默认块大小，不能假定所有输出宽度都是 weight.axis(0)。

普通属性绑定及分段布局使用通用 lowering；特殊 lower 返回 TensorRecipe、AttributeRecipe、已处理 requirements 和可证明的 output_strides，不直接修改模型。扩展须证明自己声明的融合语义；evaluate 默认关闭，不会为了检查尺寸运行第三方 forward。算子知识不进入保存/恢复逻辑。

[融合 GQA 示例](../examples/fused_attention.py) 用一个分析定义声明整 KV 组联动、候选和结构属性，没有额外执行或保存 callback。它支持整 KV 组剪枝；SDPA 规则另外支持合法的 Q/KV multiplier 收缩。

## 3. 候选、评分与预算

没有额外建立全图“等价剪枝组”。默认入口由算子定义提供，包括 Linear、Conv/ConvTranspose、Embedding 特征轴、MHA 宽度，以及第三方声明的结构域。参数别名、重复调用按稳定域去重；其余重叠通过联合 Impact 处理。

ChannelRatio 的比例相对于本轮原逻辑轴宽度，不相对于候选数量。local 使用每轴 floor(ratio × width) 上限；global 使用总宽度上的单个上限，不暗加局部比例。自动发现只在冻结基线前排除可证明完全受默认输入输出保护的域；之后的未知算子、执行限制和策略限额只产生欠达。显式 axes 不自动排除，自定义 candidates 必须指定 axes。

Candidate(key, remove, axis=None) 是提交种子的集合，不是全局不可拆块约束。axis 由默认发现关联预算域；自定义候选仍使用 ChannelRatio.axes 计量。真正整块联动由关系/约束决定。

Metric(context, candidate_batch) 返回有限、对齐的一维分数；Strategy(context) 返回已登记 key。两者共享 PlanningContext，统计由 metric 对象持有。临时组合 Candidate 可以联合评分，不假设分数可加。框架重新检查策略输出的约束、实际联合预算和执行支持。

| Metric | 对受影响参数区域并集的计算 |
| --- | --- |
| Magnitude(p=1) | sum(abs(w)) |
| Magnitude(p=2) | sqrt(sum(abs(w)²)) |
| WeightTaylor(elementwise_abs) | sum(abs(w × grad)) |
| WeightTaylor(joint_abs) | abs(sum(w × grad))，跨参数先累加 |

共享调用、别名和行列交集不重复计分。默认包含 bias 和归一化参数，不含 buffer；支持 parameter_filter。低精度至少 float32 累积，float64 不降精度。Taylor 使用调用方当前、未缩放的稠密实数梯度；缺梯度等错误不填零，不代管 backward/optimizer，不解释为逐样本 Fisher。loss reduction、梯度累积、AMP 去缩放由调用方负责。

Greedy(max_trials=10_000) 静态评分后按 (score, key) 排序，尝试联合加入并补全 Balanced/Divisible，不回溯、不搜索全部 BlockBalance 补全。每次追加须新增位置，整体可执行才接受；达到限额或没有进展时停止，不声称数学上无解。空请求也必须合法，允许从不满足 Divisible 的原宽度搜索到合法宽度。

Impact.complete 区分影响范围是否已知；计数/布局约束暂未满足不等于范围未知。影响不完整不能按部分参数评分。联合缓存最多 32 个 Impact，计入限额的查询即使命中缓存也算一次，不在搜索中物化新权重。本版未承诺大模型候选搜索性能。

## 4. 尺寸、坐标与原 forward

删除选择使用增量闭包；尺寸表达式另有依赖调度。即使张量没有删除位置，读取其他张量尺寸的消费者也要检查。例如 y.size(1) 改变 x.sum(dim=...) 的归约轴，输出 shape 相同也必须拒绝。

每项 Requirement 必须被处理：完成声明修改、证明原写法仍有效，或拒绝请求。

- reshape/view 重算有来源的尺寸读取、整数运算和 -1，不因整数恰好相等而猜测属性绑定；view 另检查 stride。
- slice/narrow/static index_select 检查保留后的原坐标；split/chunk/unbind 检查边界与端口，不只核对 shape。
- 作为索引使用的注册整数张量记录值 guard；修改这些值会使图和相关计划失效。其他权重训练更新不需要内容哈希。
- squeeze 检查新单例轴是否改变 rank。sum/mean/softmax/normalization/attention 在紧凑域重算，不统一采用置零 mask 等价参考。
- 原地写入需要确定的新存储与消费者条件。无法证明的 view/copy 别名、未知布局会阻断受影响请求。
- 原始 contiguous/channels_last/channels_last_3d 参数布局由配方记录。meta 运算提供尺寸信息；卷积家族另明确处理 channels-last，不能把原始 meta stride 当作所有设备上的证明。

未知规则、无法修改的硬编码或不支持的布局只影响相关结构路径。手工联合请求整体拒绝；自动策略可选择其他独立路径。捕获本身失败则无法建立图。

## 5. 静态计划与提交

PruningPlan 保存 analysis 摘要、原坐标配方、属性修改、候选/预算/原因，以及 before/after 结构记录。它不保存 live Impact、模型、Parameter、FX 节点、callback、对象地址或签发者身份。冻结 TensorRef 使用稳定标签；查询时以 paths 辨认原注册绑定，不依赖原图 UUID。

plan() 不修改模型、梯度、图、Pruner 或全局 RNG，不登记已签发计划，不分配新权重。自定义回调同样须只读；这不是任意 Python 代码的副作用沙箱。

```python
torch.save(plan.to_dict(), "plan.pt")
restored_plan = PruningPlan.from_dict(torch.load("plan.pt", weights_only=True))
model, result = Pruner(make_original_model()).apply(restored_plan)
```

编解码使用封闭的版本化数据类型，不动态导入 callback。反序列化及 apply 检查坐标范围、完整配方、共享绑定、结构前后条件和配置。输入文件中的静态决策仍以对应模型代码/config 为前提，不证明任意 Python 程序等价。

apply 不重新评分、追踪或调用 forward。先在 inference_mode(False)/no_grad 中分配全部张量，再集中提交；普通错误撤销绑定、属性和结构元数据。新参数是普通叶子，保留设备、dtype、requires_grad、共享关系和原顺序，受影响 .grad=None。

计划可重放到兼容原模型，不维护消费标志；实际结构变化后重复执行会因旧结构不符失败，空计划可重复执行。权重值可在计划后变化，但原评分不再保证适合当前权重；结构、模式、配置、布局和常量 guard 必须匹配，执行期间禁止并发修改。参数 hooks、训练附加状态和 optimizer state 不自动迁移。

## 6. 最终结构 checkpoint

```python
save_checkpoint(model, "pruned.pt")
model = load_checkpoint(make_original_model(), "pruned.pt", map_location="cpu")
```

主路径保存最终结构、原生 state_dict 和普通非持久 buffer 值。load 直接建立最终尺寸参数/buffer，恢复声明属性、共享绑定、模式和权重，不重放剪枝、不运行 forward、不需要图、样例或算子注册表。

成功 apply 在模型上附带隔离命名的纯数据结构记录，仅跟踪框架管理的结构及验证前提。保存时读取当前真实状态；支持多轮剪枝后继续训练、设备迁移和重新保存，不保存历史计划。检测到外部结构修改时拒绝猜测修复。

加载要求兼容原始模块骨架及配置。自定义模型类应放在稳定可导入的模块中，类型路径属于结构前提。所有张量先准备；原生 load_state_dict/extra-state hooks 在隔离模块壳上运行，普通自定义状态通过事务提交。用户 hook 的外部副作用和任意自定义资源生命周期不属于回滚保证；复杂 storage 重叠和特殊 tensor 存储明确拒绝。

同一 Parameter 的所有别名恢复到同一个新对象，重复 state_dict 键必须具有一致值。普通构造函数可能先分配完整模型，不能承诺任意构造函数免分配初始化。map_location 改变保存值的设备，调用方仍需在目标后端验证自己的模型前提。

保存 plan 用于复用删除决策；保存 checkpoint 用于恢复剪枝后的模型及训练权重。直接 torch.save(model) 继续使用 PyTorch 原生机制，不增加另一套库级封装。

支持范围见 [算子覆盖矩阵](operator-coverage.md)，测试环境见 [testing.md](testing.md)。
