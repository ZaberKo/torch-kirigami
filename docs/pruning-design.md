# 结构化剪枝：统一算子定义、静态计划与持久化

依赖核心独立于评分、选择和模型修改。固定复用 FX symbolic tracing 与 ShapeProp；保留原 Module，不改写 Python forward，不引入新 tracer 或计算 IR。

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
| lower(RewriteContext) | 可选特殊 lowering；返回 None 时，由剪枝编译层消费共享描述生成修改配方 |

OperationContext 提供原 FX 调用、规范化参数、输入输出元数据、参数/buffer 绑定、尺寸表达式和小型整数常量。OperatorSpec 包含关系、约束、requirements、候选轴、分区布局、调用契约及常量值 guard。依赖核心只分析这些事实，不调用 lower。

每个 Requirement 可通过 `arguments=(ArgumentRef("groups", 6),)` 声明自己负责验证的具体调用参数。name 使用共享别名解析，position 按 FX 参数计数，variadic 可表示剩余形状位置参数。未声明的尺寸来源参数继续要求保持不变；不再使用调用级参数变化豁免。依赖图按数据与尺寸来源统一激活执行要求，执行层必须处理所有已激活要求。扩展作者不能只声明许可而省略对应证明。

复用层次为：索引/布局/约束原语 → 算子家族 → 复合扩展。PartitionedLayout 只声明一次原权重分区；执行层直接在这些分区内计算保留区域，不重新根据 Conv 类型推导 groups。CandidateAxis 显式声明稳定域 key、逻辑预算轴和默认块大小，不能假定所有输出宽度都是 weight.axis(0)。

普通属性绑定及分段布局使用通用 lowering；特殊 lower 返回 TensorRecipe、AttributeRecipe、已处理 requirements 和可证明的 output_strides，不直接修改模型。`RewriteContext.compact_shape(ref)` 查询普通紧凑形状，分区布局须使用对应描述。扩展须证明自己声明的融合语义；evaluate_on_meta 默认关闭，不会为了检查尺寸运行第三方 forward。算子知识不进入保存/恢复逻辑。

[融合 GQA 示例](../examples/fused_attention.py) 用一个分析定义声明整 KV 组联动、候选和结构属性，没有额外执行或保存 callback。它支持整 KV 组剪枝；SDPA 规则另外支持合法的 Q/KV multiplier 收缩。

## 3. 候选、评分与预算

没有额外建立全图“等价剪枝组”。默认入口由算子定义提供，包括 Linear、Conv/ConvTranspose、Embedding 特征轴、MHA 宽度，以及第三方声明的结构域。参数别名、重复调用按稳定域去重；其余重叠通过联合 Impact 处理。

ChannelRatio 的比例相对于本轮原逻辑轴宽度，不相对于候选数量。local 使用每轴 floor(ratio × width) 上限；global 使用总宽度上的单个上限，不暗加局部比例。自动发现只在冻结基线前排除可证明完全受默认输入输出保护的域；之后的未知算子、执行限制和策略限额只产生欠达。显式 axes 不自动排除，自定义 candidates 必须指定 axes。

Candidate(key, remove, axis=None) 是提交种子的集合，不是全局不可拆块约束。axis 由默认发现关联预算域；自定义候选仍使用 ChannelRatio.axes 计量。真正整块联动由关系/约束决定。

Metric(context, candidate_batch) 返回有限、对齐的一维分数；Strategy(context) 返回已登记 key。两者共享 PlanningContext，统计由 metric 对象持有。临时组合 Candidate 可以联合评分，不假设分数可加。框架重新检查策略输出的约束、实际联合预算和执行支持。

PlanningContext 的候选、预算、轴、约束和目标是只读输入，trials、limit_reached 和 exclusions 用于记录策略诊断。候选域按实际 AxisRef 去重；稳定 key 仅标识声明，不能通过重复 key/轴包装改变分母。原生算子的参数别名和变长参数在共享入口解析，尺寸来源与执行检查复用相同语义。

| Metric | 对受影响参数区域并集的计算 |
| --- | --- |
| Magnitude(p=1) | sum(abs(w)) |
| Magnitude(p=2) | sqrt(sum(abs(w)²)) |
| WeightTaylor(elementwise_abs) | sum(abs(w × grad)) |
| WeightTaylor(joint_abs) | abs(sum(w × grad))，跨参数先累加 |

共享调用、别名和行列交集不重复计分。默认包含 bias 和归一化参数，不含 buffer；支持 parameter_filter。低精度至少 float32 累积，float64 不降精度。Taylor 使用调用方当前、未缩放的稠密实数梯度；缺梯度等错误不填零，不代管 backward/optimizer，不解释为逐样本 Fisher。loss reduction、梯度累积、AMP 去缩放由调用方负责。

L2 先按区域最大绝对值缩放再平方，以 hypot 合并区域范数，避免有限极端权重在平方时上溢或下溢。内置评分在设备端合并区域，按设备/批次传回结果；Greedy 按 Impact 缓存容量处理内置指标候选。自定义 metric 仍收到完整候选批次，不假定批次可拆，也不改变非可加联合评分契约；未对 CUDA 性能提升给出未经测量的比例。

Greedy(max_trials=10_000) 静态评分后按 (score, key) 排序，尝试联合加入并补全 Balanced/Divisible，不回溯、不搜索全部 BlockBalance 补全。每次追加须新增位置，整体可执行才接受；达到限额或没有进展时停止，不声称数学上无解。空请求也必须合法，允许从不满足 Divisible 的原宽度搜索到合法宽度。

Impact.complete 区分影响范围是否已知；计数/布局约束暂未满足不等于范围未知。影响不完整不能按部分参数评分。联合缓存最多 32 个 Impact，计入限额的查询即使命中缓存也算一次，不在搜索中物化新权重。本版未承诺大模型候选搜索性能。

## 4. 尺寸、坐标与原 forward

删除选择使用增量闭包；尺寸表达式另有依赖调度。即使张量没有删除位置，读取其他张量尺寸的消费者也要检查。例如 y.size(1) 改变 x.sum(dim=...) 的归约轴，输出 shape 相同也必须拒绝。

嵌套维度读取也检查选择器来源：`x.size(y.size(1))` 和 `x.shape[y.size(1)]` 中，内层结果不能在剪枝后改为另一个轴。通过共享参数约束验证重算值，允许重算后保持原值的表达式；来源不明则阻断相关请求，不按观察到的整数猜测轴身份。功能参数的位置仅由 Requirement.arguments 中的 ArgumentRef 描述，不在 data 中重复存储。

每项 Requirement 必须被处理：完成声明修改、证明原写法仍有效，或拒绝请求。

属性更新另作全图配置验证：在隔离配置副本上用相同 FX symbolic trace 重新捕获，要求节点、边、常量和输出与原捕获一致。副本共享 Parameter、不分配紧凑权重、不执行 ShapeProp；buffer 和 RNG 隔离。任一属性诱发的捕获差异都会拒绝相关请求，包含不消费被剪 Tensor 的另一分支。不能安全复制配置或重捕获失败也明确拒绝。opaque 内部仍由其算子规则负责。

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

AttributeRecipe 使用既有配置冻结/恢复表示保留列表及嵌套容器类型，不能把 list 与 tuple 的前提混同。静态格式已有 FrozenList 支持，未新增持久化记录类型。

```python
torch.save(plan.to_dict(), "plan.pt")
restored_plan = PruningPlan.from_dict(torch.load("plan.pt", weights_only=True))
model, result = Pruner(make_original_model()).apply(restored_plan)
```

编解码使用封闭的当前数据类型，不动态导入 callback。反序列化及 apply 检查坐标范围、完整配方、共享绑定、结构前后条件和配置。输入文件中的静态决策仍以对应模型代码/config 为前提，不证明任意 Python 程序等价。

apply 不重新评分、追踪或调用 forward。先在 inference_mode(False)/no_grad 中分配全部张量，再集中提交；普通错误撤销绑定、属性和结构元数据。新参数是普通叶子，保留设备、dtype、requires_grad、共享关系和原顺序，受影响 .grad=None。

计划可重放到兼容原模型，不维护消费标志；实际结构变化后重复执行会因旧结构不符失败，空计划可重复执行。权重值可在计划后变化，但原评分不再保证适合当前权重；结构、模式、配置、布局和常量 guard 必须匹配，执行期间禁止并发修改。参数 hooks、训练附加状态和 optimizer state 不自动迁移。

## 6. 最终结构 checkpoint

```python
save_checkpoint(model, "pruned.pt")
model = load_checkpoint(make_original_model(), "pruned.pt", map_location="cpu")
```

主路径保存最终结构、原生 state_dict 和普通非持久 buffer 值。load 直接建立最终尺寸参数/buffer，恢复声明属性、共享绑定、模式和权重，不重放剪枝、不运行 forward、不需要图、样例或算子注册表。

成功 apply 在模型上附带隔离命名的纯数据结构记录，仅跟踪框架管理的结构及验证前提。保存时读取当前真实状态；支持多轮剪枝后继续训练、设备迁移和重新保存，不保存历史计划。检测到外部结构修改时拒绝猜测修复。

加载要求兼容原始模块骨架及配置。自定义模型类应放在稳定可导入的模块中，类型路径属于结构前提。所有张量先准备；原生 load_state_dict 和 get/set_extra_state 在隔离模块壳上运行。普通自定义状态通过事务提交，包含属性创建、删除和更新；容器中的临时模块引用映射回原模块，保留共享关系。外部副作用和任意自定义资源生命周期不属于回滚保证。

注册的 state_dict 保存/加载 pre/post hooks 当前明确不支持：保存前和加载目标均检查，避免生成依赖未声明键转换的文件。get/set_extra_state 不属于这项限制。复杂 storage 共享、无法证明无内部重叠的 stride（如 expand 产生的零 stride）和特殊 tensor 存储在保存前拒绝；加载也检查。共享别名值检查允许对应位置的 NaN，同时严格检查其他值，可保存训练异常现场。

同一 Tensor 对象跨 parameter/buffer 类别注册时，在结构快照阶段明确拒绝，保存不会写出无法恢复的 checkpoint；同一类别内的别名继续支持。

静态计划与依赖图共用标量和嵌套 list/tuple 配置 guard；列表冻结为不可变数据，后续原列表修改不会修改计划。apply 拒绝任何注册的 forward/pre-forward hook，包括生成计划后新增的 hook。

capture 在隔离前、apply 和 checkpoint 在提交前拒绝全局 parameter/buffer/module registration hooks，防止回调替换准备好的对象而破坏别名、普通引用及结果映射；提交后还核对实际参数对象身份，普通失败使用撤销记录回滚。捕获退出直接恢复原 buffer 绑定，不再次调用注册回调。

普通容器中的注册 Tensor 引用随物理替换共同提交，并进入静态前提；同一容器被多个属性引用时保持共享关系。checkpoint 基于 prepared 的最终模块图生成完整普通状态编辑，包含父 setter 对子模块的修改及同类型子模块替换后的引用映射；设置与删除使用统一、明确的内部编辑记录。实际 state_dict payload 在写文件前及加载时核对注册键、shape、dtype、别名值，不仅检查注册 hooks。标准无自定义加载逻辑的模型省去原生 load_state_dict 前的重复权重复制；有额外加载逻辑时仍先准备数值供回调使用。

memory format 在 forward/stride 验证前写入完整配方；plan 不再后置修改已经验证的配方。PlanningContext 以有界缓存复用最近四个已通过执行检查的具体 Impact 结果，缓存命中也检查归属和完整引用，不缓存用户评分。最终选择按入口的原预算、约束独立传播和编译；搜索期间的缓存不替代这一步。事务及用户回调边界继续验证模型有效性。空请求合法且每个候选必然删除预算轴位置时，零预算直接返回空计划；不能据此跳过初始就不满足约束的请求分析。

同一 Parameter 的所有别名恢复到同一个新对象，重复 state_dict 键必须具有一致值。普通构造函数可能先分配完整模型，不能承诺任意构造函数免分配初始化。map_location 改变保存值的设备，调用方仍需在目标后端验证自己的模型前提。

加载回调结束后，提交前还核对最终注册张量的类型、存储共享与实际目标设备。回调不得引入与保存结构不符的注册张量布局；设备按本次分配结果核对，允许合法的 map_location 迁移。

checkpoint 通过 `weights_only=True` 加载。extra_state 应使用 Tensor 和基础标量/容器；Path 等任意 Python 对象可能可以被 torch.save 写入，却无法默认安全加载。建议由模型的 get/set_extra_state 转为字符串等基础数据；库不会自动切换到不受限反序列化。普通字典配置和依赖 grad/inference 模式的 Python 分支仍属于既有程序前提边界。

保存 plan 用于复用删除决策；保存 checkpoint 用于恢复剪枝后的模型及训练权重。直接 torch.save(model) 继续使用 PyTorch 原生机制，不增加另一套库级封装。

支持范围见 [算子覆盖矩阵](operator-coverage.md)，测试环境见 [testing.md](testing.md)。

## 捕获、执行与静态数据的一致性

- CallEffects 统一声明新存储和写入效果，捕获与执行共用；OutputContract 只保留布局。Linear/Conv、归一化等原生分配声明集中在 effects，不再分别维护两份。
- 配置重捕获比较实际绑定事实和常量值，忽略 FX 自动名称；清理生成常量属性。symbolic tracing 中未被记录的 buffer 写入拒绝建图，隔离 BN 的元数据执行更新仍允许。
- PlanningContext 对相同属性修改最多缓存 32 个验证结果，张量版本变化时失效；最终计划独立复验。缓存不保存新参数，也不改变策略尝试次数。
- 静态 AnalysisSummary 保存完整张量目录。统一解析原图/portable 标签，区分未知引用和已知空影响；PruningResult 的坐标映射共用该规则。plan、checkpoint 和模型内结构记录均直接使用当前结构，不设置格式版本，不提供历史格式迁移或兼容分支。
- F.linear 的向量权重与标量 bias 合法形式通过轴关系和广播关系复用处理；不承诺 PyTorch 本身不接受的组合。配置冻结/恢复保留 torch.Size，与 list/tuple 使用同一生命周期。

## 稀疏训练组合接口

`CandidateSpace` 统一候选发现、默认输入输出保护和预算域；`ParameterGroup` 提供带新鲜度检查的参数区域并集。`ChannelCount` 增加显式整数预算，与 `ChannelRatio` 共用规划验证。训练组件与累计比例调度位于 `torch_kirigami.sparsity`，依赖核心不反向导入它。详见[稀疏训练契约](sparse-training.md)。
