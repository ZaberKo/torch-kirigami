# 算子覆盖与剪枝边界

支持指当前 FX 图、样例元数据和规则声明下的能力，不代表该算子的任意参数、轴或存储形式都可剪枝。模块、函数和方法仅对实际登记的公开拼写匹配，不按名称猜测未知函数。

| 家族 | 结构分析与物理执行 | 默认自动候选 | 主要边界 |
| --- | --- | --- | --- |
| Linear / functional linear | 输入输出特征、权重行列和 bias | 模块输出特征 | 不继承自定义子类语义 |
| Conv / ConvTranspose 1d–3d | 普通、分组和深度卷积通道；按原分区切片拼接 | 输出通道，深度卷积默认整组 | 空间/卷积核轴固定；functional 尺寸参数须无需改写 |
| BN / LN / GN / InstanceNorm / RMSNorm / PReLU / normalize | 仿射参数、统计 buffer 和声明属性联动 | 无 | GN 固定组数与平衡；归一化在紧凑域重算 |
| Embedding | 特征轴、后续消费者 | 特征宽度 | 词表轴固定；max_norm 在执行样例前拒绝 |
| Max/Avg/AdaptivePool、MaxUnpool、pad、interpolate/Upsample | 通道与 batch 对应；池化值/索引共同约束 | 无 | 空间变化固定；不能用本规则剪 padding 产生的新通道 |
| 常用逐元素数学、激活、比较、where、masked_fill | 广播对应与多输入联动 | 无 | 未登记函数仍为未知；复杂写入/别名不推断 |
| to/type_as、常用 dtype/device 方法、clone/detach/contiguous | 数据坐标保持，转换参考的 shape 不构成广播依赖 | 无 | view/copy 切换的写入安全仍需证明 |
| matmul/mm/bmm、addmm/baddbmm、einsum | 收缩轴、自由轴和广播 batch | 无 | einsum 要求显式输出方程；不支持操作数内重复标签/对角语义 |
| cat、stack、split、chunk、unbind | 分段与端口对应 | 无 | stack/unbind 端口固定；split/chunk 原写法须保持保留坐标与端口 |
| 基础切片、narrow、index_select | 原坐标映射及执行后坐标检查 | 无 | 正步长基础索引；index_select 使用捕获常量整数向量及值 guard；不自动重写索引 buffer |
| transpose/permute、reshape/view、flatten、squeeze/unsqueeze | 轴变换和有来源的尺寸计算 | 无 | 硬编码尺寸、rank 变化或不可证明 view stride 拒绝相关请求 |
| repeat/tile、repeat_interleave、expand | 重复和广播映射 | 无 | 静态正重复因子；repeat_interleave 需标量次数和显式 dim |
| GLU、ChannelShuffle、PixelShuffle/Unshuffle | 成对 gate、通道置换和完整通道块 | 无 | Shuffle 保守要求对应组的局部保留模式一致；pixel 空间轴固定 |
| Unfold/Fold | 通道与 im2col 通道块 | 无 | batched 2D 形式；空间和 kernel 位置固定 |
| sum/mean/prod、amax/amin、logsumexp、softmax/log_softmax | 区分归约轴，保留轴联动 | 无 | 不支持返回位置索引的归约；变化后重新计算紧凑域 |
| scaled_dot_product_attention | Q/K 特征、K/V 序列、V 输出、batch/head 和 mask | 无 | GQA 整 KV 组或合法 multiplier 收缩；不负责 KV cache 更新 |
| MultiheadAttention | 打包/分离投影、自/交叉注意力、固定 head 数的平衡宽度收缩 | 宽度 | batch/token/mask 位置固定；不实现保持外部宽度的内部删 head |
| 第三方融合模块 | 一个 OperatorRule 声明关系、候选、布局/属性及必要 lowering | 规则可声明 | opaque 内部语义由扩展保证；保存不需要另写算子规则 |

默认候选使用逻辑轴宽度计量。Embedding 使用 weight 的轴 1；分组 ConvTranspose 使用完整输出通道轴，不能把每组局部 weight 列数作为总宽度。

未知算子或不支持的轴只阻断涉及它的影响分析。手工联合请求整体拒绝，自动策略可在其他独立结构路径选择候选；预算欠达会报告原因。

尚不提供动态 Python 控制流捕获、数据相关索引重映射、稀疏/量化存储重写、RNN/PackedSequence、MoE 路由、KV cache 重建、optimizer state 迁移或原 Python forward 改写。

数值验收使用独立保留域公式或手工构造的紧凑模型；能完成 forward 不是充分验收。测试范围见 [testing.md](testing.md)。
