# 独立 review 核查与修复（2026-09-08）

针对任务 `01a08099-da6f-7bc3-8a97-0d1cb318bc6a` 的 10 项 bug 逐项检查了当前代码，并运行修复前复现测试：22 项失败、1 项通过、9 项 CUDA 跳过（部分问题有多种调用形式）。这 10 类问题均属实；其中 Tensor `//`/`%` 和零 batch 的问题在依赖分析层，原执行检查会拒绝，不能说它们已经导致错误 apply。

| Review 项目 | 处理结果 |
| --- | --- |
| padding 同 shape、异坐标 | 按真实 padding 轴建规则；发生 padding/crop 的轴固定，其他轴与独立分支可剪 |
| 根/父模块 forward hooks 漏检，计划后新增 hook | 捕获前统一拒绝原模型的 forward/pre-forward hooks；图 freshness 和静态 apply 重新检查 |
| 静态计划遗漏 list 配置 | graph/plan 共用递归配置冻结；保留 list/tuple 类型区别，计划不引用原列表 |
| checkpoint extra state 删除丢失、临时模块引用泄漏 | 提交删除及属性更新；共享 deepcopy memo 将临时模块映射回原模块；覆盖普通失败回滚 |
| narrow kwargs 越界 | 共用安全的命名/位置参数读取；函数与 Tensor 方法均测试 |
| MHA kwargs 顺序影响 mask 判定 | 按签名中的 key_padding_mask/attn_mask 获取，不使用张量遍历顺序 |
| Tensor //、% 漏依赖 | Tensor 形式使用逐元素广播关系；标量尺寸表达式仍保留原表达式来源 |
| 零元素样例丢失轴意图 | 明确拒绝零元素样例或中间元数据；修复空 Selection 的投影和 compact_shape |
| state_dict hooks 产生不可加载文件 | 保存前及加载目标检查并拒绝注册的 state_dict pre/post hooks；继续支持 get/set_extra_state |
| 共享 NaN 被误判为冲突 | 对应 NaN 可相等，其余值仍严格比较，共享关系保留 |

额外处理了内部重叠 buffer：保存前检查 stride 能否证明无重叠，加载也验证，避免生成自家无法恢复的文件。该检查是保守的充分条件，不承诺所有特殊 stride 都支持。

隐式因果 attention 的 token 删除会改变三角 mask 的原坐标含义，因此 `is_causal=True` 时固定 Q/K token 轴；特征/head 等声明范围仍可分析。grad/inference 上下文相关的 Python 结构分支不能由 FX 完整检测，文档明确禁止跨上下文复用这种结构路径；普通 no_grad 查询和 inference_mode apply 仍可用。

本轮顺带将区间交集改为双指针扫描，并让 `Greedy(max_trials=0)` 先验证空请求、跳过无用评分。一般非零预算的重复传播、区域 normalize 优化和 GPU 批量评分属于后续性能工作，本轮没有宣称完成。

验收：Python 3.10 / PyTorch 2.6 CPU 和开发 PyTorch 2.14 CPU 各 **207 passed、100 CUDA skipped**；RTX 5070 Ti / PyTorch 2.14 CUDA 使用 `--require-cuda`，**307 passed、零跳过**。Ruff 检查、格式检查、剪枝和融合 attention 示例、wheel/sdist 构建通过。回归案例位于 `tests/test_review_regressions.py`。

## 2026-09-09：新增 review 的核查与处理

继续核对同一任务的新 review。远端 read_thread 未返回，本轮通过该任务的本机会话记录取得新增内容；以下处理针对其第二次审查列出的案例。

| 新发现 | 判断与处理 |
| --- | --- |
| 普通容器缓存注册 Tensor 导致 buffer 隔离/参数替换/checkpoint 漏绑定 | 属实，新增共享绑定工具，统一记录引用和重定向；支持 plain list/tuple/dict、直接属性与共享容器；捕获退出和提交失败恢复绑定 |
| 父 setter 修改子状态、setter 新建子模块后的引用仍指向临时模块 | 属实，按 prepared 最终模块图映射回原模型，对整个图准备普通状态编辑，不按 setter 所属类筛选 |
| True/1/1.0 绕过配置检查、NaN 配置不能往返 | 属实，标量 guard 携带确切类型；浮点采用稳定表示；引用容器里的相邻标量同样受保护 |
| channels-last 权重在规划时被按 contiguous 验证 | 属实，memory format 纳入验证前的最终配方，删除 plan 阶段的后置修改 |
| swapaxes/unpool/inplace 的关键词形式、narrow 负 start | 属实，共用参数名别名与 raw operand 解析；不按遍历顺序猜输入角色；narrow 共用坐标规范化 |
| expand_as 漏掉模板 Tensor 的尺寸依赖 | 属实，增加模板到输出的轴对应，保留数据源的广播关系 |
| conjugate complex buffer 的 NaN 比较回归 | 属实，分别比较实部与虚部，避免 unresolved conjugate 上调用 view_as_real |
| 覆盖 state_dict 方法仍能保存不可加载文件 | 属实，写文件前验证实际 payload 的键、shape、dtype 和别名值；加载复用同一验证 |
| 全局 forward hooks 绕过本地 hook 检查 | 属实，捕获、freshness 和 apply 使用同一全局/本地 hook 检查 |

采纳的精简包括：批量读取已验证 Tensor 绑定；复用最近四份编译结果；零预算在有明确证据时跳过评分；普通 checkpoint 加载避免二次复制权重；统一状态设置/删除记录和保留坐标助手；删除未使用的 `_attribute`、`_source_modules`、`CallContract.shape_arguments`。

在相同 5/10/20 层 Linear 手工规划探测中，完整 fingerprint 次数均为 **9**，不再随参数数量逐次扫描。默认 256 宽零预算案例的测试要求传播次数不超过 5，并禁止调用 metric；最终编译结果复用也有计数回归测试。性能结论限定于这些案例，不代表所有自动搜索都已优化。

未采纳大范围改写 Requirement 类型或整体搬迁 native 算子助手：它们不是本轮 bug 的必要修复，全面改动会扩大公开扩展接口和算子回归范围。逐区域 GPU 评分同步、一般候选评分的重复传播仍是后续优化。反序列化入口、用户回调边界、分配后提交前检查，以及 shape/stride/原坐标验证继续保留。

支持范围限于可描述的注册对象与普通容器，不承诺解析任意 Python 对象/闭包里的隐藏引用。新增测试位于 `tests/test_review_followup.py`。最终 PyTorch 2.6 CPU / 2.14 CPU 均 **230 passed、111 CUDA skipped**；RTX 5070 Ti / PyTorch 2.14 CUDA 使用 `--require-cuda`，**341 passed、零跳过**。
