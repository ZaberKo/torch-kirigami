# 测试与兼容性

项目使用根目录 `torch_kirigami/` 的 flat layout，安装后可直接导入，不需要设置 `PYTHONPATH`。Ruff、pytest 和打包设置统一位于 `pyproject.toml`；开发约定见根目录 `AGENTS.md`。

测试导航：[逐对象契约与组合清单](testing-coverage.md) · [全部内置入口](operator-test-coverage.md)。默认 pytest 自动发现全部职责目录，根级 `conftest.py` 保留统一设备 fixture 和 `--require-cuda` 验收入口。

| 目录 | 职责 |
| --- | --- |
| `tests/core/` | 坐标、区域、关系、约束、基础记录及有界穷举 |
| `tests/capture/` | 捕获、绑定、来源、hooks、状态隔离与恢复 |
| `tests/graph/` | 查询、闭包、来源解释、局部屏障、过期与分析限制 |
| `tests/operators/` | 按算子家族分组，另逐项检查全部注册入口 |
| `tests/pruning/` | 候选、评分、预算、规划、配方、执行报告及应用 |
| `tests/sparsity/` | 标量正则与梯度、结构组、累计预算、调度、门控及参数操作 |
| `tests/persistence/` | 静态计划、checkpoint、损坏输入、状态及跨进程恢复 |
| `tests/integration/` | 坐标链、共享约束、attention、策略与图、扩展及完整生命周期 |
| `tests/architecture/` | 导入方向、公开契约、扩展边界、覆盖清单完整性 |
| `tests/support/` | 少量确实共享的模型、坐标断言、独立数值参考和原生调用样例 |

同一职责的多个类共用目录，文件按行为划分。一个回归只保留一份，可以在覆盖清单中关联多个类；测试文件不互相导入。普通边界留在所属目录，跨组件组合放在 integration。新增对象或注册入口时，应同时更新契约清单和实际测试；清单校验只检查映射完整性，不能替代数值或错误路径断言。

跨层修复先检查测试有没有经过出错的公共流程：手工 OperationContext 的 analyze 测试不能代替 build/plan/apply；配置记录往返不能代替原模型 forward 与 checkpoint 恢复。每类修复保留最小失败复现，并加入相邻合法写法、拒绝后的状态检查及适用的独立数值/梯度参考。捕获等价检查同时比较实际绑定和值；分配事实由捕获与执行共同读取；静态计划查询覆盖已知空影响与未知引用；配置缓存验证成功、失败和失效路径。具体矩阵见 [跨层契约加固](testing-coverage.md#跨层契约加固)，不将注册入口数解释为完整组合覆盖率。

默认开发环境通过 uv.lock 固定 Python 3.12 对应的 PyTorch 2.14 CPU 和工具依赖：

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv build
```

最低兼容组合单独运行，不改动开发环境及锁文件：

```bash
uv run --isolated --no-project --python 3.10 \
  --with 'torch==2.6.0+cpu' --with pytest --with numpy \
  --index https://download.pytorch.org/whl/cpu python -m pytest
```

测试使用独立坐标集合检查区间/区域运算，使用小型稠密 mask 检查 reshape 索引映射。grouped convolution、attention 的紧凑模型由手工参考构造；GroupNorm 用保留域重新计算均值/方差，避免把置零误当作宽度变化的数值参考。

常规 CI 检查最低组合及最新开发组合。发布前运行 Compatibility 工作流，对 PyTorch 2.7–2.13 各次版本执行全套测试。工作流配置不意味着远端 CI 已实际运行，本地已验证结果以任务执行记录为准。安装流程参考 [uv 官方 GitHub Actions 指南](https://docs.astral.sh/uv/guides/integration/github/)。

自定义扩展应提供：输入/输出及权重对应的独立参考、正反向传播、边界/错误输入、共享调用，以及根模块和嵌套模块测试。规则成功不应只由 forward 未报错来验收。

CUDA 验收使用独立环境，明确指定 CUDA wheel，避免复用版本号相同的 CPU 构建：

```bash
uv run --isolated --no-project --python 3.12 \
  --with 'torch==2.14.0+cu130' --with pytest --with numpy \
  --index https://download.pytorch.org/whl/cu130 \
  python -m pytest --require-cuda
```

`--require-cuda` 在 CUDA 不可用时直接报错。`execution_device` fixture 将核心模型和独立数值参考分别放到 CPU、CUDA 上运行；可加 `-k cuda` 单独运行 GPU 用例。覆盖链式/共享/残差依赖、分组与深度卷积、归一化、矩阵乘、切片组合、attention，以及成功和异常路径上的输入、buffer、随机数状态恢复。普通无 GPU 的 CI 会明确跳过 CUDA 用例。

本地已在 WSL 的 RTX 5070 Ti、Python 3.12、PyTorch 2.14.0+cu130 环境完成 GPU 验收。PyTorch 2.6 的已验证范围仍为 CPU；这些结果不证明其他 GPU 架构或自定义 CUDA kernel 的行为。若工具沙箱阻断 GPU 访问，应在具备设备访问权限的执行环境重试，不能据此断言主机没有 GPU。动态 Python 分支测试必须明确捕获失败。


剪枝回归另覆盖计划只读、默认输入输出保护、预算分母、初始无效的 Divisible、分组补全、分段权重行列切片、四种评分公式、低精度累积、原 forward 尺寸/切片/stride 验证、分配及提交失败、别名保持、过期计划和 inference mode 下应用后 backward。数值参考覆盖 Linear/Conv、LN/GN/BN/softmax、矩阵乘与残差循环。

```bash
uv run --locked python examples/pruning.py
# 在上述独立 CUDA 环境使用相同 uv 参数运行：
# python -m examples.pruning --device cuda
```

`examples/pruning.py` 显式展示重新建图、重建 optimizer 和调用方采集 Taylor 梯度；不会由剪枝层代管训练。


`CUDA validation` 工作流提供相同强制验收，需先配置带 `self-hosted/linux/x64/cuda` 标签的可信 GPU runner，再手工触发；这里没有安装 runner 或执行远端工作流。独立 `--no-project` 环境从仓库根目录用 `python -m examples.pruning` 启动示例，避免直接执行脚本时找不到未安装的本地包。

## 2026-09-10 完整验收

以下为 2026-09-10 跨层契约修复与补测后的已执行记录，不代表后续修改已自动获得验收。

| 实际本地环境 | 全量结果 |
| --- | --- |
| Python 3.10 / PyTorch 2.6.0+cpu | 1043 passed，785 skipped |
| Python 3.12 / PyTorch 2.14.0+cpu | 1043 passed，785 skipped |
| Python 3.12 / PyTorch 2.14.0+cu130 / RTX 5070 Ti | 1828 passed，0 skipped，使用 --require-cuda |

CPU 跳过项包括需要 CUDA 的测试和 Tensor.cuda 的 CPU 源项，均由真实 GPU 验收覆盖。GPU 执行有一条 PyTorch cuBLAS 首次 backward 建立主上下文的 warning，没有失败或跳过。Ruff 全仓库检查、格式检查和 git diff 空白检查通过。

测试包含 66 个测试文件、254 个函数。逐对象清单记录 78 个公开导出或内部组件，逐入口清单记录 351 个注册入口（模块 78、函数 172、方法 101）。覆盖清单与测试之间的引用由架构测试检查。

这些结果证明列示契约和选定组合，不能保证任意模型或全部轴、尺寸、dtype、stride、训练态组合。PyTorch 中间版本和其他设备不属于此次实际验收环境。

本轮另通过四个 CPU 示例、wheel/sdist 构建及独立临时环境安装验证。安装验证从 site-packages 导入，执行 torch.Size Unflatten 配置的静态 plan 往返、原图引用查询、inference mode 内物理剪枝、checkpoint 恢复和普通模式 backward；临时环境退出后清理。

移除 plan、checkpoint 和模型内结构记录的格式版本后，另执行保存恢复、生命周期、配置和引用查询回归：PyTorch 2.6/2.14 CPU 各 116 passed、104 skipped；2.14 CUDA 使用 `--require-cuda`，220 passed、零跳过。Ruff 检查通过。

## 2026-09-11 通用稀疏组件验收

共享 CandidateSpace 接口、三类标量正则、整数及累计预算、门控、参数操作和七类算法示例完成后，执行以下全量回归：

| 实际本地环境 | 全量结果 |
| --- | --- |
| Python 3.10 / PyTorch 2.6.0+cpu | 1077 passed，793 skipped |
| Python 3.12 / PyTorch 2.14.0+cpu | 1077 passed，793 skipped |
| Python 3.12 / PyTorch 2.14.0+cu130 | 1870 passed，0 skipped，使用 --require-cuda |

随后仅新增两个边界回归，分别检查未知影响路径拒绝构造参数组及合法替代路径，以及共享门控参数别名去重。新增测试后的 `tests/sparsity/` 在两个 CPU 环境各为 24 passed、6 skipped，在 CUDA 环境为 30 passed、零跳过。上述全量数字不包含最后新增的两个测试。

新增回归涵盖独立正则公式与 gradcheck、SGD/AdamW、梯度累积及 AMP、失败操作不改变状态、累计预算欠达与恢复，以及公开 build/plan/apply 后继续 backward 和 checkpoint 往返。`tests/integration/test_sparse_workflows.py` 同时执行七类算法的九种默认离线命令组合；门控 Transformer 另有命令行冒烟检查。玩具任务输出只用于验证流程，不作为论文实验或模型性能结果。

Ruff 全仓库检查、格式检查和 git diff 空白检查通过。CUDA 全量执行出现一条已有的 cuBLAS 上下文初始化 warning。未执行 PyTorch 2.6 CUDA、中间版本矩阵或远端 CI；此前的构建和安装记录不代表本次重新执行了打包验收。

## 2026-09-11 审查后的六项修复验收

修复 checkpoint 临时模块复制导致原模型损坏、共享存储参数更新覆盖、累计预算丢失新受保护域、不同 mask 的共享门控评分、软剪枝无法重新生长和极小非零参数范数投影六项问题后，重新执行全量回归：

| 实际本地环境 | 全量结果 |
| --- | --- |
| Python 3.10 / PyTorch 2.6.0+cpu | 1100 passed，809 skipped |
| Python 3.12 / PyTorch 2.14.0+cpu | 1100 passed，809 skipped |
| Python 3.12 / PyTorch 2.14.0+cu130 | 1909 passed，0 skipped，使用 --require-cuda |

新增回归检查操作拒绝前后的参数值、身份、梯度和版本计数；checkpoint 自定义复制方法被绕过及加载失败隔离；共享门控物理剪枝后的独立输出参考；float16/float32/float64 次正规数投影；已受保护初始域的累计预算恢复；保留动量后的软剪枝恢复、收缩及 checkpoint。离线 CLI 矩阵另加入 Transformer 迭代剪枝的默认/0.9 比例与 MLP 的 0.99 比例。

Ruff 检查、格式检查和 git diff 空白检查通过。CUDA 只有已有的 cuBLAS 首次上下文初始化 warning。此次未运行 PyTorch 2.6 CUDA、中间版本、远端 CI 或打包安装验收。

## 2026-09-11 MACs 与延迟测量验收

新增通用测量工具并接入七类 workflow 后，PyTorch 2.14 CPU 全量回归为 **1115 passed、817 skipped**。最低支持组合 Python 3.10 / PyTorch 2.6 CPU 的测量及 workflow 回归为 **32 passed、12 skipped**；PyTorch 2.14 CUDA 同一组回归使用 --require-cuda，为 **44 passed、零跳过**。

实际运行默认 Inductor 编译的 basic/Taylor CPU 示例，以及 gated/Transformer CUDA 示例，均完成前后 MACs、参数量、延迟输出和 checkpoint 验证。前者在 batch=1 下参数量为 147→75、MACs 为 132→66；后者参数量为 671→369、MACs 为 2840→1432。短采样延迟受系统负载影响，此记录不宣称稳定加速。CUDA 示例独立环境未安装 NumPy，PyTorch 有一次 NumPy 初始化提示，程序成功完成。

测试验证编译不向原模型附加导致图失效的属性、预热/编译不进入计时、计数覆盖 attention 两次矩阵乘、冻结参数计数、计量遗漏报告和状态恢复。本轮未重新执行最低版本或 CUDA 的全量矩阵；Ruff 全仓库检查、格式及差异检查通过，无新增运行时依赖。

## 2026-09-11 预训练示例与目录调整验收

workflow 统一使用普通 `.py`，soft 的入口与优化器辅助函数也合并在 `soft_pruning.py` 中。逐任务说明移至 [examples/workflows/README.md](../examples/workflows/README.md)。新增 原 ResNet 专用入口（现已并入各 workflow） 使用官方 ResNet-18 权重和 HF ImageNet-1K；默认仅验证集与直接剪枝，训练集按需加载。

| 实际环境与范围 | 结果 |
| --- | --- |
| Python 3.12 / PyTorch 2.14 CPU：原 workflow、measurement、architecture | 53 passed，12 skipped |
| Python 3.10 / PyTorch 2.6 CPU / torchvision 0.21：新增预训练示例测试 | 11 passed，6 skipped |
| Python 3.12 / PyTorch 2.14 CUDA / torchvision 0.29：新增测试，--require-cuda | 17 passed，零跳过 |

示例依赖从 `requirements-examples.txt` 安装到独立环境，测试使用 datasets 4.8.5。新增测试离线覆盖默认不训练、按需读取训练集、ImageNet 标签顺序/别名、分批精度统计、模式恢复、五种评分/正则模式的两轮剪枝与微调，以及独立 float64 ResNet BN 置零参考、剪后 backward 和 checkpoint。float64 参考避免不同卷积宽度选择低精度 CUDA 内核时的舍入差异；CUDA 入口测试采用与 CLI 一致的 CPU 数据加载和显式设备传输。

另实际下载 `ResNet18_Weights.IMAGENET1K_V1`，通过 HF 公开元数据核对全部 1000 类，在 CPU 上验证真实权重的稠密置零参考，并执行默认 Inductor 前后测量。单独剪去 `layer1.0` 的 16/64 个内部通道时，参数量 11,689,512→11,671,048，已计 MACs 1,814,073,344→1,756,270,592；未计操作随输出报告。这只是短采样运行检查，不作为性能优势结论。

**未测完整 ImageNet accuracy**：当前 HF 会话访问 `ILSVRC/imagenet-1k` 返回 gated dataset 错误，需用户在数据集页面接受访问条件并登录。未用离线测试的合成 accuracy 替代真实结果。本轮未重新执行全仓库 pytest、完整版本矩阵或 CUDA Inductor 预训练模型测量；Ruff 全仓库检查、格式检查、差异空白检查、示例文档链接检查和单文件基础入口运行通过。

## 2026-09-11 七类 workflow 统一预训练模型验收

七类入口统一使用官方预训练 ResNet-18 或 ViT-B/16，读取预先下载的本地 ImageNet Parquet；原 ResNet 专用入口合并进共享组件，小模型夹具移至 tests/support。根目录 `.venv` 安装示例依赖后，七个脚本均通过从 workflow 目录直接执行的 `--help` 检查。

| 实际环境与范围 | 结果 |
| --- | --- |
| Python 3.12 / PyTorch 2.14 CPU / torchvision 0.29 / datasets 5.0.1：全仓库 | 1135 passed，842 skipped |
| Python 3.10 / PyTorch 2.6 CPU / torchvision 0.21 / datasets 4.8.5：预训练 workflow 与门控操作 | 47 passed，38 skipped |
| Python 3.12 / PyTorch 2.14 CUDA / torchvision 0.29：相同相关测试，--require-cuda | 85 passed，零跳过 |

最低版本环境显式选择 datasets 4.8.5 验证，示例 requirements 保持不固定版本。新增测试覆盖 CNN/ViT 各算法阶段、独立训练与验证数据、Taylor 校准不改变 BN 统计量、门控后的原地 ReLU 合法剪枝，以及多消费者原地操作拒绝时状态不变。

另实际加载官方完整预训练 ResNet 和 ViT 权重，验证 ViT 适配前向与官方前向一致、初始门控恒等，以及屏蔽 FFN 维度和物理收缩的数值一致性。两个真实预训练模型收缩后均完成 CPU 默认 Inductor 的 MACs/latency 测量。Ruff 检查、格式检查和差异空白检查通过。

本轮没有完整 ImageNet 精度结果；真实精度需用下载后的数据执行 workflow。没有重新运行最低版本或 CUDA 全仓库矩阵，也未运行真实预训练模型的 CUDA Inductor 测量。
