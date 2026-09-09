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
| `tests/persistence/` | 静态计划、checkpoint、损坏输入、状态及跨进程恢复 |
| `tests/integration/` | 坐标链、共享约束、attention、策略与图、扩展及完整生命周期 |
| `tests/architecture/` | 导入方向、公开契约、扩展边界、覆盖清单完整性 |
| `tests/support/` | 少量确实共享的模型、坐标断言、独立数值参考和原生调用样例 |

同一职责的多个类共用目录，文件按行为划分。一个回归只保留一份，可以在覆盖清单中关联多个类；测试文件不互相导入。普通边界留在所属目录，跨组件组合放在 integration。新增对象或注册入口时，应同时更新契约清单和实际测试；清单校验只检查映射完整性，不能替代数值或错误路径断言。

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

## 最近一次完整验收

以下为 2026-09-10 测试分层及补测后的已执行记录，不代表后续修改已自动获得验收。

| 实际本地环境 | 全量结果 |
| --- | --- |
| Python 3.10 / PyTorch 2.6.0+cpu | 925 passed，675 skipped |
| Python 3.12 / PyTorch 2.14.0+cpu | 925 passed，675 skipped |
| Python 3.12 / PyTorch 2.14.0+cu130 / RTX 5070 Ti | 1600 passed，0 skipped，使用 --require-cuda |

CPU 跳过项包括需要 CUDA 的测试和 Tensor.cuda 的 CPU 源项，均由真实 GPU 验收覆盖。GPU 执行有一条 PyTorch cuBLAS 首次 backward 建立主上下文的 warning，没有失败或跳过。Ruff 全仓库检查、格式检查和 git diff 空白检查通过。

测试包含 59 个测试文件、237 个函数。逐对象清单记录 77 个公开导出或内部组件，逐入口清单记录 339 个注册入口（模块 78、函数 166、方法 95）。覆盖清单与测试之间的引用由架构测试检查。

这些结果证明列示契约和选定组合，不能保证任意模型或全部轴、尺寸、dtype、stride、训练态组合。PyTorch 中间版本和其他设备不属于此次实际验收环境。
