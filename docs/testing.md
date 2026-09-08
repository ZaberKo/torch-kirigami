# 测试与兼容性

项目使用根目录 `torch_kirigami/` 的 flat layout，安装后可直接导入，不需要设置 `PYTHONPATH`。Ruff、pytest 和打包设置统一位于 `pyproject.toml`；开发约定见根目录 `AGENTS.md`。

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


第二阶段回归另覆盖计划只读、默认输入输出保护、预算分母、初始无效的 Divisible、分组补全、分段权重行列切片、四种评分公式、低精度累积、原 forward 尺寸/切片/stride 验证、分配及提交失败、别名保持、过期计划和 inference mode 下应用后 backward。数值参考覆盖 Linear/Conv、LN/GN/BN/softmax、矩阵乘与残差循环。

```bash
uv run --locked python examples/pruning.py
# 在上述独立 CUDA 环境使用相同 uv 参数运行：
# python -m examples.pruning --device cuda
```

`examples/pruning.py` 显式展示重新建图、重建 optimizer 和调用方采集 Taylor 梯度；不会由剪枝层代管训练。


`CUDA validation` 工作流提供相同强制验收，需先配置带 `self-hosted/linux/x64/cuda` 标签的可信 GPU runner，再手工触发；这里没有安装 runner 或执行远端工作流。独立 `--no-project` 环境从仓库根目录用 `python -m examples.pruning` 启动示例，避免直接执行脚本时找不到未安装的本地包。

2026-09-08 统一规则、静态计划与持久化改造后的本地验收记录：

| 环境 | 结果 |
| --- | --- |
| Python 3.10 / PyTorch 2.6.0+cpu | 177 passed，87 CUDA skipped |
| Python 3.12 / PyTorch 2.14.0+cpu | 177 passed，87 CUDA skipped |
| Python 3.12 / PyTorch 2.14.0+cu130 / RTX 5070 Ti | 264 passed，0 skipped，使用 `--require-cuda` |

新增回归覆盖尺寸消费者遗漏、同 shape 的归约轴变化、channels-last/view、静态索引值 guard、分组转置卷积、Embedding 非零预算轴、归一化、门控/重排、einsum、GQA 和 MHA 的独立数值参考。

静态计划回归验证重复规划、无签发状态、释放原图、跨进程加载、结构前提和改动配方检查。checkpoint 回归覆盖两轮剪枝后训练保存、参数/模块别名、非持久 buffer、混合模式、原生 extra-state、损坏别名值，以及新进程只依赖模型构造函数恢复。

Ruff 检查、格式检查、四个 CPU 示例和 wheel/sdist 构建通过。仓库外使用一次性虚拟环境直接安装新 wheel，验证剪枝、checkpoint 恢复及 backward，避免同版本 uv run 环境缓存误用旧包。

```bash
uv run --locked python examples/fused_attention.py
```

该融合示例只注册一个 OperatorRule，不注册独立执行或保存规则。测试结果证明这些已声明场景，不代表覆盖任意模型或全部算子参数组合。
