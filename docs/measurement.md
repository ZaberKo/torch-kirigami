# 模型复杂度与推理延迟

`torch_kirigami.measurement` 提供独立于剪枝和训练策略的 CPU/CUDA 测量工具，只有现有的 PyTorch 依赖。没有 NAS、elastic_num_params 或 NPU 逻辑。

```python
from torch_kirigami.measurement import calculate_model_complexity, measure_module_latency

# 模型须已在目标设备；测量不会永久迁移模型或改变参数绑定。
model = model.to("cuda")
complexity = calculate_model_complexity(model, (inputs,), device="cuda")
latency_ms = measure_module_latency(
    model,
    (inputs,),
    device="cuda",
    compile=True,
    warmup=5,
    repetitions=20,
)
print(complexity.macs, complexity.params, complexity.unsupported_ops, latency_ms)
```

两个函数接受 nn.Module、单个输入或位置参数 tuple/list，以及 input_kwargs。输入可嵌套 tuple/list/dict；模型所有参数和 buffer 应在同一 CPU/CUDA 设备，输入会移至该设备。省略 device 时从模型参数/buffer 推断，无参数模型默认为 CPU。

## MACs 与参数量

`calculate_model_complexity` 返回 `ModelComplexity(macs, params, unsupported_ops)`。macs 对应整个输入 batch；一个乘加记为一个 MAC，等于两个 FLOPs。统计矩阵乘、卷积及 attention 的 QK/AV 矩阵乘；不统计 bias 加法、激活、归一化和其他逐元素运算。params 统计全部唯一注册 Parameter 的元素数，包括冻结参数，共享 Parameter 去重，不含 buffer。

实现使用 [PyTorch FlopCounterMode](https://docs.pytorch.org/docs/2.14/generated/torch.utils.flop_counter.FlopCounterMode.html) 的公式和算子 profiler。在 eager 模型上关闭 inference_mode、使用 no_grad，并临时选择 SDPA 数学后端以避免 CPU 融合 attention 绕过计数。延迟测量不使用这个后端限制。MACs 是理论形状计量，不能解释为编译后实际指令数，也不是全部浮点操作总量。

显式支持范围之外的操作通过 unsupported_ops 报告；非空时 MACs 只能视为部分统计，示例不计算 MAC 减少比例。该检查针对实际执行路径，不能证明任意自定义 Python/C++ 代码的运算量。请将 eager 原模型传给复杂度函数；不要传入编译包装器。

## 延迟

`measure_module_latency` 返回整个 batch 单次推理的延迟中位数，单位 ms。CPU 使用 perf_counter；CUDA 在指定设备当前 stream 上使用预先初始化的 events，并在计时前后同步。输入复制/传输、编译、预热和测量状态隔离不计入延迟。

默认 eager；compile=True 时先调用 torch.compile，再执行一次不计时的前向和 warmup 次预热。可通过 compile_kwargs 传入 backend、mode 等原生配置。编译失败会直接报错，不静默退回 eager。每次调用单独建立包装器，剪枝后须再次测量；原生编译缓存仍由 PyTorch 管理。

测量临时使用 eval；成功或异常退出后恢复原来的逐模块训练模式、buffer 原始绑定和值、输入和 torch RNG。延迟函数也恢复 GC 开关状态。函数不隔离任意 Python 副作用或参数写入，模型的推理 forward 应保持参数只读；不要并发训练同一个模型。现有 inference 隔离限制仍适用，例如输入/buffer 与参数共享存储会拒绝。

## Workflow 输出

七类 workflow 默认测量原始模型及最终紧凑模型，打印 #Params、#MACs、latency_ms、输入 shape、dtype、设备、编译状态和计时配置，随后输出参数/MAC 减少比例及延迟加速比。默认推理 batch 为 1，与 ImageNet 训练 batch 独立；CPU 线程数由 --threads 指定。相同配置用于剪枝前后比较。

按 [workflow 说明](../examples/workflows/README.md) 安装环境后，在该目录执行：

```bash
python prune_finetune.py --metric taylor
python gate_pruning.py --model vit_b_16 \
  --device cuda --compile --benchmark-batch-size 1 --warmup 10 --repetitions 50
```

不支持算子会明确列出；减少参数或理论 MACs 不保证降低延迟。短采样延迟会受系统负载影响，不用于宣称论文实验或性能优势。
