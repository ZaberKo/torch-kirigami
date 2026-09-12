# ImageNet 剪枝流程

## 环境与数据

在仓库根目录配置环境，最后进入本目录：

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r examples/workflows/requirements-examples.txt
cd examples/workflows
```

所有流程使用 torchvision 的 **ImageNet 预训练权重**，模型选项为 `--model resnet18`（CNN，默认）和 `--model vit_b_16`（Transformer）。数据来自 [HF ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)，先下载 validation：

```bash
hf download ILSVRC/imagenet-1k --repo-type dataset \
  --include 'data/validation-*.parquet'
```

需要稀疏训练、微调或 Taylor 梯度校准时，再下载 train：

```bash
hf download ILSVRC/imagenet-1k --repo-type dataset \
  --include 'data/train-*.parquet'
```

下载使用 HF 默认缓存；脚本自动从相同缓存读取，不在线下载数据。Parquet 包含图片与标签，无需另外下载图片压缩包或标签文件；仅评估下载 validation，需要训练时再加 train，不需要 test。模型权重由 torchvision 首次运行时单独下载。

已有其他数据目录可用 `--data-dir 路径` 指定。默认评估完整 valset；`--val-samples 256` 可先运行固定子集。训练默认每轮取 512 张，`--train-samples 0` 使用完整 train split，训练与验证互不混用。样本数选项控制读取量，不减少上述命令的下载量。

CNN 剪 BasicBlock 内部通道，ViT 剪 FFN 中间维度，保留原来的 1000 类分类头。默认剪第一个 block；`--layers all` 覆盖全部 block。默认每 8 个相邻通道为一组，`--group-size 1` 可逐通道选择。ViT 使用保留官方权重与计算的显式前向适配，便于依赖分析。

各阶段打印 top-1/top-5、相对预训练模型的精度变化、loss、#Params、#MACs 和 latency；剪枝后另有目标/实际比例。加 `--device cuda` 使用 GPU，`--compile` 测量编译推理延迟。`--output runs/名称` 指定指标与 checkpoint 保存目录。

## 1. 基础剪枝 → 可选微调

[prune_finetune.py](prune_finetune.py)：按输出通道的权重幅值或 Taylor 任务梯度评分，物理剪枝并评估精度。默认不训练；开启微调时，分别记录刚剪完和微调后的结果。

```bash
python prune_finetune.py --model resnet18 --ratio 0.25
python prune_finetune.py --model vit_b_16 --ratio 0.25 --finetune-epochs 1
python prune_finetune.py --metric taylor --finetune-epochs 1
```

## 2. 多轮剪枝 ↔ 微调

[iterative_pruning.py](iterative_pruning.py)：每轮重新评分，按初始通道数计算累计预算，扣除实际已删数量，剪后重建依赖图与 optimizer。

```bash
python iterative_pruning.py --model resnet18 --rounds 3 --ratio 0.3 --finetune-epochs 1
```

## 3. BN 稀疏训练 → 剪枝

[bn_sparsity.py](bn_sparsity.py)：对 BN 缩放参数加 L1 正则，再按缩放幅值选组，剪后关闭正则微调。只适用于 ResNet；ViT 使用 LayerNorm。

```bash
python bn_sparsity.py --sparse-epochs 2 --finetune-epochs 1
```

## 4. 依赖组稀疏训练 → 剪枝

[group_sparsity.py](group_sparsity.py)：对完整依赖组使用 Group Lasso；或每轮重新选组，逐步增强平方 L2 惩罚，再物理剪枝。

```bash
python group_sparsity.py --model resnet18 --penalty lasso --finetune-epochs 1
python group_sparsity.py --model vit_b_16 --penalty squared --sparse-epochs 2 --finetune-epochs 1
```

## 5. 软剪枝与平滑衰减

[soft_pruning.py](soft_pruning.py)：先训练建立 SGD 动量，再周期性置零或降低选中参数并集的范数；两个周期之间恢复训练、重新选组，最后物理剪枝。

```bash
python soft_pruning.py --model resnet18 --operation zero --finetune-epochs 1
python soft_pruning.py --model vit_b_16 --operation decay --finetune-epochs 1
```

## 6. 门控训练 → 物理剪枝

[gate_pruning.py](gate_pruning.py)：在预训练模型的 CNN 通道或 ViT FFN 位置接入初始值为一的 gate，进行 L1 训练，按门控幅值剪枝。收缩后保留剩余 gate 值。

```bash
python gate_pruning.py --model resnet18 --sparse-epochs 2 --finetune-epochs 1
python gate_pruning.py --model vit_b_16 --sparse-epochs 2 --finetune-epochs 1
```

## 7. 稳定性驱动剪枝

[stability_pruning.py](stability_pruning.py)：重新选组并增强平方 L2 正则，用保留位置集合的 Jaccard 相似度判断是否稳定；达到阈值或搜索轮数上限后剪枝，再普通微调。

```bash
python stability_pruning.py --model vit_b_16 --sparse-epochs 4 --finetune-epochs 1
```

这些脚本展示典型算法流程，不是论文实验复现。组件公式与方法来源见[稀疏训练文档](../../docs/sparse-training.md)。共享模型、数据、训练和测量代码位于 `imagenet_models.py`、`imagenet_data.py`、`workflow_utils.py` 和 `model_metrics.py`。
