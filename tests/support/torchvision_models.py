"""Small torchvision models for executable workflow tests."""

from torch import nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock, ResNet
from torchvision.models.vision_transformer import VisionTransformer


def workflow_resnet() -> ResNet:
    """Build a narrow ResNet with real residual blocks and 1,000 output classes."""
    # Keep torchvision's residual forward and a nonuniform block layout.
    # Separate numerical tests exercise full-sized official architectures.
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False)
    model.bn1 = nn.BatchNorm2d(32)
    for stage, count in enumerate((2, 1, 1, 1), start=1):
        setattr(model, f"layer{stage}", nn.Sequential(*(BasicBlock(32, 32) for _ in range(count))))
    model.fc = nn.Linear(32, 1000)
    return model


def tiny_vit(image_size: int = 8) -> VisionTransformer:
    """Build a two-block ViT with nonzero classifier weights for numerical tests."""
    model = VisionTransformer(
        image_size=image_size,
        patch_size=image_size // 2,
        num_layers=2,
        num_heads=3,
        hidden_dim=12,
        mlp_dim=24,
        num_classes=5,
    )
    # Keep the full-size workflow input but avoid a huge protected patch kernel
    # dominating the tiny fixture's parameter budget. The token grid is still 2x2.
    if image_size == 224:
        model.conv_proj = nn.Conv2d(3, 12, kernel_size=4, stride=112)
    # Torchvision initializes the classifier to zero; nonzero weights make
    # comparisons sensitive to changes in the encoder.
    nn.init.normal_(model.heads.head.weight, std=0.1)
    return model
