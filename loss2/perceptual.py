import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class _VGGFeatureExtractor(nn.Module):
    """
    Freezes VGG-16 and exposes the four relu activations that are commonly used
    for perceptual similarity (relu1_1, relu1_2, relu2_1, relu2_2).
    Output is a list of four feature maps, one per layer.
    """

    # Indices inside vgg16.features that correspond to each relu
    _RELU_INDICES = {"1", "3", "6", "8"}

    def __init__(self):
        super().__init__()
        backbone = vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = backbone.features  # Conv + ReLU blocks

        # Freeze — perceptual loss is a fixed metric, not a learned one
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return activations at the four selected relu layers."""
        outputs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self._RELU_INDICES:
                outputs.append(x)
        return outputs  # len == 4


# ---------------------------------------------------------------------------
# Public class consumed by train.py
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss between two images computed as the mean MSE across four
    VGG-16 relu feature maps.

    Usage (train.py):
        vgg_model = VGGPerceptualLoss().to(device)
        vae_loss_fn = VAELoss(perceptual_model=vgg_model).to(device)
    """

    def __init__(self):
        super().__init__()
        self.extractor = _VGGFeatureExtractor()
        self.mse = nn.MSELoss()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_hat : (B, C, H, W) – model output, values in [0, 1] or [-1, 1]
            y     : (B, C, H, W) – ground truth, same range

        Returns:
            Scalar perceptual loss.
        """
        feats_hat = self.extractor(y_hat)
        feats_y   = self.extractor(y)

        loss = sum(
            self.mse(fh, fy) for fh, fy in zip(feats_hat, feats_y)
        ) / len(feats_hat)

        return loss