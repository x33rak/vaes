from torchvision.models import vgg16
import torch.nn as nn
from torchvision import transforms

def trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=["3", "8", "15", "22"]):
        super().__init__()
        vgg = vgg16(pretrained=True).features
        self.vgg_layers = vgg.eval()
        trainable(self.vgg_layers, False)
        self.layer_ids = set(layers)
        self.criterion = nn.MSELoss()
        self.vgg_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        x, y = self.vgg_norm((x+1)/2), self.vgg_norm((y+1)/2) # get inputs from [-1, 1] -> [0, 1] range and vgg norm
        loss = 0.0
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            y = layer(y)
            if str(i) in self.layer_ids:
                loss += self.criterion(x, y)
        return loss
