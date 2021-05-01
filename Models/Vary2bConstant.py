from torch import nn
from efficientnet_pytorch import EfficientNet


class MetaArch(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_feature = EfficientNet.from_pretrained('efficientnet-b4').eval()
