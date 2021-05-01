from torch import nn


class LongTermMemory(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class ShortTermMemory(nn.Module):
    def __init__(self):
        super().__init__()


class Attention(nn.Module):
    def __init__(self):
        super().__init__()


class MetaArch(nn.Module):
    def __init__(self):
        super().__init__()
        self.ltm = LongTermMemory()
        self.stm = ShortTermMemory()
        self.att = Attention()

    def forward(self, img):
        obj_model = self.ltm(img)
        
