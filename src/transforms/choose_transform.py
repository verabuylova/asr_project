import random
from torchvision.transforms.v2 import Compose


class ChooseTransform:

    def __init__(self, transforms: Compose, p: int):
        self.transforms = transforms
        self.p = p

    def __call__(self, x):
        if random.random() <= self.p:
            return self.transforms(x)
        return x
