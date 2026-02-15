import torchvision.transforms.v2 as transforms
import torch
import DataProcessorer as dp

train_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),

    transforms.RandomApply(
        [dp.SoftThickeningDigit(kernel_size=3, alpha=0.25)],
        p=0.08
    ),

    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3))],
        p=0.07
    ),

    transforms.RandomAffine(
        degrees=8,
        translate=(0.04, 0.04),
        scale=(0.92, 1.08),
        fill=0
    ),

    transforms.RandomApply(
        [dp.AddSaltPepperNoise(probability=0.015)],
        p=0.03
    ),

    dp.Clamp(0.0, 1.0),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.1307,), (0.3081,))
])