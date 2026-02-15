import torch
import torch.nn.functional as F

class SoftThickeningDigit(torch.nn.Module):
    def __init__(self, kernel_size=3, alpha=0.5):
        
        super().__init__()
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.padding = kernel_size // 2

    def forward(self, img):

        thick_img = F.max_pool2d(img, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        soft_img = (1 - self.alpha) * img + self.alpha * thick_img
        
        return soft_img
        
class Clamp(torch.nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clamp(img, self.min_val, self.max_val)
        
class AddSaltPepperNoise(torch.nn.Module):
    def __init__(self, probability=0.05):
        super().__init__()
        self.probability = probability

    def forward(self, img):

        noise_tensor = torch.rand_like(img)

        salt_mask = noise_tensor < (self.probability / 2)
        pepper_mask = noise_tensor > (1 - self.probability / 2)
        
        img[salt_mask] = 1.0
        img[pepper_mask] = 0.0
        return img

class ThickeningDigit(torch.nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, img):
        padding = self.kernel_size // 2
        img = F.max_pool2d(img, kernel_size=self.kernel_size, stride=1, padding=padding)
        return img