import torch
from torch import nn
import torchaudio.transforms as T

# Spectrogram Augmentations
class AddNoiseSpec(nn.Module):
    def __init__(self, noise_level=0.03, probability=0.5):
        super(AddNoiseSpec, self).__init__()
        self.noise_level = noise_level
        self.probability = probability

    def forward(self, spectrogram):
        if torch.rand(1).item() < self.probability:
            noise = torch.randn_like(spectrogram) * self.noise_level
            spectrogram = spectrogram + noise
        return spectrogram

class TimeMaskSpec(nn.Module):
    def __init__(self, time_mask_param=10, probability=0.5):
        super(TimeMaskSpec, self).__init__()
        self.time_mask = T.TimeMasking(time_mask_param)
        self.probability = probability

    def forward(self, spectrogram):
        if torch.rand(1).item() < self.probability:
            spectrogram = self.time_mask(spectrogram)
        return spectrogram

class FrequencyMaskSpec(nn.Module):
    def __init__(self, freq_mask_param=10, probability=0.5):
        super(FrequencyMaskSpec, self).__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.probability = probability

    def forward(self, spectrogram):
        if torch.rand(1).item() < self.probability:
            spectrogram = self.freq_mask(spectrogram)
        return spectrogram

class GainSpec(nn.Module):
    def __init__(self, gain_factor=[0.8, 1.2], probability=0.5):
        super(GainSpec, self).__init__()
        self.gain_factor = gain_factor
        self.probability = probability

    def forward(self, spectrogram):
        if torch.rand(1).item() < self.probability:
            gain = torch.empty(1).uniform_(*self.gain_factor).item()
            spectrogram = spectrogram * gain
        return spectrogram

class ClippingDistortionSpec(nn.Module):
    def __init__(self, clip_threshold=0.2, probability=0.5):
        super(ClippingDistortionSpec, self).__init__()
        self.clip_threshold = clip_threshold
        self.probability = probability

    def forward(self, spectrogram):
        if torch.rand(1).item() < self.probability:
            spectrogram = torch.clamp(spectrogram, -self.clip_threshold, self.clip_threshold)
        return spectrogram

class NormalizeSpec(nn.Module):
    def __init__(self):
        super(NormalizeSpec, self).__init__()

    def forward(self, spectrogram):
        mean = spectrogram.mean()
        std = spectrogram.std()
        spectrogram = (spectrogram - mean) / (std + 1e-9)
        return spectrogram

class LogarithmicScaleSpec(nn.Module):
    def __init__(self, eps=1e-9):
        super(LogarithmicScaleSpec, self).__init__()
        self.eps = eps

    def forward(self, spectrogram):
        spectrogram = torch.clamp(spectrogram, min=self.eps)
        return torch.log10(spectrogram)
