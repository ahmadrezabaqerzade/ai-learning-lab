from torch import nn
from augmentation import AddNoiseSpec, TimeMaskSpec, FrequencyMaskSpec, GainSpec, ClippingDistortionSpec, LogarithmicScaleSpec, NormalizeSpec

def get_transform_train_valid(add_noise_prob=0.5, noise_level=0.01, time_mask_param=10, time_mask_prob = 0.5,
                              freq_mask_param=10, freq_mask_prob = 0.5, gain_factor=[0.8, 1.2], gain_prob = 0.5,
                              clip_threshold=0.2, clipping_distortion_prob = 0.5):

  train_transform = nn.Sequential(AddNoiseSpec(noise_level = noise_level, probability = add_noise_prob),
                                  TimeMaskSpec(time_mask_param = time_mask_param, probability = time_mask_prob),
                                  FrequencyMaskSpec(freq_mask_param=freq_mask_param, probability=freq_mask_prob),
                                  GainSpec(gain_factor = gain_factor, probability=gain_prob),
                                  ClippingDistortionSpec(clip_threshold = clip_threshold, probability=clipping_distortion_prob),
                                  LogarithmicScaleSpec(),
                                  NormalizeSpec())
  return train_transform