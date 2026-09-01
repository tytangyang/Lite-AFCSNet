#!/usr/bin/env python3
"""检查 dcase_batch_test 转换出的 PCM 特征, 定位"全部判 bus"问题."""
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(__file__))
from dcase_batch_test import load_wav_32k  # noqa: E402

SAMPLES = os.path.join(os.path.dirname(__file__), "dataset", "samples")
CLASSES = ["airport", "bus", "metro", "metro_station", "park",
           "public_square", "shopping_mall", "street_pedestrian",
           "street_traffic", "tram"]

for c in CLASSES:
    p = os.path.join(SAMPLES, f"{c}.wav")
    rate, data = wavfile.read(p)
    raw = data.astype(np.float64)
    pcm = np.frombuffer(load_wav_32k(p), dtype=np.int16).astype(np.float64)
    print(f"{c:18s} src_rate={rate:5d} src_dtype={str(data.dtype):6s} "
          f"src_peak={np.abs(raw).max():9.0f} "
          f"-> 32k RMS={np.sqrt((pcm**2).mean()):8.1f} "
          f"peak={np.abs(pcm).max():7.0f} "
          f"first8={pcm[:8].astype(int).tolist()}")
