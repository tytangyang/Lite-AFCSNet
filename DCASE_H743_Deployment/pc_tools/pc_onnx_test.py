#!/usr/bin/env python3
"""PC 端对照实验: 用同一个 int8 QDQ ONNX 在 PC 上分类 10 个官方样本.

预处理精确复现客户 DACES_T1_1.py:
  - 32kHz 重采样 (scipy resample_poly)
  - torch.stft: n_fft=4096, win=3072(hann periodic), hop=500, center reflect, power=2
  - HTK mel filterbank 256 (norm=None, 与 torchaudio 默认 MelScale 一致)
  - log(mel + 1e-5)
对比: PC 端模型输出 vs 板端输出, 判断问题在模型还是板端链路.
用法: python pc_onnx_test.py
"""
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

ONNX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "asc_base_p0_65_int8_qdq.onnx")
SAMPLES = os.path.join(os.path.dirname(__file__), "dataset", "samples")

SR, N_FFT, WIN, HOP, N_MELS = 32000, 4096, 3072, 500, 256
CLASSES = ["airport", "bus", "metro", "metro_station", "park",
           "public_square", "shopping_mall", "street_pedestrian",
           "street_traffic", "tram"]


def load_audio_32k(path, n_samples):
    """读任意 wav -> 32kHz float32 [-1,1], 循环补齐/截断到 n_samples."""
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        scale = 2147483648.0 if np.abs(data).max() > (1 << 24) else 8388608.0
        x = data.astype(np.float64) / scale
    else:
        x = data.astype(np.float64)
    if rate != SR:
        x = resample_poly(x, SR, rate)
    if len(x) < n_samples:
        x = np.tile(x, int(np.ceil(n_samples / len(x))))[:n_samples]
    else:
        x = x[:n_samples]
    return x.astype(np.float32)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank_htk(n_mels, f_min, f_max, n_stft, sr):
    """torchaudio MelScale 默认 (htk, norm=None) 的三角滤波器组."""
    m_min, m_max = hz_to_mel(f_min), hz_to_mel(f_max)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    freqs = np.linspace(0.0, sr / 2.0, n_stft)
    fb = np.zeros((n_mels, n_stft))
    for k in range(n_mels):
        lo, ctr, hi = f_pts[k], f_pts[k + 1], f_pts[k + 2]
        up = (freqs >= lo) & (freqs <= ctr)
        dn = (freqs >= ctr) & (freqs <= hi)
        if ctr > lo:
            fb[k, up] = (freqs[up] - lo) / (ctr - lo)
        if hi > ctr:
            fb[k, dn] = (hi - freqs[dn]) / (hi - ctr)
    return fb


def logmel_torch(x, n_frames=None):
    """x: float32 [1, samples] (32k) -> log-mel [1,1,256,T]
    板端语义 (n_frames=65, samples=35072): 截取前 65 帧, 与 logmel.c 一致."""
    w = torch.hann_window(WIN, periodic=True)
    spec = torch.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                      window=w, center=True, pad_mode="reflect",
                      normalized=False, onesided=True,
                      return_complex=True)  # [B, 2049, T] complex
    if n_frames is not None:
        spec = spec[..., :n_frames]
    spec = spec.abs() ** 2.0
    fb = torch.from_numpy(mel_filterbank_htk(N_MELS, 0.0, SR / 2.0, N_FFT // 2 + 1, SR)).float()
    mel = torch.einsum("mF,bFT->bmT", fb, spec)
    return (mel + 1e-5).log().unsqueeze(1)           # [B,1,256,T]


def logmel_board(x, centered):
    """精确模拟板端 logmel.c 的 DSP (帧窗口 + rfft + HTK mel + log).

    x: float32 [samples] (32k)
    centered=False: 当前板端实现, 帧窗口 = [t*hop, t*hop+win)  (无 center 对齐)
    centered=True:  对齐 torch.stft(center=True), 帧窗口 = [t*hop-2048, t*hop-2048+win),
                    越界用 reflect 填充 (src<0 -> -src, src>=N -> 2N-1-src)
    """
    w = torch.hann_window(WIN, periodic=True).numpy().astype(np.float32)
    fb = mel_filterbank_htk(N_MELS, 0.0, SR / 2.0, N_FFT // 2 + 1, SR).astype(np.float32)
    n = len(x)
    n_frames = 65
    mel = np.zeros((N_MELS, n_frames), dtype=np.float32)
    for t in range(n_frames):
        off = t * HOP - (N_FFT // 2 if centered else 0)
        seg = np.zeros(WIN, dtype=np.float32)
        for i in range(WIN):
            src = off + i
            if src < 0:
                src = -src
            elif src >= n:
                src = 2 * n - 2 - src
            seg[i] = x[src]
        spec = np.fft.rfft(seg * w, N_FFT)
        mel[:, t] = fb @ (np.abs(spec) ** 2)
    return (np.log(mel + 1e-5)).reshape(1, 1, N_MELS, n_frames)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=32000,
                    help="输入样本数: 32000=客户 pipeline 语义(65帧), 35072=板端语义")
    ap.add_argument("--board-sim", choices=["off", "raw", "centered"], default="off",
                    help="off=torch pipeline; raw=模拟当前板端; centered=模拟修复后板端")
    ap.add_argument("--q16", action="store_true",
                    help="模拟 int16 PCM 量化 (板端收到的是 int16)")
    args = ap.parse_args()

    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    # 板端语义: 35072 样本 -> 71 帧, 模型固定 65 帧, 取前 65 帧
    n_frames = 65 if args.samples == 35072 else None
    print(f"ONNX: {os.path.basename(ONNX)}  samples={args.samples}  frames={n_frames or 'auto'}  board_sim={args.board_sim}  q16={args.q16}")
    print(f"{'scene':18s} {'PC_pred':>14s}  {'softmax 分布 (前3)':42s}")
    for c in CLASSES:
        pcm = load_audio_32k(os.path.join(SAMPLES, f"{c}.wav"), args.samples)
        if args.q16:
            # 与 dcase_batch_test.py / 板端一致: 量化到 int16 再还原 [-1,1]
            pcm = (np.clip(np.round(pcm * 32768.0), -32768, 32767).astype(np.int16)
                   .astype(np.float32) / 32768.0)
        x = torch.from_numpy(pcm).unsqueeze(0)       # [1, samples]
        if args.board_sim != "off":
            lm = logmel_board(pcm, args.board_sim == "centered")
        else:
            lm = logmel_torch(x, n_frames).numpy()    # [1,1,256,T]
        logits = sess.run(None, {in_name: lm})[0][0]  # [10]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        pred = int(probs.argmax())
        top3 = " ".join(f"{CLASSES[i]}:{probs[i]:.2f}" for i in np.argsort(probs)[::-1][:3])
        mark = "OK" if pred == CLASSES.index(c) else "X"
        print(f"{c:18s} {CLASSES[pred]:>14s} {mark}  {top3}")


if __name__ == "__main__":
    main()
