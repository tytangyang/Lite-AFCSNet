#!/usr/bin/env python3
"""对比板端 dump 的 logmel 与 PC 模拟, 并分别喂 ONNX 看分类差异.

用法: python check_board_logmel.py [scene]
"""
import os
import sys

import numpy as np
import onnxruntime as ort

ONNX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "asc_base_p0_65_int8_qdq.onnx")
SAMPLES = os.path.join(os.path.dirname(__file__), "dataset", "samples")

SR, N_FFT, WIN, HOP, N_MELS = 32000, 4096, 3072, 500, 256
CLASSES = ["airport", "bus", "metro", "metro_station", "park", "public_square",
           "shopping_mall", "street_pedestrian", "street_traffic", "tram"]


def load_wav_32k(path):
    from scipy.signal import resample_poly
    import scipy.io.wavfile as wavfile
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    scale = 2147483648.0 if np.abs(data).max() > (1 << 24) else 8388608.0
    x = data.astype(np.float64) / scale
    x = resample_poly(x, SR, sr)
    if x.size < 35072:
        reps = int(np.ceil(35072 / x.size))
        x = np.tile(x, reps)
    return x[:35072]


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank_htk(n_mels, f_min, f_max, n_stft, sr):
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


def logmel_pc(x):
    """复现板端 logmel.c: center 对齐 + reflect + Hann periodic + rfft + HTK mel + log."""
    w = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)).astype(np.float32)  # periodic hann
    fb = mel_filterbank_htk(N_MELS, 0.0, SR / 2.0, N_FFT // 2 + 1, SR).astype(np.float32)
    n = len(x)
    n_frames = 65
    mel = np.zeros((N_MELS, n_frames), dtype=np.float32)
    for t in range(n_frames):
        off = t * HOP - N_FFT // 2
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
    return np.log(mel + 1e-5)


def predict(lm):
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    logits = sess.run(None, {in_name: lm})[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    return int(probs.argmax()), probs


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "airport"
    npz = np.load("board_logmel.npy")  # [256,65]
    lm_board = npz.reshape(1, 1, N_MELS, 65).astype(np.float32)

    x = load_wav_32k(os.path.join(SAMPLES, f"{scene}.wav"))
    x16 = np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)
    xf = x16.astype(np.float32) / 32768.0
    lm_pc = logmel_pc(xf)[None, None].astype(np.float32)

    d = np.abs(lm_board - lm_pc)
    print(f"scene={scene}  logmel maxdiff={d.max():.4e} mean={d.mean():.4e}")

    # 分频段看差异 (mel 行)
    for lo, hi in [(0, 64), (64, 128), (128, 192), (192, 256)]:
        seg = d[0, 0, lo:hi]
        print(f"  mel[{lo:3d}:{hi:3d}] maxdiff={seg.max():.4e}")

    # 差异最大的 12 个点
    idx = np.argsort(d.ravel())[::-1][:12]
    for i in idx:
        m, f = np.unravel_index(i, d[0, 0].shape)
        print(f"  mel={m:3d} frame={f:3d} board={lm_board[0,0,m,f]:.4f} pc={lm_pc[0,0,m,f]:.4f} diff={d[0,0,m,f]:.4e}")

    p_board, prob_board = predict(lm_board)
    p_pc, prob_pc = predict(lm_pc)
    print(f"\nboard ONNX -> {CLASSES[p_board]} (期望 {scene})")
    print(f"pc    ONNX -> {CLASSES[p_pc]} (期望 {scene})")
    topb = np.argsort(prob_board)[::-1][:3]
    topp = np.argsort(prob_pc)[::-1][:3]
    print("  board top3:", " ".join(f"{CLASSES[i]}:{prob_board[i]:.3f}" for i in topb))
    print("  pc    top3:", " ".join(f"{CLASSES[i]}:{prob_pc[i]:.3f}" for i in topp))


if __name__ == "__main__":
    main()
