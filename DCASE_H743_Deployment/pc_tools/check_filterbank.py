#!/usr/bin/env python3
"""对比板端 mel_filter.c 的滤波器组与 torchaudio/HTK 密集滤波器组.

板端 mel_filter.c 移植 (逐行对齐):
  bin_pts[m] = f_pts[m] * N_FFT / SR
  start = floor(bin_pts[m]), center = round(bin_pts[m+1]), end = floor(bin_pts[m+2])
  上升沿 (bin<=center): (bin - bin_pts[m]) / (bin_pts[m+1] - bin_pts[m])
  下降沿 (bin>center):  (bin_pts[m+2] - bin) / (bin_pts[m+2] - bin_pts[m+1])
  权重裁剪到 [0,1]
"""
import numpy as np

SR, N_FFT, N_MELS = 32000, 4096, 256
F_MIN, F_MAX = 0.0, 16000.0
N_BINS = N_FFT // 2 + 1  # 2049


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def board_filterbank(center_round=True):
    """移植板端 mel_filter.c 的稀疏权重 -> 密集 [256, 2049].
    center_round=True : 原始实现 (峰值取整 -> 低频滤波器平顶错误)
    center_round=False: 修正版 (用分数峰值做上升/下降分支判断, 与 HTK 一致)
    """
    fb = np.zeros((N_MELS, N_BINS))
    mel_pts = np.array([hz_to_mel(F_MIN) + m * (hz_to_mel(F_MAX) - hz_to_mel(F_MIN)) / (N_MELS + 1)
                        for m in range(N_MELS + 2)])
    f_pts = mel_to_hz(mel_pts)
    bin_pts = f_pts * N_FFT / SR
    left_bin = np.floor(bin_pts).astype(int)
    right_bin = np.ceil(bin_pts).astype(int)

    for m in range(N_MELS):
        start = int(left_bin[m])
        center = int(round(bin_pts[m + 1]))
        end = int(left_bin[m + 2])
        if start < 0:
            start = 0
        if end >= N_BINS:
            end = N_BINS - 1
        if center < 0:
            center = 0
        if center >= N_BINS:
            center = N_BINS - 1
        if end < start:
            end = start
        for bin in range(start, end + 1):
            if (bin <= center) if center_round else (bin <= bin_pts[m + 1]):
                denom = bin_pts[m + 1] - bin_pts[m]
                if denom < 1e-6:
                    denom = 1e-6
                w = (bin - bin_pts[m]) / denom
            else:
                denom = bin_pts[m + 2] - bin_pts[m + 1]
                if denom < 1e-6:
                    denom = 1e-6
                w = (bin_pts[m + 2] - bin) / denom
            w = max(0.0, min(1.0, w))
            fb[m, bin] = w
    return fb


def htk_filterbank():
    """torchaudio MelScale 默认 (htk, norm=None) 的三角滤波器组."""
    m_min, m_max = hz_to_mel(F_MIN), hz_to_mel(F_MAX)
    m_pts = np.linspace(m_min, m_max, N_MELS + 2)
    f_pts = mel_to_hz(m_pts)
    freqs = np.linspace(0.0, SR / 2.0, N_BINS)
    fb = np.zeros((N_MELS, N_BINS))
    for k in range(N_MELS):
        lo, ctr, hi = f_pts[k], f_pts[k + 1], f_pts[k + 2]
        up = (freqs >= lo) & (freqs <= ctr)
        dn = (freqs >= ctr) & (freqs <= hi)
        if ctr > lo:
            fb[k, up] = (freqs[up] - lo) / (ctr - lo)
        if hi > ctr:
            fb[k, dn] = (hi - freqs[dn]) / (hi - ctr)
    return fb


def main():
    b_old = board_filterbank(center_round=True)
    b_new = board_filterbank(center_round=False)
    h = htk_filterbank()

    for tag, b in (("原始板端", b_old), ("修正版", b_new)):
        d = np.abs(b - h).max()
        nnz_b = np.count_nonzero(b)
        print(f"[{tag}] nnz={nnz_b}  global max abs diff vs HTK = {d:.6f}")
        if d > 1e-4:
            diffs = [np.abs(b[m] - h[m]).max() for m in range(N_MELS)]
            worst = np.argsort(diffs)[::-1][:5]
            for m in worst:
                print(f"   filter {m:3d}: maxdiff={diffs[m]:.6f}  "
                      f"rowsum board={b[m].sum():.4f} htk={h[m].sum():.4f}")
    print(f"\nHTK nnz: {np.count_nonzero(h)}")


if __name__ == "__main__":
    main()
