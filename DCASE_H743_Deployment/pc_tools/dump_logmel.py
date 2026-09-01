#!/usr/bin/env python3
"""板端 logmel + FFT 转储对比工具

用法:
  python dump_logmel.py COM19 [scene]

协议 (DUMP_MAGIC 5A A5 5A A5 + PCM 触发):
  [4B len] [FFT 快照 float32×4096] [4B len] [logmel float32×16640]
"""
import sys
import time
import numpy as np
import serial
from scipy.signal import resample_poly

SRC_DIR = "dataset/samples"
DUMP_MAGIC = bytes([0x5A, 0xA5, 0x5A, 0xA5])
PCM_LEN = 35072
WIN, HOP, N_FFT, SR, N_MELS = 3072, 500, 4096, 32000, 256
N_FRAMES = 65

CLASSES = ["airport", "bus", "metro", "metro_station", "park", "public_square",
           "shopping_mall", "street_pedestrian", "street_traffic", "tram"]


def load_wav_32k(path):
    import scipy.io.wavfile as wavfile
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    scale = 2147483648.0 if np.abs(data).max() > (1 << 24) else 8388608.0
    x = data.astype(np.float64) / scale
    x = resample_poly(x, SR, sr)
    if x.size < PCM_LEN:
        reps = int(np.ceil(PCM_LEN / x.size))
        x = np.tile(x, reps)
    return x[:PCM_LEN]


def recv_blob(ser, timeout_s=30):
    """读 4 字节长度头 + 数据"""
    hdr = b""
    t0 = time.time()
    while len(hdr) < 4:
        hdr += ser.read(4 - len(hdr))
        if time.time() - t0 > timeout_s:
            raise TimeoutError("等待长度头超时")
    n = int.from_bytes(hdr, "little")
    data = b""
    while len(data) < n:
        data += ser.read(n - len(data))
    return data


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM19"
    scene = sys.argv[2] if len(sys.argv) > 2 else "airport"
    if scene not in CLASSES:
        print(f"场景必须在 {CLASSES} 中")
        return

    x = load_wav_32k(f"{SRC_DIR}/{scene}.wav")
    pcm = np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2")

    ser = serial.Serial(port, 115200, timeout=2)
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.write(DUMP_MAGIC + pcm.tobytes())
    ser.flush()

    fft_data = recv_blob(ser)
    lm_data = recv_blob(ser)
    ser.close()

    fft_cap = np.frombuffer(fft_data, dtype="<f4").copy()          # [4096]
    lm_board = np.frombuffer(lm_data, dtype="<f4").copy().reshape(N_MELS, N_FRAMES)
    print(f"FFT 快照 {len(fft_data)} B, logmel {len(lm_data)} B [{N_MELS},{N_FRAMES}]")

    # ---- 与模拟第 0 帧对比: 先重建板端第 0 帧时域输入 ----
    hann = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)).astype(np.float32)
    seg = np.zeros(N_FFT, dtype=np.float32)
    for i in range(WIN):
        src = i - N_FFT // 2
        if src < 0:
            src = -src
        seg[i] = pcm[src] / 32768.0 * hann[i]
    spec_sim = np.fft.rfft(seg)
    pow_sim = np.abs(spec_sim) ** 2

    # 板端解析: DC@0, Nyquist@1, Re/Im 对
    pow_board = np.zeros(N_FFT // 2 + 1, dtype=np.float64)
    pow_board[0] = fft_cap[0] ** 2
    pow_board[N_FFT // 2] = fft_cap[1] ** 2
    for k in range(1, N_FFT // 2):
        pow_board[k] = fft_cap[2 * k] ** 2 + fft_cap[2 * k + 1] ** 2

    # 功率谱对比 (归一化后看形状, 排除整体缩放差异)
    pb = pow_board / pow_board.sum()
    ps = pow_sim / pow_sim.sum()
    d = np.abs(pb - ps)
    print(f"\n=== 帧0 功率谱对比 (归一化) ===")
    print(f"max rel diff = {d.max():.3e} @ bin={d.argmax()}")
    print(f"mean rel diff = {d.mean():.3e}")
    # 分频段
    for lo, hi, tag in ((0, 512, "0-4kHz"), (512, 1024, "4-8kHz"),
                        (1024, 1536, "8-12kHz"), (1536, 2049, "12-16kHz")):
        sub = d[lo:hi]
        bsub = pb[lo:hi]
        ssub = ps[lo:hi]
        print(f"  {tag}: board能量占比={bsub.sum():.4f} sim占比={ssub.sum():.4f} "
              f"maxdiff={sub.max():.3e}")

    # 高频段逐 bin 看
    print("\n高频 bin 对比 (原始功率, 前12个差异最大):")
    hb = pow_board[1536:]
    hs = pow_sim[1536:]
    order = np.argsort(np.abs(hb - hs))[::-1][:12]
    for k in order:
        kb = 1536 + k
        print(f"  bin {kb:4d}: board={hb[k]:.4e} sim={hs[k]:.4e}")

    np.save("board_logmel.npy", lm_board)
    np.save("board_fft_cap.npy", fft_cap)
    print("\n已保存 board_logmel.npy / board_fft_cap.npy")


if __name__ == "__main__":
    main()
